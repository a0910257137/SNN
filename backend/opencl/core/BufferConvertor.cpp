#include "BufferConvertor.h"
namespace SNN
{
    BufferConvertor::BufferConvertor(OpenCLRuntime *opencl_runtime)
    {
        this->mOpenCLRuntime = opencl_runtime;
    }
    BufferConvertor::~BufferConvertor()
    {
    }
    bool BufferConvertor::convertToNC4HW4Buffer(std::shared_ptr<Tensor> tensor, const OpenCLBufferFormat type, bool needTrans, bool needWait, bool lowMemory, int quantBit)
    {
        SNN_ASSERT((tensor->GetOpType() == DEPTHWISECONV2D) || (tensor->GetOpType() == CONV2D) || (tensor->GetOpType() == DECONV2D));
        std::string kernelName;
        cl_context *GPUcontext = mOpenCLRuntime->GetGPUContext();
        cl_device_id *device = mOpenCLRuntime->GetDevice();
        cl_command_queue *commandQueue = mOpenCLRuntime->GetCommandQue();
        const std::shared_ptr<std::vector<std::pair<float *, float *>>> mainMemory = tensor->GetMainMemory();
        const std::vector<uint8_t> &ptrIndex = tensor->GetMemoryPtrIndex();
        std::pair<float *, float *> &weight_bias = mainMemory->at(ptrIndex[0]);
        float *weightData = weight_bias.first, *biasData = weight_bias.second;
        const int weight_bytes = tensor->weight_bytes();
        cl_int err = CL_SUCCESS;
        size_t imageShape[2], gws[2];
        std::vector<int> filterBufferShape, filterImageShape;
        cl_mem filterCLBuffer, filterCLImage;
        switch (type)
        {
        case CONV2D_FILTER:
        {

            // make image buffer

            const std::vector<int> &kernelShape = tensor->KernelShape();
            int mOutputChannel = kernelShape[0], mInputChannel = kernelShape[1], mKernelHeight = kernelShape[2], mKernelWidth = kernelShape[3];
            filterBufferShape = {mOutputChannel, ROUND_UP(mInputChannel, 4), mKernelHeight, mKernelWidth};
            filterImageShape = {ROUND_UP(mInputChannel, 4), UP_DIV(mOutputChannel, 4) * mKernelHeight * mKernelWidth};
            int bufferSize = filterBufferShape[0] * filterBufferShape[1] * filterBufferShape[2] * filterBufferShape[3];
            if (mOpenCLRuntime->isSupportedFP16() && mOpenCLRuntime->isWeightCpuTransHalf())
            {
                bufferSize *= sizeof(cl_half);
            }
            else
            {
                bufferSize *= sizeof(float);
            }

            filterCLBuffer = clCreateBuffer(*GPUcontext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, bufferSize, NULL, &err);
            filterCLImage = clCreateBuffer(*GPUcontext, CL_MEM_READ_WRITE, bufferSize, NULL, &err);
            auto ptrCL = clEnqueueMapBuffer(commandQueue[0], filterCLBuffer, true, CL_MAP_WRITE, 0, bufferSize, 0, NULL, NULL, &err);
            if (ptrCL != nullptr && err == CL_SUCCESS)
            {
                ::memset(ptrCL, 0, bufferSize);
                if (mOpenCLRuntime->isWeightCpuTransHalf())
                {
                    for (int oc = 0; oc < mOutputChannel; oc++)
                    {
                        for (int ic = 0; ic < mInputChannel; ic++)
                        {
                            for (int kh = 0; kh < mKernelHeight; kh++)
                            {
                                for (int kw = 0; kw < mKernelWidth; kw++)
                                {
                                    int dst_idx = ((oc * ROUND_UP(mInputChannel, 4) + ic) * mKernelHeight + kh) * mKernelWidth + kw;
                                    int src_idx = ((oc * mInputChannel + ic) * mKernelHeight + kh) * mKernelWidth + kw;

                                    ((cl_half *)ptrCL)[dst_idx] = (cl_half)(weightData[src_idx]);
                                }
                            }
                        }
                    }
                }
                else
                {
                    const int copy_size = mKernelWidth * mKernelHeight * sizeof(float);
                    for (int oc = 0; oc < mOutputChannel; ++oc)
                    {
                        for (int ic = 0; ic < mInputChannel; ++ic)
                        {
                            ::memcpy((float *)ptrCL + (oc * ROUND_UP(mInputChannel, 4) + ic) * mKernelWidth * mKernelHeight, weightData + (oc * mInputChannel + ic) * mKernelWidth * mKernelHeight, copy_size);
                        }
                    }
                }
            }
            else
            {
                SNN_ERROR("ERROR: Map error ptrCL == nullptr \n");
            }

            auto formattedBufferShape = TensorShapeFormat(filterBufferShape, DATA_FORMAT_NHWC);
            getImageShape(formattedBufferShape, type, imageShape);
            gws[0] = imageShape[0];
            gws[1] = imageShape[1];
            kernelName = "conv2d_filter_buffer_to_nc4hw4_buffer"; // NC4HW4 (1, 4*ic/4, kw*kh*oc/4, 1)*4
            break;
        }
        case DW_CONV2D_FILTER:
        {
            kernelName = "dw_filter_buffer_to_nc4hw4_buffer"; // NC4HW4 (1, kw*kh, oc/4, 1)*4
            break;
        }

        case NHWC_BUFFER:
        case NCHW_BUFFER:
        case ARGUMENT:
            break;
        default:
            break;
        }

        if (mBufferToImageKernel == NULL || mBufferToImageKernelName != kernelName)
        {
            mBufferToImageKernelName = kernelName;
            std::set<std::string> buildOptions;
            if (needTrans)
            {
                // buildOptions.emplace("-DBUFFER_FORMAT_INP_TRANS");
                kernelName += "_floatin";
            }
            mBufferToImageKernel = mOpenCLRuntime->BuildKernel("buffer_convert_buf", kernelName, buildOptions);
        }
        size_t idx = 0;
        err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(int), &gws[0]);
        err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(int), &gws[1]);
        err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(cl_mem), &filterCLBuffer);
        if (type == CONV2D_FILTER)
        {
            const int channelHeightWidthSumSize = filterBufferShape[1] * filterBufferShape[2] * filterBufferShape[3];
            const int heightWidthSumSize = filterBufferShape[2] * filterBufferShape[3];
            int kernelShape[2] = {filterBufferShape[2], filterBufferShape[3]};
            err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(int), &filterBufferShape[0]);
            err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(kernelShape), &kernelShape);
            err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(channelHeightWidthSumSize), &channelHeightWidthSumSize);
            err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(heightWidthSumSize), &heightWidthSumSize);
        }
        else if (type == DW_CONV2D_FILTER)
        {
        }
        else
        {
            SNN_ERROR("convertToNC4HW4Buffer type not support!\n");
        }
        err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(cl_mem), &filterCLImage);
        oclCheckError(err, CL_SUCCESS);
        size_t maxWorkGroupSize;
        clGetKernelWorkGroupInfo(mBufferToImageKernel, *device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, NULL);
        size_t lws[2] = {16, MAX((unsigned int)1, maxWorkGroupSize / 16)};
        cl_event event;
        size_t roundUpGroupWorkSize[2];
        for (size_t i = 0; i < 2; ++i)
        {
            roundUpGroupWorkSize[i] = ROUND_UP(gws[i], lws[i]);
        }
        err |= clEnqueueNDRangeKernel(commandQueue[0], mBufferToImageKernel, 2, NULL, roundUpGroupWorkSize, lws, 0, NULL, &event);
        err |= clFinish(commandQueue[0]);
        err |= clReleaseMemObject(filterCLBuffer);
        oclCheckError(err, CL_SUCCESS);
        tensor->SetDeviceFilter(filterCLImage);
        return true;
    }
} // namespace SNN
