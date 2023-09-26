
#include "ImageBufferConverter.h"
namespace SNN
{
    ImageBufferConverter::ImageBufferConverter(OpenCLRuntime *opencl_runtime)
    {
        this->mOpenCLRuntime = opencl_runtime;
    }
    ImageBufferConverter::~ImageBufferConverter()
    {
    }
    bool ImageBufferConverter::ConvertImagetoBuffer(std::shared_ptr<Tensor> tensor, const OpenCLBufferFormat type, bool needwait, const std::string &buildOption)
    {
        SNN_ASSERT(tensor->GetOpType() == DepthwiseConv);

        std::string kernelName;
        switch (type)
        {
        case CONV2D_FILTER:
            kernelName = "conv2d_filter_buffer_to_image";
            break;
        case CONV2D1x1_OPT_FILTER:
            kernelName = "conv2d1x1_opt_filter_buffer_to_image";
            break;
        case DW_CONV2D_FILTER:
            kernelName = "dw_filter_buffer_to_image";
            break;
        case NHWC_BUFFER:
            kernelName = "nhwc_buffer_to_image";
            break;
        case NCHW_BUFFER:
            kernelName = "nchw_buffer_to_image";
            break;
        default:
            break;
        }
        std::set<std::string> buildOptions;
        buildOptions.emplace(buildOption);
        mBufferToImageKernelName = kernelName;
        mBufferToImageKernel = this->mOpenCLRuntime->BuildKernel("buffer_to_image", kernelName, buildOptions);
        cl_context &GPUcontext = this->mOpenCLRuntime->GetGPUContext();
        cl_device_id *device = this->mOpenCLRuntime->GetDevice();
        cl_command_queue *commandQueue = this->mOpenCLRuntime->GetCommandQue();
        cl_image_format clImageFormat;
        clImageFormat.image_channel_order = CL_RGBA;
        clImageFormat.image_channel_data_type = CL_FLOAT;
        size_t imageShape[2] = {}, maxWorkGroupSize = 0, idx = 0;
        cl_int err;
        clGetKernelWorkGroupInfo(mBufferToImageKernel, *device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, NULL);
        const std::vector<uint8_t> &kernelDims = tensor->KernelShape();
        int kernel_dims_arr[4];
        std::copy(kernelDims.begin(), kernelDims.end(), kernel_dims_arr);
        getImageShape(kernelDims, type, imageShape);
        int filterMultiplier = kernelDims[0], filterHeight = kernelDims[1],
            filterWidth = kernelDims[2], filterOutputChennel = kernelDims[3];
        std::vector<size_t> gws{imageShape[0], imageShape[1]};
        const std::shared_ptr<std::vector<std::pair<float *, float *>>> mainMemory = tensor->GetMainMemory();
        const std::vector<uint8_t> &ptrIndex = tensor->GetMemoryPtrIndex();
        std::pair<float *, float *> &weight_bias = mainMemory->at(ptrIndex[0]);
        float *weightData = weight_bias.first;
        float *biasData = weight_bias.second;
        cl_mem tmpBuffer = clCreateBuffer(GPUcontext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          sizeof(float) * imageShape[0], weightData, &err);
        oclCheckError(err, CL_SUCCESS);
        // please follow document filterHeight * filterWidth * multiplierChannel, (outputChennel + 3) / 4
        cl_mem mFilter = clCreateImage2D(GPUcontext, CL_MEM_READ_WRITE, &clImageFormat, filterHeight * filterWidth * filterMultiplier, UP_DIV(filterOutputChennel, 4), 0, NULL, &err);
        oclCheckError(err, CL_SUCCESS);
        const int heightWidthSumSize = kernelDims[1] * kernelDims[2];
        err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(int), &gws[0]);
        err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(int), &gws[1]);
        err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(cl_mem), &tmpBuffer);
        err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(kernel_dims_arr), kernel_dims_arr);
        err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(int), &heightWidthSumSize);
        err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(cl_mem), &mFilter);
        oclCheckError(err, CL_SUCCESS);
        const size_t lws[2] = {16, MAX((unsigned int)1, maxWorkGroupSize / 16)};
        size_t roundUpGroupWorkSize[2] = {ROUND_UP(gws[0], lws[0]), ROUND_UP(gws[1], lws[1])};
        err |= clEnqueueNDRangeKernel(commandQueue[0], mBufferToImageKernel, 1, NULL, roundUpGroupWorkSize, lws, 0, NULL, NULL);
        oclCheckError(err, CL_SUCCESS);
        err |= clFinish(commandQueue[0]);
        oclCheckError(err, CL_SUCCESS);
        tensor->SetDeviceFilter(mFilter);
        return true;
    }
    bool ImageBufferConverter::ConvertBuffertoImage(Tensor *inputs) {}

}