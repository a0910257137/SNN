
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
    bool ImageBufferConverter::ConvertBufferToImage(std::shared_ptr<Tensor> tensor, const OpenCLBufferFormat type, bool needwait, const std::string &buildOption)
    {
        SNN_ASSERT((tensor->GetOpType() == DEPTHWISECONV2D) ||
                   (tensor->GetOpType() == CONV2D) ||
                   (tensor->GetOpType() == DECONV2D));
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
        cl_context *GPUcontext = this->mOpenCLRuntime->GetGPUContext();
        cl_device_id *device = this->mOpenCLRuntime->GetDevice();
        cl_command_queue *commandQueue = this->mOpenCLRuntime->GetCommandQue();
        cl_image_format clImageFormat;
        clImageFormat.image_channel_order = CL_RGBA;
        clImageFormat.image_channel_data_type = CL_FLOAT;
        size_t imageShape[2] = {}, maxWorkGroupSize = 0, idx = 0;
        cl_int err = 0;
        clGetKernelWorkGroupInfo(mBufferToImageKernel, *device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, NULL);
        const std::vector<int> &kernelShape = tensor->KernelShape();
        getImageShape(kernelShape, type, imageShape);
        int filterOutputChannel = kernelShape[0], filterInputChannel = kernelShape[1], filterHeight = kernelShape[2], filterWidth = kernelShape[3];
        std::vector<size_t> gws{imageShape[0], imageShape[1]};
        const std::shared_ptr<std::vector<std::pair<float *, float *>>> mainMemory = tensor->GetMainMemory();
        const std::vector<uint8_t> &ptrIndex = tensor->GetMemoryPtrIndex();
        std::pair<float *, float *> &weight_bias = mainMemory->at(ptrIndex[0]);
        float *weightData = weight_bias.first, *biasData = weight_bias.second;
        const int weight_bytes = tensor->weight_bytes();
        int elementSize = kernelShape[0] * kernelShape[1] * kernelShape[2] * kernelShape[3], buffer_size;
        // const int bias_bytes = tensor->bias_bytes();
        if (tensor->GetOpType() != DECONV2D)
        {
            if (weightData == nullptr || biasData == nullptr)
            {
                printf("ERROR: %s has no weights or bias. \n", tensor->GetOpName().c_str());
                return false;
            }
        }
        else
        {
            if (weightData == nullptr)
            {
                printf("ERROR: Deconv 2D has no weights. \n");
                return false;
            }
        }
        if (this->mOpenCLRuntime->isWeightCpuTransHalf())
        {
            buffer_size = elementSize * sizeof(cl_half);
        }
        else
        {
            buffer_size = elementSize * sizeof(float);
        }
        // deal with opencl
        cl_mem tmpBuffer;

        err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(int), &gws[0]);
        err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(int), &gws[1]);
        oclCheckError(err, CL_SUCCESS);
        if (type == CONV2D_FILTER)
        {
            const int channelHeightWidthSumSize =
                kernelShape[1] * kernelShape[2] * kernelShape[3];
            const int heightWidthSumSize = kernelShape[2] * kernelShape[3];
            int kernel_dims_arr[2] = {kernelShape[2], kernelShape[3]};
            tmpBuffer = clCreateBuffer(*GPUcontext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, buffer_size, NULL, &err);
            oclCheckError(err, CL_SUCCESS);
            float *ptrCL = (float *)clEnqueueMapBuffer(commandQueue[0], tmpBuffer, true, CL_MAP_WRITE, 0, buffer_size, 0, NULL, NULL, &err);
            oclCheckError(err, CL_SUCCESS);
            if (ptrCL != nullptr && err == CL_SUCCESS)
            {
                if (mOpenCLRuntime->isWeightCpuTransHalf())
                {
                    for (int i = 0; i < elementSize; ++i)
                    {
                        ((cl_half *)ptrCL)[i] = (cl_half)(weightData[i]);
                    }
                }
                else
                {
                    memset(ptrCL, 0.0f, buffer_size);
                    memcpy(ptrCL, weightData, buffer_size);
                }
            }
            else
                printf("ERROR: Map memory error in biasPtrCL !! \n");
            err = clEnqueueUnmapMemObject(commandQueue[0], tmpBuffer, ptrCL, 0, NULL, NULL);
            oclCheckError(err, CL_SUCCESS);
            err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(cl_mem), &tmpBuffer);
            err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(int), &kernelShape[0]);
            err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(kernel_dims_arr), kernel_dims_arr);
            err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(int), &channelHeightWidthSumSize);
            err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(int), &heightWidthSumSize);
            oclCheckError(err, CL_SUCCESS);
        }
        else if (type == DW_CONV2D_FILTER)
        {
            const int heightWidthSumSize = kernelShape[2] * kernelShape[3];
            int kernel_dims_arr[4] = {kernelShape[0], kernelShape[1], kernelShape[2], kernelShape[3]};
            cl_mem tmpBuffer = clCreateBuffer(*GPUcontext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              buffer_size, weightData, &err);
            err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(cl_mem), &tmpBuffer);
            err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(kernel_dims_arr), kernel_dims_arr);
            err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(int), &heightWidthSumSize);
        }
        else if (type == ARGUMENT)
        {
            err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(int), &kernelShape[0]);
        }
        else if (type == CONV2D1x1_OPT_FILTER)
        {
            const int channelHeightWidthSumSize = kernelShape[1] * kernelShape[2] * kernelShape[3];
            const int heightWidthSumSize = kernelShape[2] * kernelShape[3];
            int kernel_dims_arr[2] = {kernelShape[2], kernelShape[3]};
            err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(int), &kernelShape[1]);
            err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(kernel_dims_arr), kernel_dims_arr);
            err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(int), &channelHeightWidthSumSize);
            err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(int), &heightWidthSumSize);
        }
        else
        {
            err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(int), &kernelShape[1]);
            err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(int), &kernelShape[2]);
            err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(int), &kernelShape[3]);
        }
        oclCheckError(err, CL_SUCCESS);
        cl_mem mFilter = clCreateImage2D(*GPUcontext, CL_MEM_READ_WRITE, &clImageFormat, imageShape[0], imageShape[1], 0, NULL, &err);
        err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(cl_mem), &mFilter);
        const size_t lws[2] = {16, MAX((unsigned int)1, maxWorkGroupSize / 16)};
        size_t roundUpGroupWorkSize[2] = {ROUND_UP(gws[0], lws[0]), ROUND_UP(gws[1], lws[1])};
        err |= clEnqueueNDRangeKernel(commandQueue[0], mBufferToImageKernel, 2, NULL, roundUpGroupWorkSize, lws, 0, NULL, NULL);
        err |= clFinish(commandQueue[0]);
        oclCheckError(err, CL_SUCCESS);
        tensor->SetDeviceFilter(mFilter);
        return true;
    }

    float *ImageBufferConverter::ConvertImageToNHWCBuffer(std::shared_ptr<Tensor> tensor, cl_kernel &imageToBufferKernel,
                                                          OpenCLRuntime *runtime, bool needWait, bool svmFlag)
    {

        // const std::vector<std::vector<int>> &inputShapes = tensor->InputShape();
        // SNN_ASSERT(inputShapes.size() == 1);
        // const std::vector<int> &inputShape = inputShapes[0];
        const std::vector<int> &outputShape = tensor->OutputShape();
        int in_gws[2] = {static_cast<int>(UP_DIV(outputShape[3], 4)) * outputShape[2],
                         static_cast<int>(outputShape[0] * outputShape[1])};
        if (imageToBufferKernel == NULL)
        {
            std::set<std::string> buildOptions;
            buildOptions.emplace("-DBUFFER_IMAGE_IO_TRANS");
            imageToBufferKernel = runtime->BuildKernel("buffer_to_image", "image_to_nhwc_buffer", buildOptions);
        }
        int buffer_sizes = outputShape[0] * outputShape[1] * outputShape[2] * outputShape[3] * sizeof(float);
        uint32_t idx = 0;
        cl_int err = 0;
        cl_context *GPUcontext = runtime->GetGPUContext();
        cl_command_queue *commandQueue = runtime->GetCommandQue();
        const cl_mem *mDeviceImage = tensor->GetDeviceOutputData();
        float *h_data = (float *)malloc(buffer_sizes);
        cl_mem mhostBuffer = clCreateBuffer(*GPUcontext, CL_MEM_READ_WRITE, buffer_sizes, NULL, &err);
        oclCheckError(err, CL_SUCCESS);
        err |= clSetKernelArg(imageToBufferKernel, idx++, sizeof(int), &in_gws[0]);
        err |= clSetKernelArg(imageToBufferKernel, idx++, sizeof(int), &in_gws[1]);
        err |= clSetKernelArg(imageToBufferKernel, idx++, sizeof(cl_mem), &mhostBuffer);
        err |= clSetKernelArg(imageToBufferKernel, idx++, sizeof(int), &outputShape[1]);
        err |= clSetKernelArg(imageToBufferKernel, idx++, sizeof(int), &outputShape[2]);
        err |= clSetKernelArg(imageToBufferKernel, idx++, sizeof(int), &outputShape[3]);
        err |= clSetKernelArg(imageToBufferKernel, idx++, sizeof(cl_mem), mDeviceImage);
        oclCheckError(err, CL_SUCCESS);
        const uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLRuntime->getMaxWorkGroupSize(imageToBufferKernel));
        const size_t lws[2] = {16, MAX((size_t)1, maxWorkGroupSize / 16)};
        cl_event event;
        size_t roundUpGroupWorkSize[2] = {ROUND_UP(in_gws[0], lws[0]), ROUND_UP(in_gws[1], lws[1])};
        err |= clEnqueueNDRangeKernel(commandQueue[0], imageToBufferKernel, 2, NULL, roundUpGroupWorkSize, lws, 0, NULL, &event);
        oclCheckError(err, CL_SUCCESS);
        err |= clEnqueueReadBuffer(commandQueue[0], mhostBuffer, CL_TRUE, 0, buffer_sizes, h_data, 0, NULL, NULL);
        if (needWait == true)
        {
            clWaitForEvents(1, &event);
        }
        err |= clFinish(commandQueue[0]);
        oclCheckError(err, CL_SUCCESS);
        clReleaseMemObject(mhostBuffer);
        return h_data;
    }
}