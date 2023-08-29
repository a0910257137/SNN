
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
    bool ImageBufferConverter::ConvertImagetoBuffer(Tensor *inputs, const OpenCLBufferFormat type, bool needwait, const std::string &buildOption)
    {

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
        size_t imageShape[2] = {}, maxWorkGroupSize = 0;
        size_t idx = 0;
        cl_int err;
        clGetKernelWorkGroupInfo(mBufferToImageKernel, *device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, NULL);
        if (inputs->op_type == DepthwiseConv)
        {
            cl_mem input_buffer;
            DepthwiseConvParams *params = (DepthwiseConvParams *)inputs->op_data;
            getImageShape(params->kernel_dims, type, imageShape);
            int filterMultiplier = params->kernel_dims[0], filterHeight = params->kernel_dims[1],
                filterWidth = params->kernel_dims[2], filterOutputChennel = params->kernel_dims[3];
            size_t gws[2][1] = {{static_cast<size_t>(imageShape[0])}, {static_cast<size_t>(imageShape[1])}};
            input_buffer = clCreateBuffer(GPUcontext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          sizeof(float) * imageShape[0], params->weights, &err);
            oclCheckError(err, CL_SUCCESS);
            // please follow document filterHeight * filterWidth * multiplierChannel, (outputChennel + 3) / 4
            inputs->image = clCreateImage2D(GPUcontext, CL_MEM_WRITE_ONLY, &clImageFormat, filterHeight * filterWidth * filterMultiplier, (filterOutputChennel + 3) / 4, 0, NULL, &err);
            oclCheckError(err, CL_SUCCESS);
            const int heightWidthSumSize = params->kernel_dims[1] * params->kernel_dims[2];
            err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(int), gws[0]);
            err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(int), gws[1]);
            err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(cl_mem), &input_buffer);
            err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(params->kernel_dims), params->kernel_dims);
            err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(int), &heightWidthSumSize);
            err |= clSetKernelArg(mBufferToImageKernel, idx++, sizeof(cl_mem), &inputs->image);
            oclCheckError(err, CL_SUCCESS);
            const size_t lws[2] = {16, MAX((unsigned int)1, maxWorkGroupSize / 16)};
            size_t roundUpGroupWorkSize[2] = {ROUND_UP(gws[0][0], lws[0]), ROUND_UP(gws[1][0], lws[1])};
            err |= clEnqueueNDRangeKernel(commandQueue[0], mBufferToImageKernel, 1, NULL, roundUpGroupWorkSize, lws, 0, NULL, NULL);
            oclCheckError(err, CL_SUCCESS);
            err |= clFinish(commandQueue[0]);
            oclCheckError(err, CL_SUCCESS);
        }

        // output cl buffer
        return true;
    }
    bool ImageBufferConverter::ConvertBuffertoImage(Tensor *inputs) {}

}