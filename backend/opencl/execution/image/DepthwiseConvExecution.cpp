#include "DepthwiseConvExecution.h"
#include "backend/opencl/core/runtime/OpenCLRuntime.h"
namespace SNN
{
    DepthwiseConvExecution::DepthwiseConvExecution(std::shared_ptr<Tensor> tensor, OpenCLBackend *mbackend) : ConvBaseExecution(tensor, mbackend)
    {

        this->mbackend = mbackend;
        mConvCommon = std::make_shared<ConvolutionCommon>();
        ImageBufferConverter imageBufferConvertor{mOpenCLRuntime};
        mStrides[0] = tensor->stride(0), mStrides[1] = tensor->stride(0);
        mDilations[0] = tensor->dilation(0), mDilations[1] = tensor->dilation(1);
        std::vector<uint8_t> kernelDims = tensor->KernelShape();
        uint8_t kernelMultiplier = kernelDims[0],
                kernelHeight = kernelDims[1],
                kernelWidth = kernelDims[2], kernelOutputChannel = kernelDims[3];
        oclCheckError(mStrides[0] > 0 && mStrides[1] > 1, true);
        int buffer_size = static_cast<int>(kernelMultiplier * kernelHeight * kernelWidth * kernelOutputChannel);
        std::string buildOption = "";
        if (mOpenCLRuntime->isWeightCpuTransHalf())
        {
            buffer_size *= sizeof(cl_half);
        }
        else
        {
            buffer_size *= sizeof(float);
            buildOption = "-DBUFFER_INP_FP32";
        }
        bool status = imageBufferConvertor.ConvertImagetoBuffer(tensor, DW_CONV2D_FILTER, false, buildOption);
        oclCheckError(status, true);
        std::set<std::string> buildOptions;
        std::string kernelName = "depthwise_conv2d";
        if (mStrides[0] == 1 && mStrides[1] == 1 &&
            mDilations[0] == 1 && mDilations[1] == 1)
            kernelName = "depthwise_conv2d_s1";
        if (tensor->GetActType() == kActRelu)
            buildOptions.emplace("-DRELU");
        else if (tensor->GetActType() == kActRelu6)
            buildOptions.emplace("-DRELU6");
        mKernel = mOpenCLRuntime->BuildKernel("depthwise_conv2d", kernelName, buildOptions);
        mMaxWorkGroupSize = static_cast<size_t>(mOpenCLRuntime->getMaxWorkGroupSize(mKernel));
    }
    DepthwiseConvExecution::~DepthwiseConvExecution()
    {
    }
    bool DepthwiseConvExecution::onResize(std::shared_ptr<Tensor> tensor)
    {
        const std::vector<int> &inputShape = tensor->InputShape();
        const std::vector<int> &outputShape = tensor->OutputShape();
        const std::vector<uint8_t> &kernelVectShape = tensor->KernelShape();
        mGlobalWorkSize[0] = UP_DIV(outputShape[3], 4) * UP_DIV(outputShape[2], 4);
        mGlobalWorkSize[1] = outputShape[0] * outputShape[1];
        int inputImageShape[2] = {inputShape[1], inputShape[2]};
        int outputImageShape[2] = {outputShape[1], outputShape[2]};

        int strideShape[2] = {static_cast<int>(tensor->stride(0)), static_cast<int>(tensor->stride(1))};
        int kernelShape[2] = {static_cast<int>(kernelVectShape[1]), static_cast<int>(kernelVectShape[2])};
        int dilationShape[2] = {static_cast<int>(tensor->dilation(0)), static_cast<int>(tensor->dilation(1))};
        auto padding = mConvCommon->GetPadding(tensor);
        /*---------------------------------------------------*/
        int paddingShape[2] = {padding.first, padding.second};
        const int inputChannels = inputShape[3];
        const int inputChannelBlocks[1] = {UP_DIV(inputChannels, 4)};
        std::string kernelName = "depthwise_conv2d_s1";
        cl_int err = 0;
        uint32_t idx = 0;
        cl_context &GPUcontext = mOpenCLRuntime->GetGPUContext();
        cl_device_id *device = mOpenCLRuntime->GetDevice();
        cl_command_queue *commandQueue = mOpenCLRuntime->GetCommandQue();
        cl_image_format clImageFormat;
        clImageFormat.image_channel_order = CL_RGBA;
        clImageFormat.image_channel_data_type = CL_FLOAT;
        cl_mem outputCLData = clCreateImage2D(GPUcontext, CL_MEM_WRITE_ONLY, &clImageFormat, outputShape[1], outputShape[2], 0, NULL, &err);
        const cl_mem &mFilter = tensor->GetDeviceFilter();
        const cl_mem &mBias = tensor->GetDeviceBias();
        this->mbackend->CopyToDevice(tensor.get());
        const cl_mem &inputCLData = tensor->GetDeviceInputData();

        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGlobalWorkSize[0]);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGlobalWorkSize[1]);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &inputCLData);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &mFilter);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &mBias);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &outputCLData);
        err |= clSetKernelArg(mKernel, idx++, sizeof(inputImageShape), inputImageShape);
        err |= clSetKernelArg(mKernel, idx++, sizeof(inputChannelBlocks), inputChannelBlocks);
        err |= clSetKernelArg(mKernel, idx++, sizeof(outputImageShape), outputImageShape);
        err |= clSetKernelArg(mKernel, idx++, sizeof(kernelShape), kernelShape);
        err |= clSetKernelArg(mKernel, idx++, sizeof(paddingShape), paddingShape);
        oclCheckError(err, CL_SUCCESS);
        if (mStrides[0] != 1 || mStrides[1] != 1 || mDilations[0] != 1 || mDilations[1] != 1)
        {
            err |= clSetKernelArg(mKernel, idx++, sizeof(dilationShape), dilationShape);
            err |= clSetKernelArg(mKernel, idx++, sizeof(strideShape), strideShape);
            kernelName = "depthwise_conv2d";
        }
        oclCheckError(err, CL_SUCCESS);
        mLocalWorkSize = mOpenCLRuntime->localWS2DDefault(mGlobalWorkSize, mMaxWorkGroupSize, mOpenCLRuntime, kernelName, mKernel).first;
    }
    bool DepthwiseConvExecution::onExecute()
    {
    }
}