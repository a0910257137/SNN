#include "DepthwiseConvExecution.h"
namespace SNN
{
    DepthwiseConvExecution::DepthwiseConvExecution(OpenCLBackend *mbackend)
    {
        mCLRuntime = mbackend->CLRuntime();
        mConvCommon = std::make_shared<ConvolutionCommon>();
    }
    DepthwiseConvExecution::~DepthwiseConvExecution()
    {
    }
    bool DepthwiseConvExecution::onInit(Tensor *inputs)
    {
        ImageBufferConverter imageBufferConvertor{mCLRuntime};
        params = (DepthwiseConvParams *)inputs->op_data;

        mStrides[0] = params->stride_height, mStrides[1] = params->stride_width;

        mDilations[0] = params->dilation_height_factor, mDilations[1] = params->dilation_width_factor;

        int kernelMultiplier = params->kernel_dims[0], kernelHeight = params->kernel_dims[1],
            kernelWidth = params->kernel_dims[2], kernelOutputChannel = params->kernel_dims[3];
        oclCheckError(mStrides[0] > 0 && mStrides[1] > 1, true);
        int buffer_size = kernelMultiplier * kernelHeight * kernelWidth * kernelOutputChannel;
        std::string buildOption = "";
        if (mCLRuntime->isWeightCpuTransHalf())
        {
            buffer_size *= sizeof(cl_half);
        }
        else
        {
            buffer_size *= sizeof(float);
            buildOption = "-DBUFFER_INP_FP32";
        }
        bool status = imageBufferConvertor.ConvertImagetoBuffer(inputs, DW_CONV2D_FILTER, false, buildOption);
        oclCheckError(status, true);
        std::set<std::string> buildOptions;
        std::string kernelName = "depthwise_conv2d";
        if (mStrides[0] == 1 && mStrides[1] == 1 &&
            mDilations[0] == 1 && mDilations[1] == 1)
            kernelName = "depthwise_conv2d_s1";
        if (params->activation == kActRelu)
            buildOptions.emplace("-DRELU");
        else if (params->activation == kActRelu6)
            buildOptions.emplace("-DRELU6");
        mKernel = mCLRuntime->BuildKernel("depthwise_conv2d", kernelName, buildOptions);
        mMaxWorkGroupSize = static_cast<uint32_t>(mCLRuntime->getMaxWorkGroupSize(mKernel));
    }
    bool DepthwiseConvExecution::onResize(Tensor *inputs, int *input_shape, int *output_shape)
    {
        mGlobalWorkSize[0] = UP_DIV(output_shape[3], 4) * UP_DIV(output_shape[2], 4);
        mGlobalWorkSize[1] = output_shape[0] * output_shape[1];
        // determine padding
        mPaddings[0] = params->padding; // padY
        mPaddings[1] = params->padding; // padX
    }
    bool DepthwiseConvExecution::onExecute()
    {
    }
}