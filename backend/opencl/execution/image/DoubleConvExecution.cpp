#include "DoubleConvExecution.h"
namespace SNN
{
    DoubleConvExecution::DoubleConvExecution(std::vector<std::shared_ptr<Tensor>> &tensors, OpenCLBackend *mbackend) : Execution(mbackend)
    {
        mConvCommon = std::make_shared<ConvolutionCommon>();
        numTensors = tensors.size();

        kernelName;
        if ((numTensors == 2) && (tensors[0]->GetOpType() == CONV2D) && (tensors[1]->GetOpType() == CONV2D))
        {

            kernelName = "double_conv_2d_c4h1w4";
            mBuildOptions.emplace("-DDW1_TEMP8");
            mBuildOptions.emplace("-DCONV2_TEMP8");
            mBuildOptions.emplace("-DBIAS");
            mBuildOptions.emplace("-DCONV_S1D1");
            if (tensors[0]->GetActType() == kActRelu)
                mBuildOptions.emplace("-DCONV1_RELU");
            else if (tensors[0]->GetActType() == kActRelu6)
                mBuildOptions.emplace("-DCONV1_RELU6");
            else if (tensors[0]->GetActType() == kActSigmoid)
                mBuildOptions.emplace("-DCONV1_SIGMOID");
            if (tensors[0]->GetActType() == kActRelu)
                mBuildOptions.emplace("-DCONV2_RELU");
            else if (tensors[0]->GetActType() == kActRelu6)
                mBuildOptions.emplace("-DCONV2_RELU6");
            else if (tensors[0]->GetActType() == kActSigmoid)
                mBuildOptions.emplace("-DCONV2_SIGMOID");
        }
        mKernel = mOpenCLRuntime->BuildKernel("optimization", kernelName, mBuildOptions);
        mMaxWorkGroupSize = static_cast<size_t>(mOpenCLRuntime->getMaxWorkGroupSize(mKernel));
    }
    bool DoubleConvExecution::onOptimizedResize(std::vector<std::shared_ptr<Tensor>> &tensors)
    {
        cl_int err = 0;
        uint32_t idx = 0;
        SNN_ASSERT(tensors.size() == 2);

        const std::vector<std::vector<int>> &conv1_inputShapes = tensors[0]->InputShape();
        const std::vector<std::vector<int>> &conv2_inputShapes = tensors[1]->InputShape();
        // input shape
        const std::vector<int> &conv1_inputShape = conv1_inputShapes[0];
        const std::vector<int> &conv2_inputShape = conv2_inputShapes[0];
        // output shape
        const std::vector<int> &conv1_outputShape = tensors[0]->OutputShape();
        const std::vector<int> &conv2_outputShape = tensors[1]->OutputShape();
        // kernel
        const std::vector<int> &conv1_kernelVectShape = tensors[0]->KernelShape();
        const std::vector<int> &conv2_kernelVectShape = tensors[1]->KernelShape();
        // weights and bias
        const cl_mem &conv1_Filter = tensors[1]->GetDeviceFilter();
        const cl_mem &conv1_Bias = tensors[1]->GetDeviceBias();

        const cl_mem &conv2_Filter = tensors[1]->GetDeviceFilter();
        const cl_mem &conv2_Bias = tensors[1]->GetDeviceBias();

        this->inputCLData = *tensors[0]->GetDeviceInputData();
        this->outputCLData = *tensors[1]->GetDeviceOutputData();
        // implement C4H1W4
        int conv1_outShapes[2] = {conv1_outputShape[1], conv1_outputShape[2]};
        int conv1_kernelShape[2] = {conv1_kernelVectShape[2], conv1_kernelVectShape[3]}; // OIHW
        int conv2_outShapes[2] = {conv2_outputShape[1], conv2_outputShape[2]};
        int conv2_kernelShape[2] = {conv2_kernelVectShape[2], conv2_kernelVectShape[3]}; // OIHW

        // strides and dillation
        int conv1_strideShape[2] = {tensors[0]->stride(0), tensors[0]->stride(1)}; // hw
        int conv1_dilationShape[2] = {tensors[0]->dilation(0), tensors[0]->dilation(1)};
        int conv2_strideShape[2] = {tensors[1]->stride(0), tensors[1]->stride(1)}; // hw
        int conv2_dilationShape[2] = {tensors[1]->dilation(0), tensors[1]->dilation(1)};

        auto conv1_padding = mConvCommon->GetPadding(tensors[0]); // (paddingY, paddingX)
        int conv1_paddingShape[2] = {conv1_padding.first, conv1_padding.second};
        int out_height_blk = UP_DIV(conv1_outputShape[1], 1);
        int out_width_blk = UP_DIV(conv1_outputShape[2], 4);
        int out_channel_blk = UP_DIV(conv1_outputShape[3], 4);

        const int conv1_inputHeight = conv1_inputShape.at(1);
        const int conv1_inputWidth = conv1_inputShape.at(2);
        const int conv1_inputChannels = conv1_inputShape.at(3);
        int inputImageShape[2] = {conv1_inputHeight, conv1_inputWidth};

        const int inputChannelBlocks = UP_DIV(conv1_inputChannels, 4);
        const int outputChannelBlocks = UP_DIV(conv1_outputShape[3], 4);

        mGWS[0] = static_cast<size_t>(UP_DIV(conv1_outputShape.at(3), 4) * UP_DIV(conv1_outputShape.at(2), 4));
        mGWS[1] = static_cast<size_t>(conv1_outputShape.at(0) * UP_DIV(conv1_outputShape.at(1), 1));
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[0]);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[1]);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &this->inputCLData);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &conv1_Filter);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &conv1_Bias);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &conv2_Filter);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &conv2_Bias);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &this->outputCLData);
        err |= clSetKernelArg(mKernel, idx++, sizeof(inputImageShape), &inputImageShape);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &inputChannelBlocks);
        err |= clSetKernelArg(mKernel, idx++, sizeof(conv1_outShapes), conv1_outShapes);
        err |= clSetKernelArg(mKernel, idx++, sizeof(conv1_kernelShape), &conv1_kernelShape);
        err |= clSetKernelArg(mKernel, idx++, sizeof(conv1_strideShape), &conv1_strideShape);
        err |= clSetKernelArg(mKernel, idx++, sizeof(conv1_paddingShape), &conv1_paddingShape);
        err |= clSetKernelArg(mKernel, idx++, sizeof(conv1_dilationShape), &conv1_dilationShape);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &out_width_blk);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &out_channel_blk);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &out_height_blk);
        oclCheckError(err, CL_SUCCESS);
        size_t internalGWS[2] = {mGWS[0], mGWS[1]};
        err |= clEnqueueNDRangeKernel(commandQueue[0], mKernel, 2, NULL, internalGWS, NULL, 0, nullptr, NULL);
        err |= clFinish(commandQueue[0]);
        return true;
    }
} // namespace SNN
