#include "ConvExecution.h"
#include <iostream>
#include <unistd.h>
#define UNIT 4

namespace SNN
{
    ConvExecution::ConvExecution(std::shared_ptr<Tensor> tensor, OpenCLBackend *mbackend) : ConvBaseExecution(tensor, mbackend)
    {
        this->mbackend = mbackend;
        mConvCommon = std::make_shared<ConvolutionCommon>();
        mStrides = {tensor->stride(0), tensor->stride(1)};
        mDilations = {tensor->dilation(0), tensor->dilation(1)};
        const std::vector<std::vector<int>> &inputShapes = tensor->InputShape();
        const std::vector<int> &inputShape = inputShapes[0];
        const std::vector<int> &outputShape = tensor->OutputShape();
        auto padding = mConvCommon->GetPadding(tensor);
        mPaddings[0] = padding.first;
        mPaddings[1] = padding.second;
        const std::vector<int> &kernelShape = tensor->KernelShape();
        int outputChannel = kernelShape[0], inputChannel = kernelShape[1], kernelHeight = kernelShape[2], kernelWidth = kernelShape[3];
        // select conv2d operation
        std::string kernelName = "conv_2d_c4h1w4";
        auto gpuType = mOpenCLRuntime->GetGpuType();
        if (kernelHeight == kernelWidth && kernelHeight == 1 && mPaddings[0] == 0 && mPaddings[1] == 0)
        {
            mConv1x1Opt = (mStrides[0] == 1 && mStrides[1] == 1 && gpuType == GpuType::MALI && !mWeightUseBuffer);
            if (!mUseLocalMem)
            {
                if (mConv1x1Opt)
                {
                    kernelName = "conv_2d_1x1_mali";
                }
                else
                {
                    kernelName = "conv_2d_1x1";
                }
            }
        }
        if (mConv1x1Opt && !mUseLocalMem)
        {
            cl_int err;
        }
        // skip mali type
        std::string buildOption = "";
        if (mOpenCLRuntime->isWeightCpuTransHalf() == false)
        {
            buildOption = "-DBUFFER_INP_FP32";
        }
        bool status = this->mImageConvert->ConvertBufferToImage(tensor, CONV2D_FILTER, false, buildOption);
        // create kernel
        if (mStrides[0] == 1 && mStrides[1] && mDilations[0] && mDilations[1])
        {
            mBuildOptions.emplace("-DCONV_S1D1");
        }
        mBuildOptions.emplace("-DBIAS");
        if (tensor->GetActType() == kActRelu)
            mBuildOptions.emplace("-DRELU");
        else if (tensor->GetActType() == kActRelu6)
            mBuildOptions.emplace("-DRELU6");
        else if (tensor->GetActType() == kActSigmoid)
            mBuildOptions.emplace("-DSIGMOID");
        if (mWeightUseBuffer)
            mBuildOptions.emplace("-DUSE_BUFFER");
        mKernel = mOpenCLRuntime->BuildKernel("conv_2d", kernelName, mBuildOptions);
        mMaxWorkGroupSize = static_cast<size_t>(mOpenCLRuntime->getMaxWorkGroupSize(mKernel));
    }
    bool ConvExecution::onResize(std::shared_ptr<Tensor> tensor)
    {
        const std::vector<std::vector<int>> &inputShapes = tensor->InputShape();
        SNN_ASSERT(inputShapes.size() == 1);
        const std::vector<int> &inputShape = inputShapes[0];
        const std::vector<int> &outputShape = tensor->OutputShape();
        const std::vector<int> &kernelVectShape = tensor->KernelShape();
        const int inputHeight = inputShape.at(1);
        const int inputWidth = inputShape.at(2);
        const int inputChannels = inputShape.at(3);
        const int inputChannelBlocks = UP_DIV(inputChannels, 4);
        const int outputChannelBlocks = UP_DIV(outputShape[3], 4);
        int strideShape[2] = {tensor->stride(0), tensor->stride(1)};   // hw
        int kernelShape[2] = {kernelVectShape[2], kernelVectShape[3]}; // OIHW
        int dilationShape[2] = {tensor->dilation(0), tensor->dilation(1)};
        auto padding = mConvCommon->GetPadding(tensor); // (paddingY, paddingX)
        mPaddings[0] = padding.first;
        mPaddings[1] = padding.second;
        int inputImageShape[2] = {inputHeight, inputWidth};
        int outputImageShape[2] = {outputShape[1], outputShape[2]};
        int paddingShape[2] = {mPaddings[0], mPaddings[1]};
        cl_int err = 0;
        uint32_t idx = 0;
        const cl_mem &mFilter = tensor->GetDeviceFilter();
        const cl_mem &mBias = tensor->GetDeviceBias();
        this->mbackend->CopyToDevice(tensor.get());
        // this->inputCLData = tensor->GetDeviceInputData();
        this->inputCLData = *(tensor->GetDeviceInputData());
        int imageShape[2] = {UP_DIV(outputShape.at(3), 4) * outputShape.at(2), outputShape.at(0) * outputShape.at(1)};
        this->outputCLData = clCreateImage2D(*GPUcontext, CL_MEM_READ_WRITE, &clImageFormat, imageShape[0], imageShape[1], 0, NULL, &err);
        tensor->SetDeviceOutputData(this->outputCLData);
        int min_index;
        if (mFilter == NULL || mBias == NULL || this->inputCLData == nullptr)
        {
            SNN_ASSERT(true);
        }
        if (kernelShape[0] == kernelShape[1] && kernelShape[0] == 1 && mPaddings[0] == 0 && mPaddings[1] == 0)
        {
            if (mConv1x1Opt)
            {
                idx = 0;
                if (mUseLocalMem)
                {
                    mGWS = {
                        static_cast<size_t>(UP_DIV(outputShape.at(3), 4)), static_cast<size_t>(UP_DIV(outputShape.at(2), 4)), static_cast<size_t>(outputShape.at(0) * outputShape.at(1))};
                    std::vector<size_t> lws{UNIT, UNIT, 1};
                    mLWS = lws;
                    err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[0]);
                    err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[1]);
                    err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[2]);
                    err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &this->inputCLData);
                    err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &mFilter);
                    err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &mBias);
                    err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &this->outputCLData);
                    err |= clSetKernelArg(mKernel, idx++, sizeof(int), &inputChannelBlocks);
                    err |= clSetKernelArg(mKernel, idx++, sizeof(int), &outputShape[1]);
                    err |= clSetKernelArg(mKernel, idx++, sizeof(int), &outputShape[2]);
                }
                else
                {
                    mGWS = {static_cast<size_t>(UP_DIV(outputShape.at(3), 4) * UP_DIV(outputShape.at(2), 4)),
                            static_cast<size_t>(outputShape.at(0) * outputShape.at(1))};
                    err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[0]);
                    err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[1]);
                    size_t width4 = UP_DIV(outputShape[2], 4);
                    err |= clSetKernelArg(mKernel, idx++, sizeof(size_t), &width4);
                    err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &mKernelBuffer);
                    err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &mBiasBuffer);
                    err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &this->outputCLData);
                    err |= clSetKernelArg(mKernel, idx++, sizeof(int), &inputChannelBlocks);
                    err |= clSetKernelArg(mKernel, idx++, sizeof(int), &outputShape[1]);
                    err |= clSetKernelArg(mKernel, idx++, sizeof(int), &outputShape[2]);
                    // std::cout << kernelName[min_index] << std::endl;
                }
                oclCheckError(err, CL_SUCCESS);
            }
            else
            {
                mGWS = {
                    static_cast<size_t>(UP_DIV(outputShape.at(3), 4) * static_cast<size_t>(UP_DIV(outputShape.at(2), 4))),
                    static_cast<size_t>(outputShape.at(0) * outputShape.at(1))};
                idx = 0;
                int inputImageShape[2] = {inputHeight, inputWidth};
                int outputImageShape[2] = {outputShape.at(1), outputShape.at(2)};
                err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[0]);
                err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[1]);
                err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &this->inputCLData);
                err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &mFilter);
                err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &mBias);
                err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &this->outputCLData);
                err |= clSetKernelArg(mKernel, idx++, sizeof(inputImageShape), inputImageShape);
                err |= clSetKernelArg(mKernel, idx++, sizeof(int), &inputChannelBlocks);
                err |= clSetKernelArg(mKernel, idx++, sizeof(outputImageShape), outputImageShape);
                err |= clSetKernelArg(mKernel, idx++, sizeof(strideShape), &strideShape);
                size_t width4 = UP_DIV(outputShape[2], 4);
                err |= clSetKernelArg(mKernel, idx++, sizeof(int), &width4);
                oclCheckError(err, CL_SUCCESS);
                std::string kernelName = "conv_2d_1x1";
                mLWS = mOpenCLRuntime->localWS2DDefault(mGWS, mMaxWorkGroupSize, mOpenCLRuntime, kernelName, mKernel).first;
                err |= clFinish(commandQueue[0]);
            }
        }
        else
        {
            const int total_kernel = 3;
            std::string kernelName[total_kernel] = {"conv_2d_c4h1w4", "conv_2d_c4h4w1", "conv_2d_c8h4w1"};
            int itemC[total_kernel] = {4, 4, 8};
            int itemH[total_kernel] = {1, 4, 4};
            int itemW[total_kernel] = {4, 1, 1};
            cl_kernel kernel[total_kernel];
            std::vector<size_t> globalWorkSize[total_kernel];
            std::vector<size_t> localWorkSize[total_kernel];
            std::pair<float, int> min_cost(INT_MAX, 0); //(min_time, min_index)
            int imageShape[2] = {UP_DIV(outputShape.at(3), 4) * outputShape.at(2), outputShape.at(0) * outputShape.at(1)};
            outputCLData = clCreateImage2D(*GPUcontext, CL_MEM_READ_WRITE, &clImageFormat, imageShape[0], imageShape[1], 0, NULL, &err);
            for (int knl_idx = 0; knl_idx < total_kernel; knl_idx++)
            {
                idx = 0;
                kernel[knl_idx] = mOpenCLRuntime->BuildKernel("conv_2d", kernelName[knl_idx], mBuildOptions);
                mMaxWorkGroupSize = static_cast<size_t>(mOpenCLRuntime->getMaxWorkGroupSize(kernel[knl_idx]));
                globalWorkSize[knl_idx] = {static_cast<size_t>(UP_DIV(outputShape.at(3), itemC[knl_idx]) * UP_DIV(outputShape.at(2), itemW[knl_idx])), static_cast<size_t>(outputShape.at(0) * UP_DIV(outputShape.at(1), itemH[knl_idx]))};
                err = CL_SUCCESS;
                err |= clSetKernelArg(kernel[knl_idx], idx++, sizeof(int), &globalWorkSize[knl_idx][0]);
                err |= clSetKernelArg(kernel[knl_idx], idx++, sizeof(int), &globalWorkSize[knl_idx][1]);
                err |= clSetKernelArg(kernel[knl_idx], idx++, sizeof(cl_mem), &this->inputCLData);
                err |= clSetKernelArg(kernel[knl_idx], idx++, sizeof(cl_mem), &mFilter);
                err |= clSetKernelArg(kernel[knl_idx], idx++, sizeof(cl_mem), &mBias);
                err |= clSetKernelArg(kernel[knl_idx], idx++, sizeof(cl_mem), &this->outputCLData);
                err |= clSetKernelArg(kernel[knl_idx], idx++, sizeof(inputImageShape), &inputImageShape);
                err |= clSetKernelArg(kernel[knl_idx], idx++, sizeof(int), &inputChannelBlocks);
                err |= clSetKernelArg(kernel[knl_idx], idx++, sizeof(outputImageShape), &outputImageShape);
                err |= clSetKernelArg(kernel[knl_idx], idx++, sizeof(kernelShape), &kernelShape);
                err |= clSetKernelArg(kernel[knl_idx], idx++, sizeof(strideShape), &strideShape);
                err |= clSetKernelArg(kernel[knl_idx], idx++, sizeof(paddingShape), &paddingShape);
                err |= clSetKernelArg(kernel[knl_idx], idx++, sizeof(dilationShape), &dilationShape);
                int width_blk = UP_DIV(outputShape[2], itemW[knl_idx]);
                err |= clSetKernelArg(kernel[knl_idx], idx++, sizeof(int), &width_blk);
                err |= clSetKernelArg(kernel[knl_idx], idx++, sizeof(int), &outputChannelBlocks);
                int height_blk = UP_DIV(outputShape[1], itemH[knl_idx]);
                err |= clSetKernelArg(kernel[knl_idx], idx++, sizeof(int), &height_blk);
                oclCheckError(err, CL_SUCCESS);

                std::pair<std::vector<size_t>, float_t> retTune;
                retTune = mOpenCLRuntime->localWS2DDefault(globalWorkSize[knl_idx], mMaxWorkGroupSize, mOpenCLRuntime, kernelName[knl_idx], kernel[knl_idx]);
                err |= clFlush(commandQueue[0]);
                if (min_cost.first > retTune.second)
                {
                    min_cost.first = retTune.second;
                    min_cost.second = knl_idx;
                    mLWS = {retTune.first[0], retTune.first[1]};
                }
            }
            min_index = min_cost.second;
            mGWS = {globalWorkSize[min_index][0], globalWorkSize[min_index][1]};
            this->mKernel = mOpenCLRuntime->BuildKernel("conv_2d", kernelName[min_index], mBuildOptions);
            idx = 0;
            err = CL_SUCCESS;
            err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[0]);
            err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[1]);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &this->inputCLData);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &mFilter);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &mBias);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &this->outputCLData);
            err |= clSetKernelArg(mKernel, idx++, sizeof(inputImageShape), &inputImageShape);
            err |= clSetKernelArg(mKernel, idx++, sizeof(int), &inputChannelBlocks);
            err |= clSetKernelArg(mKernel, idx++, sizeof(outputImageShape), &outputImageShape);
            err |= clSetKernelArg(mKernel, idx++, sizeof(kernelShape), &kernelShape);
            err |= clSetKernelArg(mKernel, idx++, sizeof(strideShape), &strideShape);
            err |= clSetKernelArg(mKernel, idx++, sizeof(paddingShape), &paddingShape);
            err |= clSetKernelArg(mKernel, idx++, sizeof(dilationShape), &dilationShape);
            int width_blk = UP_DIV(outputShape[2], itemW[min_index]);
            err |= clSetKernelArg(mKernel, idx++, sizeof(int), &width_blk);
            err |= clSetKernelArg(mKernel, idx++, sizeof(int), &outputChannelBlocks);
            int height_blk = UP_DIV(outputShape[1], itemH[min_index]);
            err |= clSetKernelArg(mKernel, idx++, sizeof(int), &height_blk);
            oclCheckError(err, CL_SUCCESS);
        }
        return true;
    }
    bool ConvExecution::onExecute(std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors)
    {
        int numInput = input_tensors.size();
        SNN_ASSERT(numInput == 1);
        std::shared_ptr<Tensor> input_tensor = input_tensors[0];
        std::shared_ptr<Tensor> output_tensor = output_tensors[0];
        const std::vector<std::vector<int>> &inputShapes = output_tensor->InputShape();
        this->inputCLData = *(input_tensor->GetDeviceOutputData());
        SNN_ASSERT(inputCLData != NULL);
        cl_int err = CL_SUCCESS;
        err |= clSetKernelArg(mKernel, 2, sizeof(cl_mem), &this->inputCLData);
        mOpenCLRuntime->RunKernel2D(this->mKernel, mGWS, mLWS, mOpenCLRuntime);
        oclCheckError(err, CL_SUCCESS);
        output_tensor->SetDeviceOutputData(this->outputCLData);
        output_tensor->SetDeviceInputData(this->inputCLData);
        if (err != CL_SUCCESS)
            return false;
        return true;
    }
    bool ConvExecution::onOptimizedExecute(std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors)
    {
        int numInput = input_tensors.size();
        int numOutput = output_tensors.size();
        this->inputCLData = *(input_tensors[numInput - 1]->GetDeviceOutputData());
        SNN_ASSERT(inputCLData != NULL);
        cl_int err = CL_SUCCESS;
        err |= clSetKernelArg(mKernel, 2, sizeof(cl_mem), &this->inputCLData);
        mOpenCLRuntime->RunKernel2D(this->mKernel, mGWS, mLWS, mOpenCLRuntime);
        output_tensors[numOutput - 1]->SetDeviceOutputData(this->outputCLData);
        // output_tensors[numOutput - 1]->SetDeviceInputData(this->inputCLData);
        bool status = true;
        if (err != CL_SUCCESS)
            return false;
        return status;
    }
} // namespace SNN
