#include "ConvBufExecution.h"

namespace SNN
{
    ConvBufExecution::ConvBufExecution(std::shared_ptr<Tensor> tensor, OpenCLBackend *mbackend) : ConvBaseBufExecution(tensor, mbackend)
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
        mOutputChannel = kernelShape[0], mInputChannel = kernelShape[1];
        mKernelHeight = kernelShape[2], mKernelWidth = kernelShape[3];
        kernelName = "conv_2d_c4h1w4";
        auto gpuType = mOpenCLRuntime->GetGpuType();
        mConv1x1Opt = mKernelHeight == mKernelWidth && mKernelHeight == 1 && mPaddings[0] == 0 &&
                      mPaddings[1] == 0 && mStrides[0] == 1 && mStrides[1] == 1 && inputShape[2] >= 4;
        mUseSubgroup = mOpenCLRuntime->GetGpuType() == INTEL && mOpenCLRuntime->isSupportedIntelSubgroup() && inputShape[3] >= 16;
        // Disable mUseSubgroup not implement
        const std::shared_ptr<std::vector<std::pair<float *, float *>>> mainMemory = tensor->GetMainMemory();
        const std::vector<uint8_t> &ptrIndex = tensor->GetMemoryPtrIndex();
        std::pair<float *, float *> &weight_bias = mainMemory->at(ptrIndex[0]);
        float *weightData = weight_bias.first, *biasData = weight_bias.second;
        const int weight_bytes = tensor->weight_bytes();
        int elementSize = kernelShape[0] * kernelShape[1] * kernelShape[2] * kernelShape[3];
        if (mConv1x1Opt)
        {
            kernelName = "conv_2d_1x1_c4h1w4";
            setConv1x1WeightBuffer(4, 4, elementSize, weightData);
        }
        else
        {
            if (weightData != nullptr)
            {
                bool needTrans = false;
                if (mOpenCLRuntime->isWeightCpuTransHalf() == false)
                {
                    needTrans = true;
                }
                bool status = this->mBufferConvertor->convertToNC4HW4Buffer(tensor, CONV2D_FILTER, needTrans);
            }
        }

        mBuildOptions.emplace("-DBIAS");
        if (tensor->GetActType() == kActRelu)
            mBuildOptions.emplace("-DRELU");
        else if (tensor->GetActType() == kActRelu6)
            mBuildOptions.emplace("-DRELU6");
        else if (tensor->GetActType() == kActSigmoid)
            mBuildOptions.emplace("-DSIGMOID");

        mKernel = mOpenCLRuntime->BuildKernel("conv_2d_buf", kernelName, mBuildOptions);
        printf("%s\n", kernelName.c_str());
        exit(1);
        mMaxWorkGroupSize = static_cast<size_t>(mOpenCLRuntime->getMaxWorkGroupSize(mKernel));
    }

    void ConvBufExecution::setConv1x1WeightBuffer(int packCout, int packCin, int bufferSize, const float *filterDataPtr)
    {
        if (mOpenCLRuntime->isSupportedFP16() && mOpenCLRuntime->isWeightCpuTransHalf())
        {
            bufferSize *= sizeof(cl_half);
        }
        else
        {
            bufferSize *= sizeof(float);
        }
        mKernelBuffer = clCreateBuffer(*GPUcontext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, bufferSize, NULL, &err);
        oclCheckError(err, CL_SUCCESS);
        float *kernelBufferPtr = (float *)clEnqueueMapBuffer(commandQueue[0], mKernelBuffer, true, CL_MAP_WRITE, 0, bufferSize, 0, NULL, NULL, &err);
        if (kernelBufferPtr != nullptr && err == CL_SUCCESS)
        {
            ::memset(kernelBufferPtr, 0, bufferSize);
            for (int o = 0; o < mOutputChannel; o++)
            {
                for (int i = 0; i < mInputChannel; i++)
                {
                    int bufferIdx = (o / packCout) * ROUND_UP(mInputChannel, packCin) * packCout + (i / packCin) * packCin * packCout + (o % packCout) * packCin + (i % packCin); //(Co/packCout, Ci/packCin, packCout, packCin)
                    int filterIdx = o * mInputChannel + i;
                    if (mOpenCLRuntime->isSupportedFP16())
                    {
                        kernelBufferPtr[bufferIdx] = (cl_half)(filterDataPtr)[filterIdx];
                    }
                    else
                    {
                        ((float *)kernelBufferPtr)[bufferIdx] = (float)(filterDataPtr[filterIdx]);
                    }
                }
            }
        }
        else
        {
            SNN_ERROR("ERROR: Map memory error in kernelBufferPtr !! \n");
        }
        err = clEnqueueUnmapMemObject(commandQueue[0], mKernelBuffer, kernelBufferPtr, 0, NULL, NULL);
        oclCheckError(err, CL_SUCCESS);
    }
    bool ConvBufExecution::onResize(std::shared_ptr<Tensor> tensor)
    {
        const std::vector<std::vector<int>> &inputShapes = tensor->InputShape();
        SNN_ASSERT(inputShapes.size() == 1);
        const std::vector<int> &inputShape = inputShapes[0];
        const std::vector<int> &outputShape = tensor->OutputShape();
        const int inHeight = inputShape.at(1);
        const int inWidth = inputShape.at(2);
        const int inChannel = inputShape.at(3);

        const int outHeight = outputShape.at(1);
        const int outWidth = outputShape.at(2);
        const int outChannel = outputShape.at(3);
        const int inChannelBlocks = UP_DIV(inChannel, 4);
        auto padding = mConvCommon->GetPadding(tensor); // (paddingY, paddingX)
        mPaddings[0] = padding.first;
        mPaddings[1] = padding.second;
        int inputImageShape[2] = {inHeight, inWidth};
        int outputImageShape[2] = {outputShape[1], outputShape[2]};
        int paddingShape[2] = {mPaddings[0], mPaddings[1]};
        cl_int err = 0;
        uint32_t idx = 0;
        const cl_mem &mFilter = tensor->GetDeviceFilter();
        const cl_mem &mBias = tensor->GetDeviceBias();
        this->inputCLData = *(tensor->GetDeviceInputData());
        if (mConv1x1Opt)
        {
        }
        else
        {
            int inputImageShape[2] = {inHeight, inWidth};
            int outputImageShape[2] = {outHeight, outWidth};
            int kernelShape[2] = {mKernelHeight, mKernelWidth};
            int strideShape[2] = {mStrides[0], mStrides[1]};
            int paddingShape[2] = {mPaddings[0], mPaddings[1]};
            int dilationShape[2] = {mDilations[0], mDilations[1]};
            printf("%d\n", kernelShape[0]);
            printf("%d\n", kernelShape[1]);
            printf("%d\n", kernelShape[1]);
            printf("%d\n", kernelShape[1]);
        }
        return true;
    }
} // namespace SNN
