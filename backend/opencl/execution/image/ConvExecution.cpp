#include "ConvExecution.h"
#include <iostream>
#define UNIT 4
namespace SNN
{
    ConvExecution::ConvExecution(std::shared_ptr<Tensor> tensor, OpenCLBackend *mbackend) : ConvBaseExecution(tensor, mbackend)
    {
        this->mbackend = mbackend;
        mConvCommon = std::make_shared<ConvolutionCommon>();
        mStrides = {tensor->stride(0), tensor->stride(1)};
        mDilations = {tensor->dilation(0), tensor->dilation(1)};
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
            mConv1x1Opt = (mStrides[0] == 1 && mStrides[1] == 1 && gpuType == GpuType::MALI);
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
        // skip mali type
        std::string buildOption = "";
        mImageConvert = new ImageBufferConverter(mOpenCLRuntime);
        if (mOpenCLRuntime->isWeightCpuTransHalf() == false)
        {
            buildOption = "-DBUFFER_INP_FP32";
        }
        bool status = mImageConvert->ConvertBufferToImage(tensor, CONV2D_FILTER, false, buildOption);
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
        mKernel = mOpenCLRuntime->BuildKernel("conv_2d", kernelName, mBuildOptions);
        mMaxWorkGroupSize = static_cast<size_t>(mOpenCLRuntime->getMaxWorkGroupSize(mKernel));
    }
    ConvExecution::~ConvExecution()
    {
        delete mImageConvert;
    }

    bool ConvExecution::onResize(std::shared_ptr<Tensor> tensor)
    {
        const std::vector<int> &inputShape = tensor->InputShape();
        const std::vector<int> &outputShape = tensor->OutputShape();

        const std::vector<int> &kernelVectShape = tensor->KernelShape();
        const int inputHeight = inputShape.at(1);
        const int inputWidth = inputShape.at(2);
        const int inputChannels = inputShape.at(3);
        const int inputChannelBlocks = UP_DIV(inputChannels, 4);
        int strideShape[2] = {tensor->stride(0), tensor->stride(1)};   // hw
        int kernelShape[2] = {kernelVectShape[2], kernelVectShape[3]}; // OIHW
        int dilationShape[2] = {tensor->dilation(0), tensor->dilation(1)};
        auto padding = mConvCommon->GetPadding(tensor); // (paddingY, paddingX)
        mPaddings[0] = padding.first;
        mPaddings[1] = padding.second;

        const cl_mem &mFilter = tensor->GetDeviceFilter();
        const cl_mem &mBias = tensor->GetDeviceBias();
        this->mbackend->CopyToDevice(tensor.get());
        const cl_mem &inputCLData = tensor->GetDeviceInputData();
        cl_context &GPUcontext = mOpenCLRuntime->GetGPUContext();
        cl_command_queue *commandQueue = mOpenCLRuntime->GetCommandQue();
        cl_image_format clImageFormat;
        clImageFormat.image_channel_order = CL_RGBA;
        clImageFormat.image_channel_data_type = CL_FLOAT;
        cl_mem outputCLData;
        cl_int err = 0;
        if (mFilter == NULL || mBias == NULL || inputCLData == NULL)
        {
            SNN_ASSERT(true);
        }
        std::string info = std::to_string(inputChannels) + "_" + std::to_string(kernelShape[0]) + "_" + std::to_string(kernelShape[1]) + "_" + std::to_string(mStrides[0]) + "_" + std::to_string(mStrides[1]) + "_" + std::to_string(mDilations[0]) + "_" + std::to_string(mDilations[1]);
        if (kernelShape[0] == kernelShape[1] && kernelShape[0] == 1 && mPaddings[0] == 0 && mPaddings[1] == 0)
        {
            if (mConv1x1Opt)
            {
                uint32_t idx = 0;
                if (mUseLocalMem)
                {
                    mGlobalWorkSize = {
                        static_cast<size_t>(UP_DIV(outputShape.at(3), 4)), static_cast<size_t>(UP_DIV(outputShape.at(2), 4)), static_cast<size_t>(outputShape.at(0) * outputShape.at(1))};
                    std::vector<size_t> lws{UNIT, UNIT, 1};
                    mLocalWorkSize = lws;
                    err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGlobalWorkSize[0]);
                    err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGlobalWorkSize[1]);
                    err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGlobalWorkSize[2]);
                    err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &inputCLData);
                    err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &mFilter);
                    err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &mBias);
                    err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &outputCLData);
                    err |= clSetKernelArg(mKernel, idx++, sizeof(int), &inputChannelBlocks);
                    err |= clSetKernelArg(mKernel, idx++, sizeof(int), &outputShape[1]);
                    err |= clSetKernelArg(mKernel, idx++, sizeof(int), &outputShape[2]);
                }
                else
                {
                    mGlobalWorkSize = {static_cast<size_t>(UP_DIV(outputShape.at(3), 4) * UP_DIV(outputShape.at(2), 4)),
                                       static_cast<size_t>(outputShape.at(0) * outputShape.at(1))};

                    err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGlobalWorkSize[0]);
                    err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGlobalWorkSize[1]);
                    size_t width4 = UP_DIV(outputShape[2], 4);
                    err |= clSetKernelArg(mKernel, idx++, sizeof(size_t), &width4);
                    err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &mKernelBuffer);
                    err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &mBiasBuffer);
                    err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &outputCLData);
                    err |= clSetKernelArg(mKernel, idx++, sizeof(int), &inputChannelBlocks);
                    err |= clSetKernelArg(mKernel, idx++, sizeof(int), &outputShape[1]);
                    err |= clSetKernelArg(mKernel, idx++, sizeof(int), &outputShape[2]);
                }
                oclCheckError(err, CL_SUCCESS);
            }
            else
            {
                mGlobalWorkSize = {
                    static_cast<size_t>(UP_DIV(outputShape.at(3), 4) * static_cast<size_t>(UP_DIV(outputShape.at(2), 4))),
                    static_cast<size_t>(outputShape.at(0) * outputShape.at(1))};
                int imageShape[2] = {UP_DIV(outputShape.at(3), 4) * outputShape.at(2), outputShape.at(0) * outputShape.at(1)};
                uint32_t idx = 0;
                int inputImageShape[2] = {inputHeight, inputWidth};
                int outputImageShape[2] = {outputShape.at(1), outputShape.at(2)};
                outputCLData = clCreateImage2D(GPUcontext, CL_MEM_READ_WRITE, &clImageFormat, imageShape[0], imageShape[1], 0, NULL, &err);
                err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGlobalWorkSize[0]);
                err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGlobalWorkSize[1]);
                err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &inputCLData);
                err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &mFilter);
                err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &mBias);
                err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &outputCLData);
                err |= clSetKernelArg(mKernel, idx++, sizeof(inputImageShape), inputImageShape);
                err |= clSetKernelArg(mKernel, idx++, sizeof(int), &inputChannelBlocks);
                err |= clSetKernelArg(mKernel, idx++, sizeof(outputImageShape), outputImageShape);
                err |= clSetKernelArg(mKernel, idx++, sizeof(strideShape), &strideShape);
                size_t width4 = UP_DIV(outputShape[2], 4);
                err |= clSetKernelArg(mKernel, idx++, sizeof(int), &width4);
                oclCheckError(err, CL_SUCCESS);
                std::string kernelName = "conv_2d_1x1";
                mLocalWorkSize = mOpenCLRuntime->localWS2DDefault(mGlobalWorkSize, mMaxWorkGroupSize, mOpenCLRuntime, kernelName, mKernel).first;
                err |= clFinish(commandQueue[0]);
                // Testing...
                // int buffer_sizes = inputShape[0] * inputShape[1] * inputShape[2] * inputShape[3] * sizeof(float);
                // float *inpu_data = (float *)malloc(buffer_sizes);
                // FILE *ptr;
                // const char *char_path = "/aidata/anders/data_collection/okay/WF/archives/test/test_data/320_320_3.bin";
                // ptr = fopen(char_path, "rb");
                // fread(inpu_data, buffer_sizes, 1, ptr);
                // cl_mem mhostBuffer = clCreateBuffer(GPUcontext, CL_MEM_READ_WRITE, buffer_sizes, NULL, &err);
                // err |= clEnqueueWriteBuffer(commandQueue[0], mhostBuffer, CL_TRUE, 0, buffer_sizes, inpu_data, 0, NULL, NULL);
                // oclCheckError(err, CL_SUCCESS);
                // mbackend->mHostBuffer.first = buffer_sizes;
                // mbackend->mHostBuffer.second = mhostBuffer;
                // cl_mem intputImageData = mbackend->ConvertNHWCBufferToImage(tensor.get(), false, false);
                // tensor->SetDeviceInputData(intputImageData);
                // DataFormat data_format = tensor->data_format;
                // idx = 0;
                // err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGlobalWorkSize[0]);
                // err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGlobalWorkSize[1]);
                // err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &intputImageData);
                // err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &mFilter);
                // err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &mBias);
                // err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &outputCLData);
                // err |= clSetKernelArg(mKernel, idx++, sizeof(inputImageShape), inputImageShape);
                // err |= clSetKernelArg(mKernel, idx++, sizeof(int), &inputChannelBlocks);
                // err |= clSetKernelArg(mKernel, idx++, sizeof(outputImageShape), outputImageShape);
                // err |= clSetKernelArg(mKernel, idx++, sizeof(strideShape), &strideShape);
                // err |= clSetKernelArg(mKernel, idx++, sizeof(int), &width4);
                // oclCheckError(err, CL_SUCCESS);
                // const size_t internalGlobalWS[2] = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
                // const size_t lws[2] = {mLocalWorkSize[0], mLocalWorkSize[1]};
                // // const size_t lws[2] = {5, 5};
                // err |= clEnqueueNDRangeKernel(commandQueue[0], mKernel, 2, NULL, internalGlobalWS, lws, 0, NULL, NULL);
                // oclCheckError(err, CL_SUCCESS);
                // err |= clFinish(commandQueue[0]);
                // tensor->SetDeviceOutputData(outputCLData);
                // std::set<std::string> buildOptions;
                // buildOptions.emplace("-DBUFFER_IMAGE_IO_TRANS");
                // cl_kernel imageToBufferKernel = mOpenCLRuntime->BuildKernel("buffer_to_image", "image_to_nhwc_buffer", buildOptions);
                // mImageConvert->ConvertImageToNHWCBuffer(tensor, imageToBufferKernel, mOpenCLRuntime, false, false);
                // exit(1);
            }
        }
    }
    bool ConvExecution::onExecute() {}
} // namespace SNN
