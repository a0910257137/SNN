#include "PoolExecution.h"
#include "backend/opencl/core/ImageBufferConverter.h"
namespace SNN
{

    PoolExecution::PoolExecution(std::shared_ptr<Tensor> tensor, OpenCLBackend *mbackend) : Execution(mbackend)
    {
        this->mOpenCLBackend = mbackend;
        mConvCommon = std::make_shared<ConvolutionCommon>();
        const std::vector<int> &kernelShape = tensor->KernelShape();
        mStrides[0] = tensor->stride(0);
        mStrides[1] = tensor->stride(1);
        mKernels[0] = kernelShape[2];
        mKernels[1] = kernelShape[3];
        auto padding = mConvCommon->GetPadding(tensor);
        mPaddings[0] = padding.first * 2;
        mPaddings[1] = padding.second * 2;
        if (tensor->GetPaddingType() == kPaddingValid)
        {
            mPaddings[0] = 0;
            mPaddings[1] = 0;
        }
        std::set<std::string> buildOptions;
        std::string kernelName = "pooling";
        if (tensor->GetOpType() == AVERAGE_POOL_2D)
        {
            buildOptions.emplace("-DPOOL_AVG");
        }
        mKernel = mOpenCLRuntime->BuildKernel("pooling", kernelName, buildOptions);
        mMaxWorkGroupSize = static_cast<size_t>(mOpenCLRuntime->getMaxWorkGroupSize(mKernel));
    }
    bool PoolExecution::onResize(std::shared_ptr<Tensor> tensor)
    {
        const std::vector<std::vector<int>> &inputShapes = tensor->InputShape();
        SNN_ASSERT(inputShapes.size() == 1);
        const std::vector<int> &inputShape = inputShapes[0];

        const std::vector<int> &outputShape = tensor->OutputShape();
        const std::vector<int> &kernelVectShape = tensor->KernelShape();
        // Global pooling !?

        if (tensor->GetPaddingType() == kPaddingSame)
        {
            int padNeedHeight = MAX(0, (outputShape[1] - 1) * mStrides[0] + mKernels[0] - inputShape[1]);
            int padNeedWidth = MAX(0, (outputShape[2] - 1) * mStrides[1] + mKernels[1] - inputShape[2]);
            mPaddings[0] = padNeedHeight;
            mPaddings[1] = padNeedWidth;
        }
        SNN_ASSERT(mDilations[0] == 1 && mDilations[1] == 1);
        const int batch = outputShape[0];

        const int outputHeight = outputShape[1];
        const int outputWidth = outputShape[2];
        const int channels = outputShape[3];
        const int inputHeight = inputShape[1];
        const int inputWidth = inputShape[2];
        int channelBlocks = UP_DIV(channels, 4);
        mGWS = {
            static_cast<size_t>(channelBlocks),
            static_cast<size_t>(outputWidth),
            static_cast<size_t>(batch * outputHeight),

        };
        int inputImageShape[2] = {inputHeight, inputWidth};
        int paddingShape[2] = {mPaddings[0] / 2, mPaddings[1] / 2};
        int strideShape[2] = {mStrides[0], mStrides[1]};
        int kernelShape[2] = {mKernels[0], mKernels[1]};
        mLWS = this->PoolLocalWS(mGWS, mMaxWorkGroupSize);
        uint32_t idx = 0;
        cl_int err;
        this->mOpenCLBackend->CopyToDevice(tensor.get());
        const cl_mem &inputCLData = tensor->GetDeviceInputData();
        cl_context &GPUcontext = mOpenCLRuntime->GetGPUContext();
        cl_command_queue *commandQueue = mOpenCLRuntime->GetCommandQue();
        cl_image_format clImageFormat;
        clImageFormat.image_channel_order = CL_RGBA;
        clImageFormat.image_channel_data_type = CL_FLOAT;
        int imageShape[2] = {UP_DIV(outputShape.at(3), 4) * outputShape.at(2), outputShape.at(0) * outputShape.at(1)};
        cl_mem outputCLData = clCreateImage2D(GPUcontext, CL_MEM_READ_WRITE, &clImageFormat, imageShape[0], imageShape[1], 0, NULL, &err);
        oclCheckError(err, CL_SUCCESS);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[0]);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[1]);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[2]);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &inputCLData);
        err |= clSetKernelArg(mKernel, idx++, sizeof(inputImageShape), inputImageShape);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &outputHeight);
        err |= clSetKernelArg(mKernel, idx++, sizeof(paddingShape), paddingShape);
        err |= clSetKernelArg(mKernel, idx++, sizeof(strideShape), strideShape);
        err |= clSetKernelArg(mKernel, idx++, sizeof(kernelShape), kernelShape);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &outputCLData);
        oclCheckError(err, CL_SUCCESS);
        // Testing ..
        // int buffer_sizes = inputShape[0] * inputShape[1] * inputShape[2] * inputShape[3] * sizeof(float);
        // float *inpu_data = (float *)malloc(buffer_sizes);
        // FILE *ptr;
        // const char *char_path = "/aidata/anders/data_collection/okay/WF/archives/test/test_data/avgpool/320_320_3.bin";
        // ptr = fopen(char_path, "rb");
        // fread(inpu_data, buffer_sizes, 1, ptr);
        // // for (int i = 0; i < 30; i++)
        // // {
        // //     printf("%f\n", inpu_data[i]);
        // // }
        // // exit(1);
        // cl_mem mhostBuffer = clCreateBuffer(GPUcontext, CL_MEM_READ_WRITE, buffer_sizes, NULL, &err);
        // err |= clEnqueueWriteBuffer(commandQueue[0], mhostBuffer, CL_TRUE, 0, buffer_sizes, inpu_data, 0, NULL, NULL);
        // oclCheckError(err, CL_SUCCESS);
        // mOpenCLBackend->mHostBuffer.first = buffer_sizes;
        // mOpenCLBackend->mHostBuffer.second = mhostBuffer;
        // cl_mem intputImageData = mOpenCLBackend->ConvertNHWCBufferToImage(tensor.get(), false, false);
        // tensor->SetDeviceInputData(intputImageData);
        // DataFormat data_format = tensor->data_format;
        // // // implement average pooling 2D  images
        // err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[0]);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[1]);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[2]);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &intputImageData);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(inputImageShape), inputImageShape);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(int), &outputHeight);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(paddingShape), paddingShape);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(strideShape), strideShape);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(kernelShape), kernelShape);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &outputCLData);
        // oclCheckError(err, CL_SUCCESS);
        // const size_t internalGlobalWS[3] = {mGWS[0], mGWS[1], mGWS[2]};
        // // printf("%d\n", mGWS[0]);
        // // printf("%d\n", mGWS[1]);
        // // printf("%d\n", mGWS[2]);
        // // exit(1);
        // const size_t lws[3] = {1, 4, 4};
        // err |= clEnqueueNDRangeKernel(commandQueue[0], mKernel, 3, NULL, internalGlobalWS, lws, 0, NULL, NULL);
        // oclCheckError(err, CL_SUCCESS);
        // err |= clFinish(commandQueue[0]);
        // tensor->SetDeviceOutputData(outputCLData);
        // std::set<std::string> buildOptions;
        // buildOptions.emplace("-DBUFFER_IMAGE_IO_TRANS");
        // cl_kernel imageToBufferKernel = mOpenCLRuntime->BuildKernel("buffer_to_image", "image_to_nhwc_buffer", buildOptions);
        // ImageBufferConverter mImageConvert(mOpenCLRuntime);
        // mImageConvert.ConvertImageToNHWCBuffer(tensor, imageToBufferKernel, mOpenCLRuntime, false, false);
        // exit(1);
    }

    bool PoolExecution::onExecute()
    {
    }
    std::vector<size_t> PoolExecution::PoolLocalWS(const std::vector<size_t> &gws, const size_t maxWorkGroupSize)
    {
        std::vector<size_t> lws(3, 0);
        auto maxWorkItemSizes = mOpenCLRuntime->getMaxWorkItemSizes();
        uint32_t deviceComputeUnits = mOpenCLRuntime->deviceComputeUnits();
        int coreNum = deviceComputeUnits;
        for (int i = 0, totalSizeNow = 1; i < gws.size(); ++i)
        {
            int remain = gws[i] % coreNum, groupSize = gws[i] / coreNum;
            if (remain == 0)
            {
                lws[i] = groupSize;
            }
            else
            {
                while (groupSize)
                {
                    int remain = gws[i] % groupSize;
                    if (remain == 0 && (i > 0 || groupSize <= maxWorkGroupSize))
                    {
                        lws[i] = groupSize;
                        break;
                    }
                    --groupSize;
                }
            }
            int limit = MIN(maxWorkGroupSize / totalSizeNow, maxWorkItemSizes[i]);
            lws[i] = MAX(MIN(lws[i], limit), 1);
            totalSizeNow *= lws[i];
        }
        return lws;
    }
} // namespace SNN
