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
        mKernels[0] = kernelShape.at(2);
        mKernels[1] = kernelShape.at(3);

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
        const int batch = outputShape.at(0);
        const int outputHeight = outputShape.at(1);
        const int outputWidth = outputShape.at(2);
        const int channels = outputShape.at(3);
        const int inputHeight = inputShape.at(1);
        const int inputWidth = inputShape.at(2);
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
        this->mOpenCLBackend->CopyToDevice(tensor.get());
        this->inputCLData = tensor->GetDeviceInputData();
        int imageShape[2] = {UP_DIV(outputShape.at(3), 4) * outputShape.at(2), outputShape.at(0) * outputShape.at(1)};
        cl_mem outputCLData = clCreateImage2D(*GPUcontext, CL_MEM_READ_WRITE, &clImageFormat, imageShape[0], imageShape[1], 0, NULL, &err);
        tensor->SetDeviceOutputData(outputCLData);
        this->outputCLData = tensor->GetDeviceOutputData();
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[0]);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[1]);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[2]);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), this->inputCLData);
        err |= clSetKernelArg(mKernel, idx++, sizeof(inputImageShape), inputImageShape);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &outputHeight);
        err |= clSetKernelArg(mKernel, idx++, sizeof(paddingShape), paddingShape);
        err |= clSetKernelArg(mKernel, idx++, sizeof(strideShape), strideShape);
        err |= clSetKernelArg(mKernel, idx++, sizeof(kernelShape), kernelShape);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), this->outputCLData);
        oclCheckError(err, CL_SUCCESS);
        std::string kernelName = "pooling";
        mLWS = mOpenCLRuntime->localWS3DDefault(mGWS, mMaxWorkGroupSize, mOpenCLRuntime, kernelName, mKernel).first;
        err |= clFinish(commandQueue[0]);
        // exit(1);
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
    bool PoolExecution::onExecute(std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors)
    {
        int numInput = input_tensors.size();
        SNN_ASSERT(numInput == 1);
        std::shared_ptr<Tensor> input_tensor = input_tensors[0];
        std::shared_ptr<Tensor> output_tensor = output_tensors[0];
        this->inputCLData = input_tensor->GetDeviceOutputData();
        SNN_ASSERT(inputCLData != NULL);
        cl_int err = CL_SUCCESS;
        err |= clSetKernelArg(mKernel, 3, sizeof(cl_mem), this->inputCLData);
        mOpenCLRuntime->RunKernel3D(this->mKernel, mGWS, mLWS, mOpenCLRuntime);
        oclCheckError(err, CL_SUCCESS);
        output_tensor->SetDeviceOutputData(*this->outputCLData);
        output_tensor->SetDeviceInputData(*this->inputCLData);
        bool status = true;
        if (err != CL_SUCCESS)
            return false;
        return status;
    }
} // namespace SNN
