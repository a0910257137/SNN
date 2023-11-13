
#include "DeconvExecution.h"
namespace SNN
{
    DeconvExecution::DeconvExecution(std::shared_ptr<Tensor> tensor, OpenCLBackend *mbackend) : Execution(mbackend)
    {
        this->mOpenCLBackend = mbackend;
        const std::vector<int> &kernelShape = tensor->KernelShape(); // OIHW
        mStrides = {tensor->stride(0), tensor->stride(1)};
        mDilations = {1, 1};
        SNN_ASSERT(mStrides[0] > 0 && mStrides[1] > 0);
        const std::vector<std::vector<int>> &inputShapes = tensor->InputShape();
        const std::vector<int> &outputShape = tensor->OutputShape();
        const std::vector<int> &inputShape = inputShapes.at(0);
        int inputChannel = inputShape[3];
        int outputChannel = outputShape[3];
        int kernelHeight = kernelShape[2];
        int kernelWidth = kernelShape[3];
        SNN_ASSERT(outputChannel == kernelShape[0]);
        SNN_ASSERT(inputChannel == kernelShape[1]);
        const float *filterDataPtr = nullptr;
        int weightSize = 0;
        int imageShape[2] = {inputChannel, UP_DIV(outputChannel, 4) * kernelHeight * kernelWidth};
        int elementSize = outputChannel * inputChannel * kernelHeight * kernelWidth;
        int buffer_size;
        if (mOpenCLRuntime->isWeightCpuTransHalf())
        {
            buffer_size = elementSize * sizeof(cl_half);
        }
        else
        {
            buffer_size = elementSize * sizeof(float);
        }
        const std::shared_ptr<std::vector<std::pair<float *, float *>>> mainMemory = tensor->GetMainMemory();
        const std::vector<uint8_t> &ptrIndex = tensor->GetMemoryPtrIndex();
        std::pair<float *, float *> &weight_bias = mainMemory->at(ptrIndex[0]);
        float *weightData = weight_bias.first, *biasData = weight_bias.second;

        cl_int err = 0;
        uint32_t idx = 0;
        cl_mem filterBufferCL = clCreateBuffer(*GPUcontext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, buffer_size, NULL, &err);
        oclCheckError(err, CL_SUCCESS);
        float *ptrCL = (float *)clEnqueueMapBuffer(commandQueue[0], filterBufferCL, true, CL_MAP_WRITE, 0, buffer_size, 0, NULL, NULL, &err);
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
        err = clEnqueueUnmapMemObject(commandQueue[0], filterBufferCL, ptrCL, 0, NULL, NULL);
        oclCheckError(err, CL_SUCCESS);
        std::string buildOption = "";
        if (mOpenCLRuntime->isWeightCpuTransHalf() == false)
            buildOption = "-DBUFFER_INP_FP32";

        mImageConvert->ConvertBufferToImage(tensor, CONV2D_FILTER, false, buildOption);
        std::set<std::string> buildOptions;
        std::string kernelName = "deconv_2d";
        // not support bias in deconvolution 2d
        // buildOptions.emplace("-DBIAS");
        if (tensor->GetActType() == kActRelu)
            buildOptions.emplace("-DRELU");

        else if (tensor->GetActType() == kActRelu6)
            buildOptions.emplace("-DRELU6");
        mKernel = mOpenCLRuntime->BuildKernel("deconv_2d", kernelName, buildOptions);
        mMaxWorkGroupSize = static_cast<size_t>(mOpenCLRuntime->getMaxWorkGroupSize(mKernel));
        clReleaseMemObject(filterBufferCL);
    }
    bool DeconvExecution::onResize(std::shared_ptr<Tensor> tensor)
    {
        const std::vector<std::vector<int>> &inputShapes = tensor->InputShape();
        const std::vector<int> &outputShape = tensor->OutputShape();
        const std::vector<int> &inputShape = inputShapes.at(0);
        const std::vector<int> &kernelShape = tensor->KernelShape(); // OIHW
        const int outputBatch = outputShape[0];
        const int outputHeight = outputShape[1];
        const int outputWidth = outputShape[2];
        const int outputChannels = outputShape[3];
        const int inputChannels = inputShape[3];
        const int outputChannelBlocks = UP_DIV(outputChannels, 4);
        const int strideHeight = mStrides[0];
        const int strideWidth = mStrides[1];
        auto padding = mConvCommon->GetPadding(tensor);
        auto ky = kernelShape[2];
        auto kx = kernelShape[3];
        auto kernelSize = ky * kx;
        const int transPadH = ky - 1 - padding.first;
        const int transPadW = kx - 1 - padding.second;
        const int alignHeight = mStrides[0] - 1 - transPadH;
        const int alignWidth = mStrides[1] - 1 - transPadW;
        mGWS = {static_cast<size_t>(outputChannelBlocks),
                static_cast<size_t>(outputWidth),
                static_cast<size_t>(outputHeight * outputBatch)};
        int inputImageShape[2] = {inputShape.at(1), inputShape.at(2)};
        int outputImageShape[2] = {outputHeight, outputWidth};
        int strideShape[2] = {strideHeight, strideWidth};
        int paddingShape[2] = {transPadH, transPadW};
        int alignShape[2] = {alignHeight, alignWidth};
        int ks[2] = {ky, kx};
        int intputChannelBlocks = UP_DIV(inputChannels, 4);
        cl_int err = 0;
        uint32_t idx = 0;
        int intputCLImageShape[2] = {UP_DIV(inputShape.at(3), 4) * inputShape.at(2), inputShape.at(0) * inputShape.at(1)};
        cl_mem inputCLData = clCreateImage2D(*GPUcontext, CL_MEM_READ_WRITE, &clImageFormat, intputCLImageShape[0], intputCLImageShape[1], 0, NULL, &err);
        int outputCLImageShape[2] = {UP_DIV(outputShape.at(3), 4) * outputShape.at(2), outputShape.at(0) * outputShape.at(1)};
        cl_mem outputCLData = clCreateImage2D(*GPUcontext, CL_MEM_READ_WRITE, &clImageFormat, outputCLImageShape[0], outputCLImageShape[1], 0, NULL, &err);
        const cl_mem &mFilter = tensor->GetDeviceFilter();
        oclCheckError(err, CL_SUCCESS);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[0]);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[1]);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[2]);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &inputCLData);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &mFilter);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &outputCLData);
        err |= clSetKernelArg(mKernel, idx++, sizeof(inputImageShape), inputImageShape);
        err |= clSetKernelArg(mKernel, idx++, sizeof(outputImageShape), outputImageShape);
        err |= clSetKernelArg(mKernel, idx++, sizeof(strideShape), strideShape);
        err |= clSetKernelArg(mKernel, idx++, sizeof(alignShape), alignShape);
        err |= clSetKernelArg(mKernel, idx++, sizeof(paddingShape), paddingShape);
        err |= clSetKernelArg(mKernel, idx++, sizeof(ks), ks);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &kernelSize);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &intputChannelBlocks);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &outputChannelBlocks);
        oclCheckError(err, CL_SUCCESS);
        std::string kernelName = "deconv2d";
        oclCheckError(err, CL_SUCCESS);
        mLWS = mOpenCLRuntime->localWS3DDefault(mGWS, mMaxWorkGroupSize, mOpenCLRuntime, kernelName, mKernel).first;
        err |= clFinish(commandQueue[0]);
        oclCheckError(err, CL_SUCCESS);
        // exit(1);
        // Test..
        // int buffer_sizes = inputShape[0] * inputShape[1] * inputShape[2] * inputShape[3] * sizeof(float);
        // float *inpu_data = (float *)malloc(buffer_sizes);
        // FILE *ptr;
        // const char *char_path = "/aidata/anders/data_collection/okay/WF/archives/test/test_data/deconv2D/320_320_3.bin";
        // ptr = fopen(char_path, "rb");
        // fread(inpu_data, buffer_sizes, 1, ptr);
        // cl_mem mhostBuffer = clCreateBuffer(GPUcontext, CL_MEM_READ_WRITE, buffer_sizes, NULL, &err);
        // err |= clEnqueueWriteBuffer(commandQueue[0], mhostBuffer, CL_TRUE, 0, buffer_sizes, inpu_data, 0, NULL, NULL);
        // oclCheckError(err, CL_SUCCESS);
        // mOpenCLBackend->mHostBuffer.first = buffer_sizes;
        // mOpenCLBackend->mHostBuffer.second = mhostBuffer;
        // cl_mem intputImageData = mOpenCLBackend->ConvertNHWCBufferToImage(inputShape, tensor->data_format, false, false);
        // tensor->SetDeviceInputData(intputImageData);
        // DataFormat data_format = tensor->data_format;
        // err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[0]);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[1]);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[2]);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &intputImageData);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &mFilter);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &outputCLData);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(inputImageShape), inputImageShape);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(outputImageShape), outputImageShape);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(strideShape), strideShape);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(alignShape), alignShape);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(paddingShape), paddingShape);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(ks), ks);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(int), &kernelSize);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(int), &intputChannelBlocks);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(int), &outputChannelBlocks);
        // oclCheckError(err, CL_SUCCESS);
        // const size_t internalGlobalWS[3] = {mGWS[0], mGWS[1], mGWS[2]};
        // const size_t lws[3] = {1, 5, 5};
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
        return true;
    }
    bool DeconvExecution::onExecute(std::vector<std::shared_ptr<Tensor>> &inputs, std::vector<std::shared_ptr<Tensor>> &outputs)
    {
        return true;
    }

} // namespace SNN
