#include "DepthwiseConvExecution.h"
#include "backend/opencl/core/runtime/OpenCLRuntime.h"

namespace SNN
{
    DepthwiseConvExecution::DepthwiseConvExecution(std::shared_ptr<Tensor> tensor, OpenCLBackend *mbackend) : ConvBaseExecution(tensor, mbackend)
    {

        this->mbackend = mbackend;
        mConvCommon = std::make_shared<ConvolutionCommon>();
        mImageConvert = new ImageBufferConverter(mOpenCLRuntime);
        mStrides[0] = tensor->stride(0), mStrides[1] = tensor->stride(0);
        mDilations[0] = tensor->dilation(0), mDilations[1] = tensor->dilation(1);
        std::vector<int> kernelShape = tensor->KernelShape();
        int kernelMultiplier = kernelShape[0],
            kernelOutputChannel = kernelShape[1],
            kernelHeight = kernelShape[2],
            kernelWidth = kernelShape[3];

        // oclCheckError(mStrides[0] > 0 && mStrides[1] > 1, true);
        int buffer_size = kernelMultiplier * kernelHeight * kernelWidth * kernelOutputChannel;
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
        bool status = mImageConvert->ConvertBufferToImage(tensor, DW_CONV2D_FILTER, false, buildOption);
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
        delete mImageConvert;
    }
    bool DepthwiseConvExecution::onResize(std::shared_ptr<Tensor> tensor)
    {
        const std::vector<std::vector<int>> &inputShapes = tensor->InputShape();
        SNN_ASSERT(inputShapes.size() == 1);
        const std::vector<int> &inputShape = inputShapes[0];

        const std::vector<int> &outputShape = tensor->OutputShape();
        const std::vector<int> &kernelVectShape = tensor->KernelShape();
        mGlobalWorkSize[0] = UP_DIV(outputShape[3], 4) * UP_DIV(outputShape[2], 4);
        mGlobalWorkSize[1] = outputShape[0] * outputShape[1];
        int inputImageShape[2] = {inputShape[1], inputShape[2]};
        int outputImageShape[2] = {outputShape[1], outputShape[2]};
        int strideShape[2] = {tensor->stride(0), tensor->stride(1)};
        int kernelShape[2] = {kernelVectShape[2], kernelVectShape[3]};
        int dilationShape[2] = {tensor->dilation(0), tensor->dilation(1)};
        auto padding = mConvCommon->GetPadding(tensor);
        /*---------------------------------------------------*/
        int paddingShape[2] = {padding.first, padding.second};
        const int inputChannels = inputShape[3];
        const int inputChannelBlocks[1] = {UP_DIV(inputChannels, 4)};
        std::string kernelName = "depthwise_conv2d_s1";
        cl_int err = 0;
        uint32_t idx = 0;
        cl_context &GPUcontext = mOpenCLRuntime->GetGPUContext();
        cl_command_queue *commandQueue = mOpenCLRuntime->GetCommandQue();
        cl_image_format clImageFormat;
        clImageFormat.image_channel_order = CL_RGBA;
        clImageFormat.image_channel_data_type = CL_FLOAT;
        int imageShape[2] = {UP_DIV(outputShape.at(3), 4) * outputShape.at(2), outputShape.at(0) * outputShape.at(1)};
        cl_mem outputCLData = clCreateImage2D(GPUcontext, CL_MEM_READ_WRITE, &clImageFormat, imageShape[0], imageShape[1], 0, NULL, &err);
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
        err |= clFinish(commandQueue[0]);
        oclCheckError(err, CL_SUCCESS);
        // Testing ..
        // int buffer_sizes = inputShape[0] * inputShape[1] * inputShape[2] * inputShape[3] * sizeof(float);
        // float *inpu_data = (float *)malloc(buffer_sizes);
        // FILE *ptr;
        // const char *char_path = "/aidata/anders/data_collection/okay/WF/archives/test/test_data/640_640_3.bin";
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
        // // implement depth-wise convolution
        // idx = 0;
        // err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGlobalWorkSize[0]);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGlobalWorkSize[1]);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &intputImageData);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &mFilter);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &mBias);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &outputCLData);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(inputImageShape), inputImageShape);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(inputChannelBlocks), inputChannelBlocks);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(outputImageShape), outputImageShape);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(kernelShape), kernelShape);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(paddingShape), paddingShape);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(dilationShape), dilationShape);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(strideShape), strideShape);
        // oclCheckError(err, CL_SUCCESS);
        // const size_t internalGlobalWS[2] = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
        // const size_t lws[2] = {mLocalWorkSize[0], mLocalWorkSize[1]};
        // const size_t lws[2] = {5, 5};
        // err |= clEnqueueNDRangeKernel(commandQueue[0], mKernel, 2, NULL, internalGlobalWS, lws, 0, NULL, NULL);
        // oclCheckError(err, CL_SUCCESS);
        // err |= clFinish(commandQueue[0]);
        // tensor->SetDeviceOutputData(outputCLData);
        // std::set<std::string> buildOptions;
        // buildOptions.emplace("-DBUFFER_IMAGE_IO_TRANS");
        // cl_kernel imageToBufferKernel = mOpenCLRuntime->BuildKernel("buffer_to_image", "image_to_nhwc_buffer", buildOptions);
        // mImageConvert->ConvertImageToNHWCBuffer(tensor, imageToBufferKernel, mOpenCLRuntime, false, false);
        return true;
    }
    bool DepthwiseConvExecution::onExecute()
    {
    }
}