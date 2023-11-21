#include "DepthwiseConvExecution.h"
#include "backend/opencl/core/runtime/OpenCLRuntime.h"

namespace SNN
{
    DepthwiseConvExecution::DepthwiseConvExecution(std::shared_ptr<Tensor> tensor, OpenCLBackend *mbackend) : ConvBaseExecution(tensor, mbackend)
    {

        this->mbackend = mbackend;
        mConvCommon = std::make_shared<ConvolutionCommon>();
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
        std::string kernelName = "depthwise_conv2d";
        // mBuildOptions.emplace("-DNO_BIAS");
        if (mStrides[0] == 1 && mStrides[1] == 1 &&
            mDilations[0] == 1 && mDilations[1] == 1)
            kernelName = "depthwise_conv2d_s1";
        if (tensor->GetActType() == kActRelu)
            mBuildOptions.emplace("-DRELU");
        else if (tensor->GetActType() == kActRelu6)
            mBuildOptions.emplace("-DRELU6");
        else if (tensor->GetActType() == kActSigmoid)
            mBuildOptions.emplace("-DSIGMOID");
        // for (auto &m : mBuildOptions)
        //     std::cout << m << std::endl;
        mKernel = mOpenCLRuntime->BuildKernel("depthwise_conv2d", kernelName, mBuildOptions);
        mMaxWorkGroupSize = static_cast<size_t>(mOpenCLRuntime->getMaxWorkGroupSize(mKernel));
    }
    bool DepthwiseConvExecution::onResize(std::shared_ptr<Tensor> tensor)
    {
        const std::vector<std::vector<int>> &inputShapes = tensor->InputShape();
        SNN_ASSERT(inputShapes.size() == 1);
        const std::vector<int> &inputShape = inputShapes[0];
        const std::vector<int> &outputShape = tensor->OutputShape();
        const std::vector<int> &kernelVectShape = tensor->KernelShape();
        mGWS[0] = UP_DIV(outputShape[3], 4) * UP_DIV(outputShape[2], 4);
        mGWS[1] = outputShape[0] * outputShape[1];
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
        int imageShape[2] = {UP_DIV(outputShape.at(3), 4) * outputShape.at(2), outputShape.at(0) * outputShape.at(1)};
        // All values get from tensor and the class object inputCLData and outputCLData as
        cl_mem outputCLData = clCreateImage2D(*GPUcontext, CL_MEM_READ_WRITE, &clImageFormat, imageShape[0], imageShape[1], 0, NULL, &err);
        const cl_mem &mFilter = tensor->GetDeviceFilter();
        const cl_mem &mBias = tensor->GetDeviceBias();
        this->mbackend->CopyToDevice(tensor.get());
        this->inputCLData = tensor->GetDeviceInputData();
        tensor->SetDeviceOutputData(outputCLData);
        this->outputCLData = tensor->GetDeviceOutputData();
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[0]);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[1]);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), this->inputCLData);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &mFilter);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &mBias);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), this->outputCLData);
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
        mLWS = mOpenCLRuntime->localWS2DDefault(mGWS, mMaxWorkGroupSize, mOpenCLRuntime, kernelName, mKernel).first;
        err |= clFinish(commandQueue[0]);
        oclCheckError(err, CL_SUCCESS);
        // Testing ..
        // int buffer_sizes = inputShape[0] * inputShape[1] * inputShape[2] * inputShape[3] * sizeof(float);
        // float *inpu_data = (float *)malloc(buffer_sizes);
        // FILE *ptr;
        // const char *char_path = "/aidata/anders/data_collection/okay/total/archives/WF/scale_down/tflite/stem_outputs.bin";
        // ptr = fopen(char_path, "rb");
        // fread(inpu_data, buffer_sizes, 1, ptr);
        // cl_mem mhostBuffer = clCreateBuffer(*GPUcontext, CL_MEM_READ_WRITE, buffer_sizes, NULL, &err);
        // err |= clEnqueueWriteBuffer(commandQueue[0], mhostBuffer, CL_TRUE, 0, buffer_sizes, inpu_data, 0, NULL, NULL);
        // oclCheckError(err, CL_SUCCESS);
        // mbackend->mHostBuffer.first = buffer_sizes;
        // mbackend->mHostBuffer.second = mhostBuffer;
        // cl_mem intputImageData = mbackend->ConvertNHWCBufferToImage(inputShape, tensor->data_format, false, false);
        // tensor->SetDeviceInputData(intputImageData);
        // DataFormat data_format = tensor->data_format;
        // // implement depth-wise convolution
        // idx = 0;
        // err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[0]);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[1]);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &intputImageData);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &mFilter);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &mBias);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &outputCLData);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(inputImageShape), inputImageShape);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(inputChannelBlocks), inputChannelBlocks);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(outputImageShape), outputImageShape);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(kernelShape), kernelShape);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(paddingShape), paddingShape);
        // oclCheckError(err, CL_SUCCESS);
        // const size_t internalGlobalWS[2] = {mGWS[0], mGWS[1]};
        // // const size_t lws[2] = {mLWS[0], mLWS[1]};
        // const size_t lws[2] = {5, 5};
        // err |= clEnqueueNDRangeKernel(commandQueue[0], mKernel, 2, NULL, internalGlobalWS, lws, 0, NULL, NULL);
        // oclCheckError(err, CL_SUCCESS);
        // err |= clFinish(commandQueue[0]);
        // tensor->SetDeviceOutputData(outputCLData);
        // std::set<std::string> mBuildOptions;
        // mBuildOptions.emplace("-DBUFFER_IMAGE_IO_TRANS");
        // cl_kernel imageToBufferKernel = mOpenCLRuntime->BuildKernel("buffer_to_image", "image_to_nhwc_buffer", mBuildOptions);
        // mImageConvert->ConvertImageToNHWCBuffer(tensor, imageToBufferKernel, mOpenCLRuntime, false, false);
        // exit(1);
        return true;
    }
    bool DepthwiseConvExecution::onExecute(std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors)
    {
        int numInput = input_tensors.size();
        // std::cout << numInput << std::endl;
        SNN_ASSERT(numInput == 1);
        std::shared_ptr<Tensor> input_tensor = input_tensors[0];
        std::shared_ptr<Tensor> output_tensor = output_tensors[0];
        const std::vector<std::vector<int>> &inputShapes = input_tensor->InputShape();
        const std::vector<int> &inputShape = inputShapes[0];
        // const std::vector<std::vector<int>> &inputShapes = output_tensor->InputShape();
        // const std::vector<int> &inputShape = inputShapes[0];
        this->inputCLData = input_tensor->GetDeviceOutputData();
        SNN_ASSERT(inputCLData != NULL);
        cl_int err = CL_SUCCESS;
        err |= clSetKernelArg(mKernel, 2, sizeof(cl_mem), this->inputCLData);
        mOpenCLRuntime->RunKernel2D(this->mKernel, mGWS, mLWS, mOpenCLRuntime);
        oclCheckError(err, CL_SUCCESS);
        output_tensor->SetDeviceOutputData(*this->outputCLData);
        output_tensor->SetDeviceInputData(*this->inputCLData);
        bool status = true;
        // printf("-------------------------------------------------------------------\n");
        // std::set<std::string> mBuildOptions;
        // mBuildOptions.emplace("-DBUFFER_IMAGE_IO_TRANS");
        // cl_kernel imageToBufferKernel = mOpenCLRuntime->BuildKernel("buffer_to_image", "image_to_nhwc_buffer", mBuildOptions);
        // float *outputData = mImageConvert->ConvertImageToNHWCBuffer(output_tensor, imageToBufferKernel, mOpenCLRuntime, false, false);
        // for (int i = 0; i < 30; i++)
        // {

        //     std::cout << outputData[i] << std::endl;
        // }
        // free(outputData);
        if (err != CL_SUCCESS)
            return false;
        return status;

        return true;
    }
}