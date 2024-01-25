#include "StemExecution.h"
namespace SNN
{
    StemExecution::StemExecution(std::vector<std::shared_ptr<Tensor>> &tensors, OpenCLBackend *mbackend) : Execution(mbackend)
    {
        mConvCommon = std::make_shared<ConvolutionCommon>();
        const std::vector<std::vector<int>> &inputShapes = tensors[0]->InputShape();
        const std::vector<int> &inputShape = inputShapes[0];
        int N = inputShape[0];
        int H = inputShape[1];
        int W = inputShape[2];
        int C = inputShape[3];
        bufferSize = N * H * W * C * sizeof(float);
        kernelName = "stem_conv_2d_c8h4w1";
        mBuildOptions.emplace("-DDW1_TEMP8");
        mBuildOptions.emplace("-DCONV2_TEMP8");
        mBuildOptions.emplace("-DBIAS");
        if (tensors[1]->GetActType() == kActRelu)
            mBuildOptions.emplace("-DRELU");
        else if (tensors[1]->GetActType() == kActRelu6)
            mBuildOptions.emplace("-DRELU6");
        else if (tensors[1]->GetActType() == kActSigmoid)
            mBuildOptions.emplace("-DSIGMOID");
        mBuildOptions.emplace("-DBUFFER_IMAGE_IO_TRANS");
        mKernel = mOpenCLRuntime->BuildKernel("optimization", kernelName, mBuildOptions);
        mMaxWorkGroupSize = static_cast<size_t>(mOpenCLRuntime->getMaxWorkGroupSize(mKernel));
    }
    bool StemExecution::onOptimizedResize(std::vector<std::shared_ptr<Tensor>> &tensors)
    {
        cl_int err = 0;
        uint32_t idx = 0;
        SNN_ASSERT(tensors.size() == 2);
        const std::vector<std::vector<int>> &inputShapes = tensors[0]->InputShape();
        const std::vector<std::vector<int>> &conv_inputShapes = tensors[1]->InputShape();
        // input shape
        const std::vector<int> &inputShape = inputShapes[0];
        const std::vector<int> &conv_inputShape = conv_inputShapes[0];
        // output shape
        const std::vector<int> &outputShape = tensors[0]->OutputShape();
        const std::vector<int> &conv_outputShape = tensors[1]->OutputShape();
        // kernel
        const std::vector<int> &conv_kernelVectShape = tensors[1]->KernelShape();
        // weights and bias
        const cl_mem &conv_Filter = tensors[1]->GetDeviceFilter();
        const cl_mem &conv_Bias = tensors[1]->GetDeviceBias();
        this->inputCLData = *tensors[0]->GetDeviceInputData();
        this->outputCLData = *tensors[1]->GetDeviceOutputData();
        const int inputChannelBlocks = UP_DIV(inputShape[3], 4);
        // implement C8H4W1
        int outShapes[2] = {conv_outputShape[1], conv_outputShape[2]};
        int kernelShape[2] = {conv_kernelVectShape[2], conv_kernelVectShape[3]}; // OIHW
        int strideShape[2] = {tensors[1]->stride(0), tensors[1]->stride(1)};     // hw
        int dilationShape[2] = {tensors[1]->dilation(0), tensors[1]->dilation(1)};
        auto padding = mConvCommon->GetPadding(tensors[1]); // (paddingY, paddingX)
        int paddingShape[2] = {padding.first, padding.second};
        int out_height_blk = UP_DIV(conv_outputShape[1], 4);
        int out_width_blk = UP_DIV(conv_outputShape[2], 1);
        int out_channel_blk = UP_DIV(conv_outputShape[3], 4);
        mGWS[0] = static_cast<size_t>(UP_DIV(conv_outputShape.at(3), 8) * UP_DIV(conv_outputShape.at(2), 1));
        mGWS[1] = static_cast<size_t>(conv_outputShape.at(0) * UP_DIV(conv_outputShape.at(1), 4));
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[0]);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[1]);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &this->inputCLData);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &inputShape[1]);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &inputShape[2]);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &inputShape[3]);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &conv_Filter);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &conv_Bias);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &inputChannelBlocks);
        err |= clSetKernelArg(mKernel, idx++, sizeof(outShapes), outShapes);
        err |= clSetKernelArg(mKernel, idx++, sizeof(kernelShape), &kernelShape);
        err |= clSetKernelArg(mKernel, idx++, sizeof(strideShape), &strideShape);
        err |= clSetKernelArg(mKernel, idx++, sizeof(paddingShape), &paddingShape);
        err |= clSetKernelArg(mKernel, idx++, sizeof(dilationShape), &dilationShape);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &out_width_blk);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &out_channel_blk);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &out_height_blk);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &this->outputCLData);
        oclCheckError(err, CL_SUCCESS);
        std::string kernelName = "stem_conv_2d_c8h4w1";
        mLWS = mOpenCLRuntime->localWS2DDefault(mGWS, mMaxWorkGroupSize, mOpenCLRuntime, kernelName, mKernel).first;
        // size_t internalGWS[2] = {mGWS[0], mGWS[1]};
        // size_t internalLWS[2] = {mLWS[0], mLWS[1]};
        // err |= clEnqueueNDRangeKernel(commandQueue[0], mKernel, 2, NULL, internalGWS, NULL, 0, nullptr, NULL);
        // oclCheckError(err, CL_SUCCESS);
        // tensors[1]->SetDeviceInputData(this->inputCLData);
        // tensors[1]->SetDeviceOutputData(this->outputCLData);
        // std::set<std::string> buildOptions;
        // buildOptions.emplace("-DBUFFER_IMAGE_IO_TRANS");
        // cl_kernel imageToBufferKernel = mOpenCLRuntime->BuildKernel("buffer_to_image", "image_to_nhwc_buffer", buildOptions);
        // float *outputData = mImageConvert->ConvertImageToNHWCBuffer(tensors[1], imageToBufferKernel, mOpenCLRuntime, false, false);
        // int size = 160 * 160 * 32;
        // FILE *pfile;
        // pfile = fopen("/aidata/anders/data_collection/okay/WF/archives/test/test_data/optimization/optimized.binary", "wb");
        // fwrite(outputData, 1, size * sizeof(float), pfile);
        // fclose(pfile);
        return true;
    }
    bool StemExecution::onInputExecute(float *input_data, std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors)
    {
        int numInput = input_tensors.size();
        int numOutput = output_tensors.size();
        SNN_ASSERT(numInput == 2);
        cl_int err = CL_SUCCESS;
        err |= clEnqueueWriteBuffer(commandQueue[0], this->inputCLData, CL_TRUE, 0, bufferSize, input_data, 0, NULL, NULL);
        oclCheckError(err, CL_SUCCESS);
        err |= clSetKernelArg(mKernel, 2, sizeof(cl_mem), &this->inputCLData);
        mOpenCLRuntime->RunKernel2D(mKernel, mGWS, mLWS, mOpenCLRuntime);
        oclCheckError(err, CL_SUCCESS);
        std::shared_ptr<Tensor> tensor = output_tensors[numOutput - 1];
        tensor->SetDeviceOutputData(this->outputCLData);
        tensor->SetDeviceInputData(this->inputCLData);
        bool status = true;
        if (err != CL_SUCCESS)
            return false;
        return status;
    }
}