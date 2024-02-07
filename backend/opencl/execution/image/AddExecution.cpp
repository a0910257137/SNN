#include "AddExecution.h"
namespace SNN
{
    AddExecution::AddExecution(std::vector<std::shared_ptr<Tensor>> &tensors, OpenCLBackend *mbackend) : Execution(mbackend)
    {
        mConvCommon = std::make_shared<ConvolutionCommon>();
        numTensors = tensors.size();
        if ((tensors[0]->GetOpType() == RESIZE_NEAREST_NEIGHBOR) && (tensors[1]->GetOpType() == CONV2D) && (tensors[2]->GetOpType() == ADD))
        {
            kernelName = "resize1_s11k11_conv2_s22_add";
            const std::vector<int> &resize1_outputShape = tensors[0]->OutputShape();
            const std::vector<int> &conv2_outputShape = tensors[1]->OutputShape();
            SNN_ASSERT(resize1_outputShape[3] == conv2_outputShape[3]);
            mBuildOptions.emplace("-DDW1_TEMP8");
            mBuildOptions.emplace("-DCONV2_TEMP8");
            mBuildOptions.emplace("-DBIAS");
            if (tensors[0]->GetActType() == kActRelu)
                mBuildOptions.emplace("-DCONV1_RELU");
            else if (tensors[0]->GetActType() == kActRelu6)
                mBuildOptions.emplace("-DCONV1_RELU6");
            else if (tensors[0]->GetActType() == kActSigmoid)
                mBuildOptions.emplace("-DCONV1_SIGMOID");
            const std::vector<std::vector<int>> &resize1_inputShapes = tensors[0]->InputShape();
            SNN_ASSERT(resize1_inputShapes.size() == 1);
            const std::vector<int> &resize1_inputShape = resize1_inputShapes[0];
            float heightScale = (float)resize1_inputShape[1] / (float)resize1_outputShape[1];
            float widthScale = (float)resize1_inputShape[2] / (float)resize1_outputShape[2];
            mCordTransform[0] = widthScale;
            mCordTransform[1] = widthScale < 1.0f ? 0.25 : -0.25;
            mCordTransform[2] = heightScale;
            mCordTransform[3] = heightScale < 1.0f ? 0.25 : -0.25;
        }
        else if ((tensors[0]->GetOpType() == CONV2D) && (tensors[1]->GetOpType() == CONV2D) && (tensors[2]->GetOpType() == ADD))
        {
            kernelName = "conv1_s11k11_premap_add";
            const std::vector<int> &dw1_outputShape = tensors[0]->OutputShape();
            const std::vector<int> &dw2_outputShape = tensors[1]->OutputShape();
            SNN_ASSERT(dw1_outputShape[3] == dw2_outputShape[3]);
            mBuildOptions.emplace("-DDW1_TEMP8");
            mBuildOptions.emplace("-DCONV2_TEMP8");
            mBuildOptions.emplace("-DBIAS");
            if (tensors[0]->GetActType() == kActRelu)
                mBuildOptions.emplace("-DCONV1_RELU");
            else if (tensors[0]->GetActType() == kActRelu6)
                mBuildOptions.emplace("-DCONV1_RELU6");
            if (tensors[2]->GetActType() == kActRelu)
                mBuildOptions.emplace("-DADD3_RELU");
            else if (tensors[2]->GetActType() == kActRelu6)
                mBuildOptions.emplace("-DADD3_RELU6");
        }
        else if (((tensors[0]->GetOpType() == DEPTHWISECONV2D) && (tensors[1]->GetOpType() == DEPTHWISECONV2D) && (tensors[2]->GetOpType() == ADD)) || ((tensors[0]->GetOpType() == DEPTHWISECONV2D) && (tensors[1]->GetOpType() == ADD) && (tensors[2]->GetOpType() == ADD)) || ((tensors[0]->GetOpType() == DEPTHWISECONV2D) && (tensors[1]->GetOpType() == CONV2D) && (tensors[2]->GetOpType() == ADD)))
        {
            kernelName = "dw1_s11k33_premap_add";
            const std::vector<int> &dw1_outputShape = tensors[0]->OutputShape();
            const std::vector<int> &premap_outputShape = tensors[1]->OutputShape();

            SNN_ASSERT(dw1_outputShape[1] == premap_outputShape[1]);
            SNN_ASSERT(dw1_outputShape[2] == premap_outputShape[2]);
            SNN_ASSERT(dw1_outputShape[3] == premap_outputShape[3]);
            mBuildOptions.emplace("-DDW1_TEMP8");
            mBuildOptions.emplace("-DCONV2_TEMP8");
            mBuildOptions.emplace("-DBIAS");
            if (tensors[0]->GetActType() == kActRelu)
                mBuildOptions.emplace("-DDW1_RELU");
            else if (tensors[0]->GetActType() == kActRelu6)
                mBuildOptions.emplace("-DDW1_RELU6");
            else if (tensors[0]->GetActType() == kActSigmoid)
                mBuildOptions.emplace("-DDW1_SIGMOID");
            if (tensors[2]->GetActType() == kActRelu)
                mBuildOptions.emplace("-DADD3_RELU");
            else if (tensors[2]->GetActType() == kActRelu6)
                mBuildOptions.emplace("-DADD3_RELU6");
            else if (tensors[2]->GetActType() == kActSigmoid)
                mBuildOptions.emplace("-DADD3_SIGMOID");
        }

        mKernel = mOpenCLRuntime->BuildKernel("optimization", kernelName, mBuildOptions);
        mMaxWorkGroupSize = static_cast<size_t>(mOpenCLRuntime->getMaxWorkGroupSize(mKernel));
    }

    bool AddExecution::onOptimizedResize(std::vector<std::shared_ptr<Tensor>> &tensors)
    {
        cl_int err = 0;
        uint32_t idx = 0;
        if ((tensors[0]->GetOpType() == RESIZE_NEAREST_NEIGHBOR) && (tensors[1]->GetOpType() == CONV2D) && (tensors[2]->GetOpType() == ADD))
        {
            const std::vector<std::vector<int>> &resize1_inputShapes = tensors[0]->InputShape();
            const std::vector<std::vector<int>> &conv2_inputShapes = tensors[1]->InputShape();
            const std::vector<std::vector<int>> &add3_inputShapes = tensors[2]->InputShape();
            SNN_ASSERT(resize1_inputShapes.size() == 1);
            SNN_ASSERT(add3_inputShapes.size() == 2);
            const std::vector<int> &resize1_inputShape = resize1_inputShapes[0];
            const std::vector<int> &resize1_outputShape = tensors[0]->OutputShape();
            const int resize1_inputBatch = resize1_inputShape[0];
            const int resize1_inputHeight = resize1_inputShape[1];
            const int resize1_inputWidth = resize1_inputShape[2];
            const int resize1_inputChannel = resize1_inputShape[3];
            SNN_ASSERT((resize1_outputShape[1] > 0) && (resize1_outputShape[2] > 0));
            mGWS = {static_cast<size_t>(UP_DIV(resize1_inputChannel, 4)), static_cast<size_t>(resize1_outputShape[2]), static_cast<size_t>(resize1_outputShape[1] * resize1_inputBatch)};
            this->inputCLData1 = *tensors[0]->GetDeviceInputData();
            this->inputCLData2 = *tensors[1]->GetDeviceOutputData();
            this->outputCLData = *tensors[2]->GetDeviceOutputData();
            err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[0]);
            err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[1]);
            err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[2]);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &this->inputCLData1);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &this->inputCLData2);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &this->outputCLData);
            err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mCordTransform[2]);
            err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mCordTransform[0]);
            err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mCordTransform[3]);
            err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mCordTransform[1]);
            err |= clSetKernelArg(mKernel, idx++, sizeof(int), &resize1_inputHeight);
            err |= clSetKernelArg(mKernel, idx++, sizeof(int), &resize1_inputWidth);
            err |= clSetKernelArg(mKernel, idx++, sizeof(int), &resize1_outputShape[1]);
            mLWS = mOpenCLRuntime->localWS3DDefault(mGWS, mMaxWorkGroupSize, mOpenCLRuntime, kernelName, mKernel).first;
            err |= clFinish(commandQueue[0]);
            oclCheckError(err, CL_SUCCESS);
            // size_t internalGWS[3] = {mGWS[0], mGWS[1], mGWS[2]};
            // size_t internalLWS[3] = {mLWS[0], mLWS[1], mLWS[2]};
            // cl_event event;
            // err |= clEnqueueNDRangeKernel(commandQueue[0], mKernel, 3, NULL, internalGWS, internalLWS, 0, nullptr, NULL);
            // err |= clFinish(commandQueue[0]);
            // err = clWaitForEvents(1, &event);
            // cl_ulong time_start, time_end;
            // float total_time;
            // clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
            // clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
            // total_time = (time_end - time_start) / 1000000.0f;
            // printf("%f\n", total_time);
            // tensors[2]->SetDeviceInputData(this->inputCLData1);
            // tensors[2]->SetDeviceOutputData(this->outputCLData);
            // std::set<std::string> buildOptions;
            // buildOptions.emplace("-DBUFFER_IMAGE_IO_TRANS");
            // cl_kernel imageToBufferKernel = mOpenCLRuntime->BuildKernel("buffer_to_image", "image_to_nhwc_buffer", buildOptions);
            // float *outputData = mImageConvert->ConvertImageToNHWCBuffer(tensors[2], imageToBufferKernel, mOpenCLRuntime, false, false);
            // int size = 20 * 20 * 32;
            // FILE *pfile;
            // pfile = fopen("/aidata/anders/data_collection/okay/WF/archives/test/test_data/optimization/optimized.binary", "wb");
            // fwrite(outputData, 1, size * sizeof(float), pfile);
            // fclose(pfile);
            // for (int i = 0; i < 34; i++)
            // {
            //     std::cout << outputData[i] << std::endl;
            // }
            // exit(1);
        }
        else if ((tensors[0]->GetOpType() == CONV2D) && (tensors[1]->GetOpType() == CONV2D) && (tensors[2]->GetOpType() == ADD))
        {
            const std::vector<std::vector<int>> &conv1_inputShapes = tensors[0]->InputShape();
            const std::vector<std::vector<int>> &conv2_inputShapes = tensors[1]->InputShape();
            const std::vector<std::vector<int>> &add3_inputShapes = tensors[2]->InputShape();
            SNN_ASSERT(conv1_inputShapes.size() == 1);
            SNN_ASSERT(conv2_inputShapes.size() == 1);
            SNN_ASSERT(add3_inputShapes.size() == 2);
            // input shape
            const std::vector<int> &conv1_inputShape = conv1_inputShapes[0];
            const std::vector<int> &conv2_inputShape = conv2_inputShapes[0];
            // output shape
            const std::vector<int> &conv1_outputShape = tensors[0]->OutputShape();
            const std::vector<int> &conv2_outputShape = tensors[1]->OutputShape();
            // kernel vector
            const std::vector<int> &conv1_kernelVectShape = tensors[0]->KernelShape();
            const std::vector<int> &conv2_kernelVectShape = tensors[1]->KernelShape();
            // input shape
            // default the same input size and size kernel stride and size
            int conv1_inputImageShape[2] = {conv1_inputShape[1], conv1_inputShape[2]};
            int conv2_inputImageShape[2] = {conv2_inputShape[1], conv2_inputShape[2]};
            // output shape
            int conv1_outputImageShape[2] = {conv1_outputShape[1], conv1_outputShape[2]};
            int conv2_outputImageShape[2] = {conv2_outputShape[1], conv2_outputShape[2]};
            //  stride
            int conv1_strideShape[2] = {tensors[0]->stride(0), tensors[0]->stride(1)};
            int conv2_strideShape[2] = {tensors[1]->stride(0), tensors[1]->stride(1)};
            // kernel shape
            int conv1_kernelShape[2] = {conv1_kernelVectShape[2], conv1_kernelVectShape[3]};
            int conv2_kernelShape[2] = {conv2_kernelVectShape[2], conv2_kernelVectShape[3]};
            this->inputCLData1 = *tensors[0]->GetDeviceInputData();
            this->inputCLData2 = *tensors[1]->GetDeviceOutputData();
            this->outputCLData = *tensors[2]->GetDeviceOutputData();
            const cl_mem &conv1_mFilter = tensors[0]->GetDeviceFilter();
            const cl_mem &conv1_mBias = tensors[0]->GetDeviceBias();
            // const cl_mem &conv2_mFilter = tensors[1]->GetDeviceFilter();
            // const cl_mem &conv2_mBias = tensors[1]->GetDeviceBias();
            int inputChannels = conv1_inputShape[3];
            const int inputChannelBlocks = UP_DIV(inputChannels, 4);
            mGWS = {
                static_cast<size_t>(UP_DIV(conv1_outputShape.at(3), 4) * static_cast<size_t>(UP_DIV(conv1_outputShape.at(2), 4))),
                static_cast<size_t>(conv1_outputShape.at(0) * conv1_outputShape.at(1))};
            int inputImageShape[2] = {conv1_inputShape.at(1), conv1_inputShape.at(2)};
            int outputImageShape[2] = {conv1_outputShape.at(1), conv1_outputShape.at(2)};
            err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[0]);
            err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[1]);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &this->inputCLData1); // 2
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &conv1_mFilter);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &conv1_mBias);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &this->inputCLData2); // 5
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &this->outputCLData);
            err |= clSetKernelArg(mKernel, idx++, sizeof(inputImageShape), inputImageShape);
            err |= clSetKernelArg(mKernel, idx++, sizeof(int), &inputChannelBlocks);
            err |= clSetKernelArg(mKernel, idx++, sizeof(outputImageShape), outputImageShape);
            err |= clSetKernelArg(mKernel, idx++, sizeof(conv1_strideShape), &conv1_strideShape);
            size_t width4 = UP_DIV(conv1_outputShape[2], 4);
            err |= clSetKernelArg(mKernel, idx++, sizeof(int), &width4);
            oclCheckError(err, CL_SUCCESS);
            mLWS = mOpenCLRuntime->localWS2DDefault(mGWS, mMaxWorkGroupSize, mOpenCLRuntime, kernelName, mKernel).first;
            err |= clFinish(commandQueue[0]);
            // size_t internalGWS[2] = {mGWS[0], mGWS[1]};
            // size_t internalLWS[2] = {mLWS[0], mLWS[1]};
            // // cl_event event;
            // err |= clEnqueueNDRangeKernel(commandQueue[0], mKernel, 2, NULL, internalGWS, internalLWS, 0, nullptr, NULL);
            // // err |= clFinish(commandQueue[0]);
            // // err = clWaitForEvents(1, &event);
            // // cl_ulong time_start, time_end;
            // // float total_time;
            // // clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
            // // clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
            // // total_time = (time_end - time_start) / 1000000.0f;
            // // printf("%f\n", total_time);
            // // exit(1);
            // printf("==========================================\n");
            // tensors[2]->SetDeviceInputData(this->inputCLData1);
            // tensors[2]->SetDeviceOutputData(this->outputCLData);
            // std::set<std::string> buildOptions;
            // buildOptions.emplace("-DBUFFER_IMAGE_IO_TRANS");
            // cl_kernel imageToBufferKernel = mOpenCLRuntime->BuildKernel("buffer_to_image", "image_to_nhwc_buffer", buildOptions);
            // float *outputData = mImageConvert->ConvertImageToNHWCBuffer(tensors[2], imageToBufferKernel, mOpenCLRuntime, false, false);
            // int size = 20 * 20 * 32;
            // FILE *pfile;
            // pfile = fopen("/aidata/anders/data_collection/okay/WF/archives/test/test_data/optimization/optimized.binary", "wb");
            // fwrite(outputData, 1, size * sizeof(float), pfile);
            // fclose(pfile);
            // for (int i = 0; i < 34; i++)
            // {
            //     std::cout << outputData[i] << std::endl;
            // }
            // exit(1);
        }
        else if (((tensors[0]->GetOpType() == DEPTHWISECONV2D) && (tensors[1]->GetOpType() == DEPTHWISECONV2D) && (tensors[2]->GetOpType() == ADD)) || ((tensors[0]->GetOpType() == DEPTHWISECONV2D) && (tensors[1]->GetOpType() == ADD) && (tensors[2]->GetOpType() == ADD)) || ((tensors[0]->GetOpType() == DEPTHWISECONV2D) && (tensors[1]->GetOpType() == CONV2D) && (tensors[2]->GetOpType() == ADD)))
        {
            const std::vector<std::vector<int>> &dw1_inputShapes = tensors[0]->InputShape();
            const std::vector<std::vector<int>> &add3_inputShapes = tensors[2]->InputShape();
            SNN_ASSERT(dw1_inputShapes.size() == 1);
            SNN_ASSERT(add3_inputShapes.size() == 2);
            const std::vector<int> &dw1_inputShape = dw1_inputShapes[0];
            const std::vector<int> &dw1_outputShape = tensors[0]->OutputShape();
            // kernel vector
            const std::vector<int> &dw1_kernelVectShape = tensors[0]->KernelShape();
            // input shape
            int dw1_inputImageShape[2] = {dw1_inputShape[1], dw1_inputShape[2]};
            // output shape
            int dw1_outputImageShape[2] = {dw1_outputShape[1], dw1_outputShape[2]};
            //  stride
            int dw1_strideShape[2] = {tensors[0]->stride(0), tensors[0]->stride(1)};
            // kernel shape
            int dw1_kernelShape[2] = {dw1_kernelVectShape[2], dw1_kernelVectShape[3]};
            // dilation
            int dw1_dilationShape[2] = {tensors[0]->dilation(0), tensors[0]->dilation(1)};
            std::pair<int, int> dw1_padding = mConvCommon->GetPadding(tensors[0]);
            int dw1_paddingShape[2] = {dw1_padding.first, dw1_padding.second};
            const int dw1_inputChannels = dw1_inputShape[3];
            const int dw1_inputChannelBlocks[1] = {UP_DIV(dw1_inputChannels, 4)};
            mGWS[0] = UP_DIV(dw1_outputShape[3], 4) * UP_DIV(dw1_outputShape[2], 4);
            mGWS[1] = dw1_outputShape[0] * dw1_outputShape[1];
            this->inputCLData1 = *tensors[0]->GetDeviceInputData();
            this->inputCLData2 = *tensors[1]->GetDeviceOutputData();
            this->outputCLData = *tensors[2]->GetDeviceOutputData();
            const cl_mem &dw1_Filter = tensors[0]->GetDeviceFilter();
            const cl_mem &dw1_Bias = tensors[0]->GetDeviceBias();
            err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[0]);
            err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[1]);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &this->inputCLData1);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &this->inputCLData2);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &this->outputCLData);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &dw1_Filter);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &dw1_Bias);
            err |= clSetKernelArg(mKernel, idx++, sizeof(dw1_inputImageShape), dw1_inputImageShape);
            err |= clSetKernelArg(mKernel, idx++, sizeof(dw1_inputChannelBlocks), dw1_inputChannelBlocks);
            err |= clSetKernelArg(mKernel, idx++, sizeof(dw1_outputImageShape), dw1_outputImageShape);
            err |= clSetKernelArg(mKernel, idx++, sizeof(dw1_kernelShape), dw1_kernelShape);
            err |= clSetKernelArg(mKernel, idx++, sizeof(dw1_paddingShape), dw1_paddingShape);
            mLWS = mOpenCLRuntime->localWS2DDefault(mGWS, mMaxWorkGroupSize, mOpenCLRuntime, kernelName, mKernel).first;
            err |= clFinish(commandQueue[0]);
            oclCheckError(err, CL_SUCCESS);
            // size_t internalGWS[2] = {mGWS[0], mGWS[1]};
            // size_t internalLWS[2] = {mLWS[0], mLWS[1]};
            // cl_event event;
            // err |= clEnqueueNDRangeKernel(commandQueue[0], mKernel, 2, NULL, internalGWS, internalLWS, 0, nullptr, NULL);
            // err |= clFinish(commandQueue[0]);
            // tensors[2]->SetDeviceInputData(this->inputCLData1);
            // tensors[2]->SetDeviceOutputData(this->outputCLData);
            // std::string kernelName = "dw_s11k33_conv_s11k11_conv_s11k11";
            // mLWS = mOpenCLRuntime->localWS2DDefault(mGWS, mMaxWorkGroupSize, mOpenCLRuntime, kernelName, mKernel).first;
            // err |= clFinish(commandQueue[0]);
            // size_t internalGWS[2] = {mGWS[0], mGWS[1]};
            // size_t internalLWS[2] = {mLWS[0], mLWS[1]};
            // cl_event event;
            // err |= clEnqueueNDRangeKernel(commandQueue[0], mKernel, 2, NULL, internalGWS, NULL, 0, nullptr, &event);
            // err |= clFinish(commandQueue[0]);
            // oclCheckError(err, CL_SUCCESS);
            // err = clWaitForEvents(1, &event);
            // cl_ulong time_start, time_end;
            // float total_time;
            // clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
            // clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
            // total_time = (time_end - time_start) / 1000000.0f;
            // printf("%f\n", total_time);
            // tensors[2]->SetDeviceInputData(this->inputCLData);
            // tensors[2]->SetDeviceOutputData(this->outputCLData);
            // std::set<std::string> buildOptions;
            // buildOptions.emplace("-DBUFFER_IMAGE_IO_TRANS");
            // cl_kernel imageToBufferKernel = mOpenCLRuntime->BuildKernel("buffer_to_image", "image_to_nhwc_buffer", buildOptions);
            // float *outputData = mImageConvert->ConvertImageToNHWCBuffer(tensors[2], imageToBufferKernel, mOpenCLRuntime, false, false);
            // // int size = 80 * 80 * 192;
            // // FILE *pfile;
            // // pfile = fopen("/aidata/anders/data_collection/okay/WF/archives/test/test_data/optimization/optimized.binary", "wb");
            // // fwrite(outputData, 1, size * sizeof(float), pfile);
            // // fclose(pfile);
            // for (int i = 0; i < 34; i++)
            // {
            //     std::cout << outputData[i] << std::endl;
            // }
            // exit(1);
        }

        return true;
    }
    bool AddExecution::onExecute(std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors)
    {
        return true;
    }
    bool AddExecution::onOptimizedExecute(std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors)
    {
        int numInput = input_tensors.size();
        int numOutput = output_tensors.size();
        SNN_ASSERT(input_tensors.size() == 2);
        this->inputCLData1 = *(input_tensors[0]->GetDeviceOutputData());
        this->inputCLData2 = *(input_tensors[1]->GetDeviceOutputData());
        SNN_ASSERT(inputCLData1 != NULL);
        SNN_ASSERT(inputCLData2 != NULL);
        cl_int err = CL_SUCCESS;
        if ((output_tensors[0]->GetOpType() == RESIZE_NEAREST_NEIGHBOR) && (output_tensors[1]->GetOpType() == CONV2D) && (output_tensors[2]->GetOpType() == ADD))
        {
            err |= clSetKernelArg(mKernel, 3, sizeof(cl_mem), &this->inputCLData1);
            err |= clSetKernelArg(mKernel, 4, sizeof(cl_mem), &this->inputCLData2);
            mOpenCLRuntime->RunKernel3D(this->mKernel, mGWS, mLWS, mOpenCLRuntime);
        }
        else if ((output_tensors[0]->GetOpType() == CONV2D) && (output_tensors[1]->GetOpType() == CONV2D) && (output_tensors[2]->GetOpType() == ADD))
        {

            err |= clSetKernelArg(mKernel, 2, sizeof(cl_mem), &this->inputCLData1);
            err |= clSetKernelArg(mKernel, 5, sizeof(cl_mem), &this->inputCLData2);
            mOpenCLRuntime->RunKernel2D(this->mKernel, mGWS, mLWS, mOpenCLRuntime);
        }
        else if (((output_tensors[0]->GetOpType() == DEPTHWISECONV2D) && (output_tensors[1]->GetOpType() == DEPTHWISECONV2D) && (output_tensors[2]->GetOpType() == ADD)) || ((output_tensors[0]->GetOpType() == DEPTHWISECONV2D) && (output_tensors[1]->GetOpType() == ADD) && (output_tensors[2]->GetOpType() == ADD)) || ((output_tensors[0]->GetOpType() == DEPTHWISECONV2D) && (output_tensors[1]->GetOpType() == CONV2D) && (output_tensors[2]->GetOpType() == ADD)))
        {

            err |= clSetKernelArg(mKernel, 2, sizeof(cl_mem), &this->inputCLData1);
            err |= clSetKernelArg(mKernel, 3, sizeof(cl_mem), &this->inputCLData2);
            mOpenCLRuntime->RunKernel2D(this->mKernel, mGWS, mLWS, mOpenCLRuntime);
        }
        output_tensors[numOutput - 1]->SetDeviceOutputData(this->outputCLData);
        bool status = true;
        if (err != CL_SUCCESS)
            return false;
        return status;
    }
}