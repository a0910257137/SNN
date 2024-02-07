#include "SeperableConvExecution.h"
namespace SNN
{
    SeperableConvExecution::SeperableConvExecution(std::vector<std::shared_ptr<Tensor>> &tensors, OpenCLBackend *mbackend) : Execution(mbackend)
    {
        mConvCommon = std::make_shared<ConvolutionCommon>();
        numTensors = tensors.size();
        kernelName;

        if ((numTensors == 3) && (tensors[0]->GetOpType() == DEPTHWISECONV2D) && (tensors[1]->GetOpType() == CONV2D) && (tensors[2]->GetOpType() == CONV2D))
        {
            kernelName = "dw1_s11k33_conv2_s11k11_conv3_s11k11";
            const std::vector<int> &dw1_outputShape = tensors[0]->OutputShape();
            const std::vector<int> &conv2_outputShape = tensors[1]->OutputShape();
            int dw1_outChannels = UP_DIV(dw1_outputShape[3], 4);
            int conv2_outChannels = UP_DIV(conv2_outputShape[3], 4);
            std::string dw1_temp = "-DDW1_TEMP" + std::to_string(dw1_outChannels);
            std::string conv2_temp = "-DCONV2_TEMP" + std::to_string(conv2_outChannels);
            mBuildOptions.emplace(dw1_temp);
            mBuildOptions.emplace(conv2_temp);
            mBuildOptions.emplace("-DBIAS");
            if (tensors[0]->GetActType() == kActRelu)
                mBuildOptions.emplace("-DDW1_RELU");
            else if (tensors[0]->GetActType() == kActRelu6)
                mBuildOptions.emplace("-DDW1_RELU6");
            else if (tensors[0]->GetActType() == kActSigmoid)
                mBuildOptions.emplace("-DDW1_SIGMOID");
            if (tensors[1]->GetActType() == kActRelu)
                mBuildOptions.emplace("-DCONV2_RELU");
            else if (tensors[1]->GetActType() == kActRelu6)
                mBuildOptions.emplace("-DCONV2_RELU6");
            else if (tensors[1]->GetActType() == kActSigmoid)
                mBuildOptions.emplace("-DCONV2_SIGMOID");

            if (tensors[2]->GetActType() == kActRelu)
                mBuildOptions.emplace("-DCONV3_RELU");
            else if (tensors[2]->GetActType() == kActRelu6)
                mBuildOptions.emplace("-DCONV3_RELU6");
            else if (tensors[2]->GetActType() == kActSigmoid)
                mBuildOptions.emplace("-DCONV3_SIGMOID");
        }
        else if ((numTensors == 3) && (tensors[0]->GetOpType() == DEPTHWISECONV2D) && (tensors[1]->GetOpType() == DEPTHWISECONV2D) && (tensors[2]->GetOpType() == DEPTHWISECONV2D))
        {
            kernelName = "dw_s11k33_dw_s11k11_dw_s11k11";
            mBuildOptions.emplace("-DDW1_TEMP8");
            mBuildOptions.emplace("-DCONV2_TEMP4");
            mBuildOptions.emplace("-DBIAS");
            if (tensors[0]->GetActType() == kActRelu)
                mBuildOptions.emplace("-DDW1_RELU");
            else if (tensors[0]->GetActType() == kActRelu6)
                mBuildOptions.emplace("-DDW1_RELU6");
            else if (tensors[0]->GetActType() == kActSigmoid)
                mBuildOptions.emplace("-DDW1_SIGMOID");
            if (tensors[1]->GetActType() == kActRelu)
                mBuildOptions.emplace("-DDW2_RELU");
            else if (tensors[1]->GetActType() == kActRelu6)
                mBuildOptions.emplace("-DDW2_RELU6");
            else if (tensors[1]->GetActType() == kActSigmoid)
                mBuildOptions.emplace("-DDW2_SIGMOID");
            if (tensors[2]->GetActType() == kActRelu)
                mBuildOptions.emplace("-DDW3_RELU");
            else if (tensors[2]->GetActType() == kActRelu6)
                mBuildOptions.emplace("-DDW3_RELU6");
            else if (tensors[2]->GetActType() == kActSigmoid)
                mBuildOptions.emplace("-DDW3_SIGMOID");
        }
        else if ((numTensors == 2) && (tensors[0]->GetOpType() == DEPTHWISECONV2D) && (tensors[1]->GetOpType() == CONV2D))
        {
            const std::vector<int> &dw1_outputShape = tensors[0]->OutputShape();
            int dw1_outChannels = UP_DIV(dw1_outputShape[3], 4);
            std::string dw1_temp = "-DDW1_TEMP" + std::to_string(dw1_outChannels);
            kernelName = "dw1_s11k33_conv2_s11k11";
            mBuildOptions.emplace("-DBIAS");
            mBuildOptions.emplace(dw1_temp);
            mBuildOptions.emplace("-DCONV2_TEMP4");
            mBuildOptions.emplace("-DBIAS");
            if (tensors[0]->GetActType() == kActRelu)
                mBuildOptions.emplace("-DDW1_RELU");
            else if (tensors[0]->GetActType() == kActRelu6)
                mBuildOptions.emplace("-DDW1_RELU6");
            else if (tensors[0]->GetActType() == kActSigmoid)
                mBuildOptions.emplace("-DDW1_SIGMOID");
            if (tensors[1]->GetActType() == kActRelu)
                mBuildOptions.emplace("-DCONV2_RELU");
            else if (tensors[1]->GetActType() == kActRelu6)
                mBuildOptions.emplace("-DCONV2_RELU6");
            else if (tensors[1]->GetActType() == kActSigmoid)
                mBuildOptions.emplace("-DCONV2_SIGMOID");
        }
        mKernel = mOpenCLRuntime->BuildKernel("optimization", kernelName, mBuildOptions);
        mMaxWorkGroupSize = static_cast<size_t>(mOpenCLRuntime->getMaxWorkGroupSize(mKernel));
    }

    bool SeperableConvExecution::onOptimizedResize(std::vector<std::shared_ptr<Tensor>> &tensors)
    {
        cl_int err = 0;
        uint32_t idx = 0;
        if ((numTensors == 3) && (tensors[0]->GetOpType() == DEPTHWISECONV2D) && (tensors[1]->GetOpType() == CONV2D) && (tensors[2]->GetOpType() == CONV2D))
        {
            const std::vector<std::vector<int>> &dw1_inputShapes = tensors[0]->InputShape();
            const std::vector<std::vector<int>> &conv2_inputShapes = tensors[1]->InputShape();
            const std::vector<std::vector<int>> &conv3_inputShapes = tensors[2]->InputShape();
            SNN_ASSERT(dw1_inputShapes.size() == 1);
            SNN_ASSERT(conv2_inputShapes.size() == 1);
            SNN_ASSERT(conv3_inputShapes.size() == 1);
            const std::vector<int> &dw1_inputShape = dw1_inputShapes[0];
            const std::vector<int> &conv2_inputShape = conv2_inputShapes[0];
            const std::vector<int> &conv3_inputShape = conv3_inputShapes[0];

            const std::vector<int> &dw1_outputShape = tensors[0]->OutputShape();
            const std::vector<int> &conv2_outputShape = tensors[1]->OutputShape();
            const std::vector<int> &conv3_outputShape = tensors[2]->OutputShape();
            // kernel vector
            const std::vector<int> &dw1_kernelVectShape = tensors[0]->KernelShape();
            const std::vector<int> &conv2_kernelVectShape = tensors[1]->KernelShape();
            const std::vector<int> &conv3_kernelVectShape = tensors[2]->KernelShape();
            // input shape
            int dw1_inputImageShape[2] = {dw1_inputShape[1], dw1_inputShape[2]};
            int conv2_inputImageShape[2] = {conv2_inputShape[1], conv2_inputShape[2]};
            int conv3_inputImageShape[2] = {conv3_inputShape[1], conv3_inputShape[2]};
            // output shape
            int dw1_outputImageShape[2] = {dw1_outputShape[1], dw1_outputShape[2]};
            int conv2_outputImageShape[2] = {conv2_outputShape[1], conv2_outputShape[2]};
            int conv3_outputImageShape[2] = {conv3_outputShape[1], conv3_outputShape[2]};
            //  stride
            int dw1_strideShape[2] = {tensors[0]->stride(0), tensors[0]->stride(1)};
            int conv2_strideShape[2] = {tensors[1]->stride(0), tensors[1]->stride(1)};
            int conv3_strideShape[2] = {tensors[2]->stride(0), tensors[2]->stride(1)};
            // kernel shape
            int dw1_kernelShape[2] = {dw1_kernelVectShape[2], dw1_kernelVectShape[3]};
            int conv2_kernelShape[2] = {conv2_kernelVectShape[2], conv2_kernelVectShape[3]};
            int conv3_kernelShape[2] = {conv3_kernelVectShape[2], conv3_kernelVectShape[3]};
            // dilation
            int dw1_dilationShape[2] = {tensors[0]->dilation(0), tensors[0]->dilation(1)};
            int conv2_dilationShape[2] = {tensors[1]->dilation(0), tensors[1]->dilation(1)};
            int conv3_dilationShape[2] = {tensors[2]->dilation(0), tensors[2]->dilation(1)};

            std::pair<int, int> dw1_padding = mConvCommon->GetPadding(tensors[0]);
            int dw1_paddingShape[2] = {dw1_padding.first, dw1_padding.second};
            const int dw1_inputChannels = dw1_inputShape[3];
            const int dw1_inputChannelBlocks[1] = {UP_DIV(dw1_inputChannels, 4)};
            mGWS[0] = UP_DIV(dw1_outputShape[2], 4); // pesudo oc/4 w/4
            mGWS[1] = dw1_outputShape[0] * dw1_outputShape[1];
            this->inputCLData = *tensors[0]->GetDeviceInputData();
            this->outputCLData = *tensors[2]->GetDeviceOutputData();
            const cl_mem &dw1_Filter = tensors[0]->GetDeviceFilter();
            const cl_mem &dw1_Bias = tensors[0]->GetDeviceBias();
            const cl_mem &conv2_Filter = tensors[1]->GetDeviceFilter();
            const cl_mem &conv2_Bias = tensors[1]->GetDeviceBias();
            const cl_mem &conv3_Filter = tensors[2]->GetDeviceFilter();
            const cl_mem &conv3_Bias = tensors[2]->GetDeviceBias();

            int transShape[4] = {dw1_outputShape[3], conv2_outputShape[3], conv3_outputShape[3], 0};
            err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[0]);
            err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[1]);
            err |= clSetKernelArg(mKernel, idx++, sizeof(transShape), transShape);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &this->inputCLData);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &this->outputCLData);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &dw1_Filter);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &dw1_Bias);
            err |= clSetKernelArg(mKernel, idx++, sizeof(dw1_inputImageShape), dw1_inputImageShape);
            err |= clSetKernelArg(mKernel, idx++, sizeof(dw1_inputChannelBlocks), dw1_inputChannelBlocks);
            err |= clSetKernelArg(mKernel, idx++, sizeof(dw1_outputImageShape), dw1_outputImageShape);
            err |= clSetKernelArg(mKernel, idx++, sizeof(dw1_kernelShape), dw1_kernelShape);
            err |= clSetKernelArg(mKernel, idx++, sizeof(dw1_paddingShape), dw1_paddingShape);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &conv2_Filter);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &conv2_Bias);
            err |= clSetKernelArg(mKernel, idx++, sizeof(conv2_inputImageShape), conv2_inputImageShape);
            err |= clSetKernelArg(mKernel, idx++, sizeof(conv2_outputImageShape), conv2_outputImageShape);
            err |= clSetKernelArg(mKernel, idx++, sizeof(conv2_strideShape), &conv2_strideShape);
            size_t width4 = UP_DIV(conv2_outputShape[2], 4);
            err |= clSetKernelArg(mKernel, idx++, sizeof(int), &width4);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &conv3_Filter);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &conv3_Bias);
            err |= clSetKernelArg(mKernel, idx++, sizeof(conv3_inputImageShape), conv3_inputImageShape);
            err |= clSetKernelArg(mKernel, idx++, sizeof(conv3_outputImageShape), conv3_outputImageShape);
            err |= clSetKernelArg(mKernel, idx++, sizeof(conv3_strideShape), &conv3_strideShape);
            oclCheckError(err, CL_SUCCESS);
            mLWS = mOpenCLRuntime->localWS2DDefault(mGWS, mMaxWorkGroupSize, mOpenCLRuntime, kernelName, mKernel).first;
            // err |= clFinish(commandQueue[0]);
            // size_t internalGWS[2] = {mGWS[0], mGWS[1]};
            // err |= clEnqueueNDRangeKernel(commandQueue[0], mKernel, 2, NULL, internalGWS, NULL, 0, nullptr, NULL);
            // err |= clFinish(commandQueue[0]);
            // tensors[2]->SetDeviceInputData(this->inputCLData);
            // tensors[2]->SetDeviceOutputData(this->outputCLData);
            // std::set<std::string> buildOptions;
            // buildOptions.emplace("-DBUFFER_IMAGE_IO_TRANS");
            // cl_kernel imageToBufferKernel = mOpenCLRuntime->BuildKernel("buffer_to_image", "image_to_nhwc_buffer", buildOptions);
            // float *outputData = mImageConvert->ConvertImageToNHWCBuffer(tensors[2], imageToBufferKernel, mOpenCLRuntime, false, false);
            // // int size = 160 * 160 * 96;
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
        else if ((numTensors == 3) && (tensors[0]->GetOpType() == DEPTHWISECONV2D) && (tensors[1]->GetOpType() == DEPTHWISECONV2D) && (tensors[2]->GetOpType() == DEPTHWISECONV2D))
        {
            const std::vector<std::vector<int>> &dw1_inputShapes = tensors[0]->InputShape();
            const std::vector<std::vector<int>> &dw2_inputShapes = tensors[1]->InputShape();
            const std::vector<std::vector<int>> &dw3_inputShapes = tensors[2]->InputShape();
            SNN_ASSERT(dw1_inputShapes.size() == 1);
            SNN_ASSERT(dw2_inputShapes.size() == 1);
            SNN_ASSERT(dw3_inputShapes.size() == 1);
            // get input and  output
            const std::vector<int> &dw1_inputShape = dw1_inputShapes[0];
            const std::vector<int> &dw2_inputShape = dw2_inputShapes[0];
            const std::vector<int> &dw3_inputShape = dw3_inputShapes[0];
            const std::vector<int> &dw1_outputShape = tensors[0]->OutputShape();
            const std::vector<int> &dw2_outputShape = tensors[1]->OutputShape();
            const std::vector<int> &dw3_outputShape = tensors[2]->OutputShape();
            // kernel vector
            const std::vector<int> &dw1_kernelVectShape = tensors[0]->KernelShape();
            const std::vector<int> &dw2_kernelVectShape = tensors[1]->KernelShape();
            const std::vector<int> &dw3_kernelVectShape = tensors[2]->KernelShape();
            // input shape
            int dw1_inputImageShape[2] = {dw1_inputShape[1], dw1_inputShape[2]};
            int dw2_inputImageShape[2] = {dw2_inputShape[1], dw2_inputShape[2]};
            int dw3_inputImageShape[2] = {dw3_inputShape[1], dw3_inputShape[2]};
            // output shape
            int dw1_outputImageShape[2] = {dw1_outputShape[1], dw1_outputShape[2]};
            int dw2_outputImageShape[2] = {dw2_outputShape[1], dw2_outputShape[2]};
            int dw3_outputImageShape[2] = {dw3_outputShape[1], dw3_outputShape[2]};
            //  stride
            int dw1_strideShape[2] = {tensors[0]->stride(0), tensors[0]->stride(1)};
            int dw2_strideShape[2] = {tensors[1]->stride(0), tensors[1]->stride(1)};
            int dw3_strideShape[2] = {tensors[2]->stride(0), tensors[2]->stride(1)};
            // kernel shape
            int dw1_kernelShape[2] = {dw1_kernelVectShape[2], dw1_kernelVectShape[3]};
            int dw2_kernelShape[2] = {dw2_kernelVectShape[2], dw2_kernelVectShape[3]};
            int dw3_kernelShape[2] = {dw3_kernelVectShape[2], dw3_kernelVectShape[3]};
            // dilation
            int dw1_dilationShape[2] = {tensors[0]->dilation(0), tensors[0]->dilation(1)};
            int dw2_dilationShape[2] = {tensors[1]->dilation(0), tensors[1]->dilation(1)};
            int dw3_dilationShape[2] = {tensors[2]->dilation(0), tensors[2]->dilation(1)};
            std::pair<int, int> dw1_padding = mConvCommon->GetPadding(tensors[0]);
            int dw1_paddingShape[2] = {dw1_padding.first, dw1_padding.second};
            const int dw1_inputChannels = dw1_inputShape[3];
            const int dw1_inputChannelBlocks[1] = {UP_DIV(dw1_inputChannels, 4)};
            std::pair<int, int> dw2_padding = mConvCommon->GetPadding(tensors[1]);
            int dw2_paddingShape[2] = {dw2_padding.first, dw2_padding.second};
            const int dw2_inputChannels = dw2_inputShape[3];
            const int dw2_inputChannelBlocks[1] = {UP_DIV(dw2_inputChannels, 4)};
            mGWS[0] = UP_DIV(dw1_outputShape[3], 4) * UP_DIV(dw1_outputShape[2], 4); // pesudo oc/4 w/4
            mGWS[1] = dw1_outputShape[0] * dw1_outputShape[1];
            this->inputCLData = *tensors[0]->GetDeviceInputData();
            this->outputCLData = *tensors[2]->GetDeviceOutputData();
            const cl_mem &dw1_Filter = tensors[0]->GetDeviceFilter();
            const cl_mem &dw1_Bias = tensors[0]->GetDeviceBias();
            const cl_mem &dw2_Filter = tensors[1]->GetDeviceFilter();
            const cl_mem &dw2_Bias = tensors[1]->GetDeviceBias();
            const cl_mem &dw3_Filter = tensors[2]->GetDeviceFilter();
            const cl_mem &dw3_Bias = tensors[2]->GetDeviceBias();

            int transShape[4] = {dw1_outputShape[3], dw2_outputShape[3], dw3_outputShape[3], 0};
            err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[0]);
            err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[1]);
            err |= clSetKernelArg(mKernel, idx++, sizeof(transShape), transShape);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &this->inputCLData);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &this->outputCLData);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &dw1_Filter);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &dw1_Bias);
            err |= clSetKernelArg(mKernel, idx++, sizeof(dw1_inputImageShape), dw1_inputImageShape);
            err |= clSetKernelArg(mKernel, idx++, sizeof(dw1_inputChannelBlocks), dw1_inputChannelBlocks);
            err |= clSetKernelArg(mKernel, idx++, sizeof(dw1_outputImageShape), dw1_outputImageShape);
            err |= clSetKernelArg(mKernel, idx++, sizeof(dw1_kernelShape), dw1_kernelShape);
            err |= clSetKernelArg(mKernel, idx++, sizeof(dw1_paddingShape), dw1_paddingShape);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &dw2_Filter);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &dw2_Bias);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &dw3_Filter);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &dw3_Bias);
            oclCheckError(err, CL_SUCCESS);
            mLWS = mOpenCLRuntime->localWS2DDefault(mGWS, mMaxWorkGroupSize, mOpenCLRuntime, kernelName, mKernel).first;
            err |= clFinish(commandQueue[0]);
            size_t internalGWS[2] = {mGWS[0], mGWS[1]};
            size_t internalLWS[2] = {mLWS[0], mLWS[1]};
            oclCheckError(err, CL_SUCCESS);
            // cl_event event;
            // err |= clEnqueueNDRangeKernel(commandQueue[0], mKernel, 2, NULL, internalGWS, internalLWS, 0, nullptr, NULL);
            // err |= clFinish(commandQueue[0]);
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
            // int size = 80 * 80 * 96;
            // FILE *pfile;
            // pfile = fopen("/aidata/anders/data_collection/okay/WF/archives/test/test_data/optimization/optimized.binary", "wb");
            // fwrite(outputData, 1, size * sizeof(float), pfile);
            // fclose(pfile);
        }
        else if ((numTensors == 2) && (tensors[0]->GetOpType() == DEPTHWISECONV2D) && (tensors[1]->GetOpType() == CONV2D))
        {

            const std::vector<std::vector<int>> &dw1_inputShapes = tensors[0]->InputShape();
            const std::vector<std::vector<int>> &conv2_inputShapes = tensors[1]->InputShape();
            SNN_ASSERT(dw1_inputShapes.size() == 1);
            SNN_ASSERT(conv2_inputShapes.size() == 1);
            const std::vector<int> &dw1_inputShape = dw1_inputShapes[0];
            const std::vector<int> &conv2_inputShape = conv2_inputShapes[0];

            const std::vector<int> &dw1_outputShape = tensors[0]->OutputShape();
            const std::vector<int> &conv2_outputShape = tensors[1]->OutputShape();
            // kernel vector
            const std::vector<int> &dw1_kernelVectShape = tensors[0]->KernelShape();
            const std::vector<int> &conv2_kernelVectShape = tensors[1]->KernelShape();
            // input shape
            int dw1_inputImageShape[2] = {dw1_inputShape[1], dw1_inputShape[2]};
            int conv2_inputImageShape[2] = {conv2_inputShape[1], conv2_inputShape[2]};
            // output shape
            int dw1_outputImageShape[2] = {dw1_outputShape[1], dw1_outputShape[2]};
            int conv2_outputImageShape[2] = {conv2_outputShape[1], conv2_outputShape[2]};
            //  stride
            int dw1_strideShape[2] = {tensors[0]->stride(0), tensors[0]->stride(1)};
            int conv2_strideShape[2] = {tensors[1]->stride(0), tensors[1]->stride(1)};
            // kernel shape
            int dw1_kernelShape[2] = {dw1_kernelVectShape[2], dw1_kernelVectShape[3]};
            int conv2_kernelShape[2] = {conv2_kernelVectShape[2], conv2_kernelVectShape[3]};
            // dilation
            int dw1_dilationShape[2] = {tensors[0]->dilation(0), tensors[0]->dilation(1)};
            int conv2_dilationShape[2] = {tensors[1]->dilation(0), tensors[1]->dilation(1)};

            std::pair<int, int> dw1_padding = mConvCommon->GetPadding(tensors[0]);
            int dw1_paddingShape[2] = {dw1_padding.first, dw1_padding.second};
            const int dw1_inputChannels = dw1_inputShape[3];
            const int dw1_inputChannelBlocks[1] = {UP_DIV(dw1_inputChannels, 4)};
            mGWS[0] = UP_DIV(dw1_outputShape[2], 4); // pesudo oc/4 w/4
            mGWS[1] = dw1_outputShape[0] * dw1_outputShape[1];
            this->inputCLData = *tensors[0]->GetDeviceInputData();
            this->outputCLData = *tensors[1]->GetDeviceOutputData();
            const cl_mem &dw1_Filter = tensors[0]->GetDeviceFilter();
            const cl_mem &dw1_Bias = tensors[0]->GetDeviceBias();
            const cl_mem &conv2_Filter = tensors[1]->GetDeviceFilter();
            const cl_mem &conv2_Bias = tensors[1]->GetDeviceBias();
            int transShape[4] = {dw1_outputShape[3], conv2_outputShape[3], 0, 0};
            err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[0]);
            err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[1]);
            err |= clSetKernelArg(mKernel, idx++, sizeof(transShape), transShape);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &this->inputCLData);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &this->outputCLData);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &dw1_Filter);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &dw1_Bias);
            err |= clSetKernelArg(mKernel, idx++, sizeof(dw1_inputImageShape), dw1_inputImageShape);
            err |= clSetKernelArg(mKernel, idx++, sizeof(dw1_inputChannelBlocks), dw1_inputChannelBlocks);
            err |= clSetKernelArg(mKernel, idx++, sizeof(dw1_outputImageShape), dw1_outputImageShape);
            err |= clSetKernelArg(mKernel, idx++, sizeof(dw1_kernelShape), dw1_kernelShape);
            err |= clSetKernelArg(mKernel, idx++, sizeof(dw1_paddingShape), dw1_paddingShape);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &conv2_Filter);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &conv2_Bias);
            err |= clSetKernelArg(mKernel, idx++, sizeof(conv2_inputImageShape), conv2_inputImageShape);
            err |= clSetKernelArg(mKernel, idx++, sizeof(conv2_outputImageShape), conv2_outputImageShape);
            err |= clSetKernelArg(mKernel, idx++, sizeof(conv2_strideShape), &conv2_strideShape);
            oclCheckError(err, CL_SUCCESS);
            mLWS = mOpenCLRuntime->localWS2DDefault(mGWS, mMaxWorkGroupSize, mOpenCLRuntime, kernelName, mKernel).first;
            err |= clFinish(commandQueue[0]);
            // size_t internalGWS[2] = {mGWS[0], mGWS[1]};
            // size_t internalLWS[2] = {mLWS[0], mLWS[1]};
            // cl_event event;
            // err |= clEnqueueNDRangeKernel(commandQueue[0], mKernel, 2, NULL, internalGWS, NULL, 0, nullptr, &event);
            // err = clWaitForEvents(1, &event);
            // cl_ulong time_start, time_end;
            // float total_time;
            // clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
            // clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
            // total_time = (time_end - time_start) / 1000000.0f;
            // printf("%f\n", total_time);
            // tensors[1]->SetDeviceInputData(this->inputCLData);
            // tensors[1]->SetDeviceOutputData(this->outputCLData);
            // std::set<std::string> buildOptions;
            // buildOptions.emplace("-DBUFFER_IMAGE_IO_TRANS");
            // cl_kernel imageToBufferKernel = mOpenCLRuntime->BuildKernel("buffer_to_image", "image_to_nhwc_buffer", buildOptions);
            // float *outputData = mImageConvert->ConvertImageToNHWCBuffer(tensors[1], imageToBufferKernel, mOpenCLRuntime, false, false);
            // int size = 40 * 40 * 32;
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
        return true;
    }
    bool SeperableConvExecution::onOptimizedExecute(std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors)
    {
        int numInput = input_tensors.size();
        int numOutput = output_tensors.size();
        this->inputCLData = *(input_tensors[numInput - 1]->GetDeviceOutputData());
        SNN_ASSERT(inputCLData != NULL);
        cl_int err = CL_SUCCESS;
        err |= clSetKernelArg(mKernel, 3, sizeof(cl_mem), &this->inputCLData);
        mOpenCLRuntime->RunKernel2D(this->mKernel, mGWS, mLWS, mOpenCLRuntime);
        output_tensors[numOutput - 1]->SetDeviceOutputData(this->outputCLData);
        output_tensors[numOutput - 1]->SetDeviceInputData(this->inputCLData);
        bool status = true;
        if (err != CL_SUCCESS)
            return false;
        return status;
    }
}