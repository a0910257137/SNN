#include "EltwiseExecution.h"

namespace SNN
{
    EltwiseExecution::EltwiseExecution(std::shared_ptr<Tensor> tensor, OpenCLBackend *mbackend, const std::string &compute) : Execution(mbackend)
    {
        this->mOpenCLBackend = mbackend;
        std::string kernelName = "binary";
        mCompute = compute;
        mBuildOptions.emplace("-DOPERATOR=" + compute);
        // mMWGS
        if (tensor->GetActType() == kActRelu)
            mBuildOptions.emplace("-DRELU");
        else if (tensor->GetActType() == kActRelu6)
            mBuildOptions.emplace("-DRELU6");
        else if (tensor->GetActType() == kActSigmoid)
            mBuildOptions.emplace("-DSIGMOID");
        mKernel = mOpenCLRuntime->BuildKernel("binary", kernelName, mBuildOptions);
        mMaxWorkGroupSize = static_cast<size_t>(mOpenCLRuntime->getMaxWorkGroupSize(mKernel));
    }
    uint32_t EltwiseExecution::RealSize(const std::vector<int> &inputShape)
    {
        uint32_t num = 1;
        for (int i = 0; i < inputShape.size(); i++)
        {
            num *= inputShape[i];
        }
        return num;
    }
    bool EltwiseExecution::CheckSize(const std::vector<std::vector<int>> &inputShapes, const std::vector<int> &outputShape)
    {
        int i, j, chs = 0;
        int numInputs = inputShapes.size();
        for (i = 0; i < numInputs; i++)
        {
            SNN_ASSERT(inputShapes[i].size() == outputShape.size());
            for (j = 0; j < 4; j++)
            {
                if (inputShapes[i][j] != outputShape[j])
                    return false;
            }
        }
        return true;
    }
    bool EltwiseExecution::onResize(std::shared_ptr<Tensor> tensor)
    {
        const std::vector<std::vector<int>> &inputShapes = tensor->InputShape();
        const std::vector<int> &outputShape = tensor->OutputShape();
        SNN_ASSERT(inputShapes.size() >= 2);
        bool err_size = this->CheckSize(inputShapes, outputShape);
        SNN_ASSERT(err_size);
        int shape[4] = {outputShape[0], outputShape[1], outputShape[2], UP_DIV(outputShape[3], 4)};
        int fullCount[2] = {1, 1};
        int activationType = 0;
        mGWS = {(size_t)UP_DIV(outputShape[3], 4) * outputShape[2],
                (size_t)outputShape[0] * outputShape[1]};
        uint32_t idx = 0;
        cl_int err = 0;
        int imageShape[2] = {UP_DIV(outputShape.at(3), 4) * outputShape.at(2), outputShape.at(0) * outputShape.at(1)};
        this->outputCLData = clCreateImage2D(*GPUcontext, CL_MEM_READ_WRITE, &clImageFormat, imageShape[0], imageShape[1], 0, NULL, &err);
        this->inputCLData0 = clCreateImage2D(*GPUcontext, CL_MEM_READ_WRITE, &clImageFormat, imageShape[0], imageShape[1], 0, NULL, &err);
        this->inputCLData1 = clCreateImage2D(*GPUcontext, CL_MEM_READ_WRITE, &clImageFormat, imageShape[0], imageShape[1], 0, NULL, &err);
        tensor->SetDeviceOutputData(this->outputCLData);
        // this->outputCLData = tensor->GetDeviceOutputData();
        // this->inputCLData0 = &input0;
        // this->inputCLData1 = &input1;
        if (inputShapes.size() == 2)
        {
            fullCount[0] = this->RealSize(inputShapes[0]) == 1 ? 0 : 1;
            fullCount[1] = this->RealSize(inputShapes[1]) == 1 ? 0 : 1;
            err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[0]);
            err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[1]);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &this->inputCLData0);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &this->inputCLData1);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &this->outputCLData);
            err |= clSetKernelArg(mKernel, idx++, sizeof(shape), &shape);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &fullCount);
            err |= clSetKernelArg(mKernel, idx++, sizeof(int), &activationType);
            oclCheckError(err, CL_SUCCESS);
        }
        std::string kernelName = "binary";
        mLWS = mOpenCLRuntime->localWS2DDefault(mGWS, mMaxWorkGroupSize, mOpenCLRuntime, kernelName, mKernel).first;
        // exit(1);
        // Test..
        // int buffer_sizes;
        // buffer_sizes = inputShapes[0][0] * inputShapes[0][1] * inputShapes[0][2] * inputShapes[0][3] * sizeof(float);
        // float *inpu_data0 = (float *)malloc(buffer_sizes);
        // FILE *ptr;
        // const char *char_path = "/aidata/anders/data_collection/okay/WF/archives/test/test_data/addn/320_320_3.bin";
        // ptr = fopen(char_path, "rb");
        // fread(inpu_data0, buffer_sizes, 1, ptr);
        // cl_mem mhostBuffer;
        // mhostBuffer = clCreateBuffer(GPUcontext, CL_MEM_READ_WRITE, buffer_sizes, NULL, &err);
        // err |= clEnqueueWriteBuffer(commandQueue[0], mhostBuffer, CL_TRUE, 0, buffer_sizes, inpu_data0, 0, NULL, NULL);
        // oclCheckError(err, CL_SUCCESS);
        // mOpenCLBackend->mHostBuffer.first = buffer_sizes;
        // mOpenCLBackend->mHostBuffer.second = mhostBuffer;
        // cl_mem intputImageData0 = mOpenCLBackend->ConvertNHWCBufferToImage(inputShapes[0], tensor->data_format, false, false);
        // buffer_sizes = inputShapes[1][0] * inputShapes[1][1] * inputShapes[1][2] * inputShapes[1][3] * sizeof(float);
        // float *inpu_data1 = (float *)malloc(buffer_sizes);
        // memcpy(inpu_data1, inpu_data0, buffer_sizes);
        // mhostBuffer = clCreateBuffer(GPUcontext, CL_MEM_READ_WRITE, buffer_sizes, NULL, &err);
        // err |= clEnqueueWriteBuffer(commandQueue[0], mhostBuffer, CL_TRUE, 0, buffer_sizes, inpu_data0, 0, NULL, NULL);
        // oclCheckError(err, CL_SUCCESS);
        // mOpenCLBackend->mHostBuffer.first = buffer_sizes;
        // mOpenCLBackend->mHostBuffer.second = mhostBuffer;
        // cl_mem intputImageData1 = mOpenCLBackend->ConvertNHWCBufferToImage(inputShapes[1], tensor->data_format, false, false);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[0]);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[1]);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &intputImageData0);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &intputImageData1);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &output);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(shape), &shape);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &fullCount);
        // oclCheckError(err, CL_SUCCESS);
        // size_t internalGlobalWS[2] = {mGWS[0], mGWS[1]};
        // size_t lws[3] = {5, 5};
        // err |= clEnqueueNDRangeKernel(commandQueue[0], mKernel, 2, NULL, internalGlobalWS, lws, 0, NULL, NULL);
        // oclCheckError(err, CL_SUCCESS);
        // err |= clFinish(commandQueue[0]);
        // oclCheckError(err, CL_SUCCESS);
        // tensor->SetDeviceOutputData(output);
        // ImageBufferConverter mImageConvert(mOpenCLRuntime);
        // std::set<std::string> buildOptions;
        // buildOptions.emplace("-DBUFFER_IMAGE_IO_TRANS");
        // cl_kernel imageToBufferKernel = mOpenCLRuntime->BuildKernel("buffer_to_image", "image_to_nhwc_buffer", buildOptions);
        // mImageConvert.ConvertImageToNHWCBuffer(tensor, imageToBufferKernel, mOpenCLRuntime, false, false);
        // exit(1);
        return true;
    }
    bool EltwiseExecution::onExecute(std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors)
    {
        SNN_ASSERT(input_tensors.size() == 2);
        std::shared_ptr<Tensor> input_tensor0 = input_tensors[0];
        std::shared_ptr<Tensor> input_tensor1 = input_tensors[1];
        std::shared_ptr<Tensor> output_tensor = output_tensors[0];
        this->inputCLData0 = *input_tensor0->GetDeviceOutputData();
        this->inputCLData1 = *input_tensor1->GetDeviceOutputData();
        cl_int err = CL_SUCCESS;
        err |= clSetKernelArg(mKernel, 2, sizeof(cl_mem), &this->inputCLData0);
        err |= clSetKernelArg(mKernel, 3, sizeof(cl_mem), &this->inputCLData1);
        mOpenCLRuntime->RunKernel2D(this->mKernel, mGWS, mLWS, mOpenCLRuntime);
        oclCheckError(err, CL_SUCCESS);
        output_tensor->SetDeviceOutputData(this->outputCLData);
        // printf("-------------------------------------------------------------------\n");
        // std::set<std::string> buildOptions;
        // buildOptions.emplace("-DBUFFER_IMAGE_IO_TRANS");
        // cl_kernel imageToBufferKernel = mOpenCLRuntime->BuildKernel("buffer_to_image", "image_to_nhwc_buffer", buildOptions);
        // mImageConvert->ConvertImageToNHWCBuffer(output_tensor, imageToBufferKernel, mOpenCLRuntime, false, false);
        // exit(1);
        bool status = true;
        if (err != CL_SUCCESS)
            return false;
        return true;
    }
    bool EltwiseExecution::onOptimizedExecute(std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors)
    {
        int numInput = input_tensors.size();
        int numOutput = output_tensors.size();
        SNN_ASSERT(input_tensors.size() == 2);
        this->inputCLData0 = *(input_tensors[0]->GetDeviceOutputData());
        this->inputCLData1 = *(input_tensors[1]->GetDeviceOutputData());
        SNN_ASSERT(inputCLData0 != NULL);
        SNN_ASSERT(inputCLData1 != NULL);
        cl_int err = CL_SUCCESS;
        err |= clSetKernelArg(mKernel, 2, sizeof(cl_mem), &this->inputCLData0);
        err |= clSetKernelArg(mKernel, 3, sizeof(cl_mem), &this->inputCLData1);
        mOpenCLRuntime->RunKernel2D(this->mKernel, mGWS, mLWS, mOpenCLRuntime);
        output_tensors[numOutput - 1]->SetDeviceOutputData(this->outputCLData);
        bool status = true;
        if (err != CL_SUCCESS)
            return false;
        return status;
    }
} // namespace SNN
