#include "InterpExecution.h"
#include "backend/opencl/core/ImageBufferConverter.h"
namespace SNN
{
    InterpExecution::InterpExecution(std::shared_ptr<Tensor> tensor, OpenCLBackend *mbackend) : Execution(mbackend)
    {
        this->mOpenCLBackend = mbackend;
        int opType = tensor->GetOpType();
        const std::vector<std::vector<int>> &inputShapes = tensor->InputShape();
        SNN_ASSERT(inputShapes.size() == 1);
        const std::vector<int> &inputShape = inputShapes[0];
        const std::vector<int> &outputShape = tensor->OutputShape();
        float heightScale = (float)inputShape[1] / (float)outputShape[1];
        float widthScale = (float)inputShape[2] / (float)outputShape[2];
        std::set<std::string> buildOptions;
        std::string kernelName = "interp";
        if (opType == OpTypes::RESIZE_NEAREST_NEIGHBOR)
        {
            mCordTransform[0] = widthScale;
            mCordTransform[1] = widthScale < 1.0f ? -0.25 : 0.25;
            mCordTransform[2] = heightScale;
            mCordTransform[3] = heightScale < 1.0f ? -0.25 : 0.25;
            mKernel = mOpenCLRuntime->BuildKernel("nearest", kernelName, buildOptions);
        }
        else
        {
            mKernel = mOpenCLRuntime->BuildKernel("interp", kernelName, buildOptions);
        }
        mMaxWorkGroupSize = static_cast<size_t>(mOpenCLRuntime->getMaxWorkGroupSize(mKernel));
    }
    bool InterpExecution::onResize(std::shared_ptr<Tensor> tensor)
    {
        const std::vector<std::vector<int>> &inputShapes = tensor->InputShape();
        SNN_ASSERT(inputShapes.size() == 1);
        const std::vector<int> &inputShape = inputShapes[0];
        const std::vector<int> &outputShape = tensor->OutputShape();
        const int inputBatch = inputShape[0];
        const int inputHeight = inputShape[1];
        const int inputWidth = inputShape[2];
        const int inputChannel = inputShape[3];
        const int channelBlocks = UP_DIV(inputChannel, 4);
        const int outputHeight = outputShape[1];
        const int outputWidth = outputShape[2];
        mGWS = {static_cast<size_t>(channelBlocks), static_cast<size_t>(outputWidth), static_cast<size_t>(outputHeight * inputBatch)};
        SNN_ASSERT(outputHeight > 0 && outputWidth > 0);
        cl_int err = 0;
        uint32_t idx = 0;
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
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), this->outputCLData);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mCordTransform[2]);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mCordTransform[0]);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mCordTransform[3]);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mCordTransform[1]);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &inputHeight);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &inputWidth);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &outputHeight);
        std::string kernelName = "interp";
        mLWS = mOpenCLRuntime->localWS3DDefault(mGWS, mMaxWorkGroupSize, mOpenCLRuntime, kernelName, mKernel).first;
        err |= clFinish(commandQueue[0]);
        oclCheckError(err, CL_SUCCESS);
        // exit(1);
        // Testing ..
        // int buffer_sizes = inputShape[0] * inputShape[1] * inputShape[2] * inputShape[3] * sizeof(float);
        // float *inpu_data = (float *)malloc(buffer_sizes);
        // FILE *ptr;
        // const char *char_path = "/aidata/anders/data_collection/okay/WF/archives/test/test_data/resize/160_160_3.bin";
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
        // // implement nearest resizing  images
        // err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[0]);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[1]);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[2]);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &intputImageData);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &outputCLData);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mCordTransform[2]);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mCordTransform[0]);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mCordTransform[3]);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mCordTransform[1]);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(int), &inputHeight);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(int), &inputWidth);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(int), &outputHeight);
        // oclCheckError(err, CL_SUCCESS);
        // const size_t internalGlobalWS[3] = {mGWS[0], mGWS[1], mGWS[2]};
        // // printf("%d\n", inputHeight);
        // // printf("%d\n", inputWidth);
        // // printf("%d\n", outputHeight);
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
        return true;
    }
    bool InterpExecution::onExecute(std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors)
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
