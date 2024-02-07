
#include "InputExecution.h"

namespace SNN
{
    InputExecution::InputExecution(std::shared_ptr<Tensor> tensor, OpenCLBackend *mbackend) : Execution(mbackend)
    {
        this->mbackend = mbackend;
        const std::vector<std::vector<int>> &inputShapes = tensor->InputShape();
        SNN_ASSERT(inputShapes.size() == 1);
        const std::vector<int> &inputShape = inputShapes[0];
        int N = inputShape[0];
        int H = inputShape[1];
        int W = inputShape[2];
        int C = inputShape[3];
        bufferSize = N * H * W * C * sizeof(float);

        std::set<std::string> buildOptions;
        if (mOpenCLRuntime->isSupportedFP16() == false)
        {
            buildOptions.emplace("-DBUFFER_IMAGE_IO_TRANS");
        }
        mKernel = mOpenCLRuntime->BuildKernel("buffer_to_image", "nhwc_buffer_to_image", buildOptions);
    }
    bool InputExecution::onResize(std::shared_ptr<Tensor> tensor)
    {
        cl_int err = 0;
        const std::vector<std::vector<int>> &inputShapes = tensor->InputShape();
        SNN_ASSERT(inputShapes.size() == 1);
        const std::vector<int> &inputShape = inputShapes[0];
        const std::vector<int> outputShape = TensorShapeFormat(inputShape, tensor->data_format);
        size_t imageWidth = outputShape[2] * UP_DIV(outputShape[3], 4), imageHeight = outputShape[0] * outputShape[1];
        int outputGlobalWorkSize[2] = {static_cast<int>(UP_DIV(outputShape[3], 4) * outputShape[2]),
                                       static_cast<int>(outputShape[0] * outputShape[1])};
        int N = inputShape[0], H = inputShape[1], W = inputShape[2], C = inputShape[3];

        bufferSize = N * H * W * C * sizeof(float);
        this->inputCLData = clCreateBuffer(*GPUcontext, CL_MEM_READ_WRITE, bufferSize, NULL, &err);
        this->outputCLData = clCreateImage2D(*GPUcontext, CL_MEM_READ_WRITE, &clImageFormat, imageWidth, imageHeight, 0, NULL, &err);
        uint32_t idx = 0;
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &outputGlobalWorkSize[0]);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &outputGlobalWorkSize[1]);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &this->inputCLData);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &outputShape[1]);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &outputShape[2]);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &outputShape[3]);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &this->outputCLData);
        oclCheckError(err, CL_SUCCESS);
        const uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLRuntime->getMaxWorkGroupSize(mKernel));
        size_t roundUpGroupWorkSize[2] = {};
        // size_t lws[2] = {16, MAX((size_t)1, maxWorkGroupSize / 16)};
        mLWS = {16, MAX((size_t)1, maxWorkGroupSize / 16)};
        for (size_t i = 0; i < 2; ++i)
        {
            mGWS[i] = ROUND_UP(outputGlobalWorkSize[i], mLWS[i]);
        }
        return true;
    }
    bool InputExecution::onExecute(std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors)
    {

        return true;
    }
    bool InputExecution::onInputExecute(float *input_data, std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors)
    {
        cl_int err = CL_SUCCESS;
        SNN_ASSERT(input_tensors.size() == 1);
        SNN_ASSERT(output_tensors.size() == 1);
        std::shared_ptr<Tensor> input_tensor = input_tensors[0];
        std::shared_ptr<Tensor> output_tensor = output_tensors[0];
        err |= clEnqueueWriteBuffer(commandQueue[0], this->inputCLData, CL_TRUE, 0, bufferSize, input_data, 0, NULL, NULL);
        oclCheckError(err, CL_SUCCESS);
        err |= clSetKernelArg(mKernel, 2, sizeof(cl_mem), &this->inputCLData);
        mOpenCLRuntime->RunKernel2D(mKernel, mGWS, mLWS, mOpenCLRuntime);
        oclCheckError(err, CL_SUCCESS);
        output_tensor->SetDeviceInputData(this->inputCLData);
        output_tensor->SetDeviceOutputData(this->outputCLData);
        // std::set<std::string> buildOptions;
        // buildOptions.emplace("-DBUFFER_IMAGE_IO_TRANS");
        // cl_kernel imageToBufferKernel = mOpenCLRuntime->BuildKernel("buffer_to_image", "image_to_nhwc_buffer", buildOptions);
        // float *outputData = mImageConvert->ConvertImageToNHWCBuffer(output_tensor, imageToBufferKernel, mOpenCLRuntime, false, false);
        // for (int i = 0; i < 10; i++)
        // {
        //     std::cout << outputData[i] << std::endl;
        // }
        // free(outputData);
        // clReleaseMemObject(inputCLData);
        return true;
    }
} // namespace SNN
