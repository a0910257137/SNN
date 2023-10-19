#include "EltwiseExecution.h"
#include "backend/opencl/core/ImageBufferConverter.h"

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
        cl_context &GPUcontext = mOpenCLRuntime->GetGPUContext();
        cl_command_queue *commandQueue = mOpenCLRuntime->GetCommandQue();
        cl_image_format clImageFormat;
        clImageFormat.image_channel_order = CL_RGBA;
        clImageFormat.image_channel_data_type = CL_FLOAT;
        int imageShape[2] = {UP_DIV(outputShape.at(3), 4) * outputShape.at(2), outputShape.at(0) * outputShape.at(1)};
        cl_mem output = clCreateImage2D(GPUcontext, CL_MEM_READ_WRITE, &clImageFormat, imageShape[0], imageShape[1], 0, NULL, &err);
        if (inputShapes.size() == 2)
        {

            // cl_mem input0 = clCreateImage2D(GPUcontext, CL_MEM_READ_WRITE, &clImageFormat, imageShape[0], imageShape[1], 0, NULL, &err);
            // cl_mem input1 = clCreateImage2D(GPUcontext, CL_MEM_READ_WRITE, &clImageFormat, imageShape[0], imageShape[1], 0, NULL, &err);
            // cl_mem output = clCreateImage2D(GPUcontext, CL_MEM_READ_WRITE, &clImageFormat, imageShape[0], imageShape[1], 0, NULL, &err);
            fullCount[0] = this->RealSize(inputShapes[0]) == 1 ? 0 : 1;
            fullCount[1] = this->RealSize(inputShapes[1]) == 1 ? 0 : 1;
            // err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[0]);
            // err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[1]);
            // err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &input0);
            // err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &input1);
            // err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &output);
            // err |= clSetKernelArg(mKernel, idx++, sizeof(shape), &shape);
            // err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &fullCount);
            // oclCheckError(err, CL_SUCCESS);
        }
        // std::string kernelName = "binary";
        // mLWS = mOpenCLRuntime->localWS2DDefault(mGWS, mMaxWorkGroupSize, mOpenCLRuntime, kernelName, mKernel).first;
        // Test..
        int buffer_sizes;
        buffer_sizes = inputShapes[0][0] * inputShapes[0][1] * inputShapes[0][2] * inputShapes[0][3] * sizeof(float);
        float *inpu_data0 = (float *)malloc(buffer_sizes);
        FILE *ptr;
        const char *char_path = "/aidata/anders/data_collection/okay/WF/archives/test/test_data/addn/320_320_3.bin";
        ptr = fopen(char_path, "rb");
        fread(inpu_data0, buffer_sizes, 1, ptr);
        cl_mem mhostBuffer;
        mhostBuffer = clCreateBuffer(GPUcontext, CL_MEM_READ_WRITE, buffer_sizes, NULL, &err);
        err |= clEnqueueWriteBuffer(commandQueue[0], mhostBuffer, CL_TRUE, 0, buffer_sizes, inpu_data0, 0, NULL, NULL);
        oclCheckError(err, CL_SUCCESS);
        mOpenCLBackend->mHostBuffer.first = buffer_sizes;
        mOpenCLBackend->mHostBuffer.second = mhostBuffer;
        cl_mem intputImageData0 = mOpenCLBackend->ConvertNHWCBufferToImage(inputShapes[0], tensor->data_format, false, false);
        buffer_sizes = inputShapes[1][0] * inputShapes[1][1] * inputShapes[1][2] * inputShapes[1][3] * sizeof(float);
        float *inpu_data1 = (float *)malloc(buffer_sizes);
        memcpy(inpu_data1, inpu_data0, buffer_sizes);
        mhostBuffer = clCreateBuffer(GPUcontext, CL_MEM_READ_WRITE, buffer_sizes, NULL, &err);
        err |= clEnqueueWriteBuffer(commandQueue[0], mhostBuffer, CL_TRUE, 0, buffer_sizes, inpu_data0, 0, NULL, NULL);
        oclCheckError(err, CL_SUCCESS);
        mOpenCLBackend->mHostBuffer.first = buffer_sizes;
        mOpenCLBackend->mHostBuffer.second = mhostBuffer;
        cl_mem intputImageData1 = mOpenCLBackend->ConvertNHWCBufferToImage(inputShapes[1], tensor->data_format, false, false);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[0]);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[1]);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &intputImageData0);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &intputImageData1);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &output);
        err |= clSetKernelArg(mKernel, idx++, sizeof(shape), &shape);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &fullCount);
        oclCheckError(err, CL_SUCCESS);

        size_t internalGlobalWS[2] = {mGWS[0], mGWS[1]};
        size_t lws[3] = {5, 5};
        err |= clEnqueueNDRangeKernel(commandQueue[0], mKernel, 2, NULL, internalGlobalWS, lws, 0, NULL, NULL);
        oclCheckError(err, CL_SUCCESS);
        err |= clFinish(commandQueue[0]);
        oclCheckError(err, CL_SUCCESS);
        tensor->SetDeviceOutputData(output);
        ImageBufferConverter mImageConvert(mOpenCLRuntime);
        std::set<std::string> buildOptions;
        buildOptions.emplace("-DBUFFER_IMAGE_IO_TRANS");
        cl_kernel imageToBufferKernel = mOpenCLRuntime->BuildKernel("buffer_to_image", "image_to_nhwc_buffer", buildOptions);
        mImageConvert.ConvertImageToNHWCBuffer(tensor, imageToBufferKernel, mOpenCLRuntime, false, false);
        exit(1);
    }
    bool EltwiseExecution::onExecute()
    {
    }
} // namespace SNN
