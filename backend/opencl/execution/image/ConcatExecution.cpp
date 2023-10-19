#include "ConcatExecution.h"
#include "backend/opencl/core/ImageBufferConverter.h"
namespace SNN
{

    ConcatExecution::ConcatExecution(std::shared_ptr<Tensor> tensor, OpenCLBackend *mbackend) : Execution(mbackend)
    {
        this->mOpenCLBackend = mbackend;
        axis = tensor->GetConcatAxis();
        const std::vector<std::vector<int>> &inputShapes = tensor->InputShape();
        numInputs = inputShapes.size();
        std::set<std::string> buildOptions;
        std::string kernelName;
        if (numInputs == 2)
            kernelName = "concat_channel";
        if (numInputs > 2)
            kernelName = "concat_channel_multi";
        if (inputShapes[0].at(3) % 4 == 0)
        {
            buildOptions.emplace("-DDIVISIBLE_FOUR");
        }
        mKernel = mOpenCLRuntime->BuildKernel("concat", kernelName, buildOptions);
        mMaxWorkGroupSize = static_cast<size_t>(mOpenCLRuntime->getMaxWorkGroupSize(mKernel));
    }

    bool ConcatExecution::onResize(std::shared_ptr<Tensor> tensor)
    {

        const std::vector<std::vector<int>> &inputShapes = tensor->InputShape();
        const std::vector<int> &outputShape = tensor->OutputShape();
        SNN_ASSERT(numInputs > 1);
        // only support channel concat and we check shape
        int i, j, chs = 0;
        for (i = 0; i < numInputs; i++)
        {
            SNN_ASSERT(inputShapes[i].size() == outputShape.size());
            for (j = 0; j < 3; j++)
            {
                if (inputShapes[i][j] != outputShape[j])
                    return false;
            }
            chs += inputShapes[i][3];
        }
        if (chs != outputShape[3])
            return false;
        const int batch = outputShape[0];
        const int height = outputShape[1];
        const int width = outputShape[2];
        const int channel = outputShape[3];
        const int channel_blk = UP_DIV(channel, 4);
        mGWS = {
            static_cast<size_t>(channel_blk),
            static_cast<size_t>(width),
            static_cast<size_t>(batch * height),
        };
        if (mKernel == NULL)
        {
            printf("ERROR: conat kernel is NULL !!");
            return false;
        }
        std::string kernelName;
        cl_int err = 0;
        uint32_t idx = 0;
        cl_context &GPUcontext = mOpenCLRuntime->GetGPUContext();
        cl_command_queue *commandQueue = mOpenCLRuntime->GetCommandQue();
        cl_image_format clImageFormat;
        clImageFormat.image_channel_order = CL_RGBA;
        clImageFormat.image_channel_data_type = CL_FLOAT;
        int outputImageShape[2] = {UP_DIV(channel, 4) * width, batch * height};
        cl_mem outputCLData = clCreateImage2D(GPUcontext, CL_MEM_READ_WRITE, &clImageFormat, outputImageShape[0], outputImageShape[1], 0, NULL, &err);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[0]);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[1]);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[2]);
        if (numInputs == 2)
        {
            kernelName = "concat_channel";
            int inputImageShape0[2] = {UP_DIV(inputShapes[0].at(3), 4) * inputShapes[0].at(2), inputShapes[0].at(0) * inputShapes[0].at(1)};
            int inputImageShape1[2] = {UP_DIV(inputShapes[1].at(3), 4) * inputShapes[1].at(2), inputShapes[1].at(0) * inputShapes[1].at(1)};
            cl_mem inputCLData0 = clCreateImage2D(GPUcontext, CL_MEM_READ_WRITE, &clImageFormat, inputImageShape0[0], inputImageShape0[1], 0, NULL, &err);
            cl_mem inputCLData1 = clCreateImage2D(GPUcontext, CL_MEM_READ_WRITE, &clImageFormat, inputImageShape1[0], inputImageShape1[1], 0, NULL, &err);

            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &inputCLData0);
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &inputCLData1);
            err |= clSetKernelArg(mKernel, idx++, sizeof(int), &inputShapes[0].at(3));
            err |= clSetKernelArg(mKernel, idx++, sizeof(int), &inputShapes[1].at(3));
            err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &outputCLData);
            oclCheckError(err, CL_SUCCESS);
        }
        else
        {
            printf("ERORR: Not support multi-channels ! \n");
            SNN_ASSERT(false);
        }
        mLWS = mOpenCLRuntime->localWS2DDefault(mGWS, mMaxWorkGroupSize, mOpenCLRuntime, kernelName, mKernel).first;
        err |= clFinish(commandQueue[0]);
        oclCheckError(err, CL_SUCCESS);
        // Test ...
        // int buffer_sizes;
        // buffer_sizes = inputShapes[0][0] * inputShapes[0][1] * inputShapes[0][2] * inputShapes[0][3] * sizeof(float);
        // float *inpu_data0 = (float *)malloc(buffer_sizes);
        // FILE *ptr;
        // const char *char_path = "/aidata/anders/data_collection/okay/WF/archives/test/test_data/concat/320_320_3.bin";
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
        // err |= clSetKernelArg(mKernel, idx++, sizeof(int), &mGWS[2]);
        // kernelName = "concat_channel";
        // err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &intputImageData0);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &intputImageData1);
        // err |= clSetKernelArg(mKernel, idx++, sizeof(int), &inputShapes[0].at(3));
        // err |= clSetKernelArg(mKernel, idx++, sizeof(int), &inputShapes[1].at(3));
        // err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &outputCLData);
        // oclCheckError(err, CL_SUCCESS);
        // size_t internalGlobalWS[3] = {mGWS[0], mGWS[1], mGWS[2]};
        // size_t lws[3] = {1, 5, 5};
        // err |= clEnqueueNDRangeKernel(commandQueue[0], mKernel, 3, NULL, internalGlobalWS, lws, 0, NULL, NULL);
        // oclCheckError(err, CL_SUCCESS);
        // err |= clFinish(commandQueue[0]);
        // oclCheckError(err, CL_SUCCESS);
        // tensor->SetDeviceOutputData(outputCLData);
        // ImageBufferConverter mImageConvert(mOpenCLRuntime);
        // std::set<std::string> buildOptions;
        // buildOptions.emplace("-DBUFFER_IMAGE_IO_TRANS");
        // cl_kernel imageToBufferKernel = mOpenCLRuntime->BuildKernel("buffer_to_image", "image_to_nhwc_buffer", buildOptions);
        // mImageConvert.ConvertImageToNHWCBuffer(tensor, imageToBufferKernel, mOpenCLRuntime, false, false);
        // exit(1);
        return true;
    }

    bool ConcatExecution::onExecute()
    {
    }

    bool ConcatExecution::Concat2(std::shared_ptr<Tensor> tensor)
    {
        // if(tensor->is_init)
        // {

        // }
    }
} // namespace SNN
