#include "OpenCLBackend.h"

#include "include/SNN/SNNDefine.h"
namespace SNN
{
    OpenCLBackend::OpenCLBackend(bool permitFloat16)
    {
        this->permitFloat16 = permitFloat16;
        _mCLRuntime = new OpenCLRuntime();
        this->OnSetCache();
    }
    OpenCLBackend::~OpenCLBackend()
    {
        delete _mCLRuntime;
    }
    bool OpenCLBackend::OnSetCache()
    {
        std::set<std::string> buildOptions;
        buildOptions.emplace("-DBUFFER_IMAGE_IO_TRANS");
        mNHWCBufferToImageFloat = _mCLRuntime->BuildKernel("buffer_to_image", "nhwc_buffer_to_image", buildOptions);
        mImageToNHWCBufferFloat = _mCLRuntime->BuildKernel("buffer_to_image", "image_to_nhwc_buffer", buildOptions);
        // mNCHWBufferToImageFloat = _mCLRuntime->BuildKernel("buffer_to_image", "nchw_buffer_to_image", buildOptions);
        // mImageToNCHWBufferFloat = _mCLRuntime->BuildKernel("buffer_to_image", "image_to_nchw_buffer", buildOptions);
        return true;
    }
    void OpenCLBackend::_AllocHostBuffer(int length) const
    {
        SNN_ASSERT(length > 0);
        if (nullptr != mHostBuffer.second && length <= mHostBuffer.first)
            return;
        mHostBuffer.first = length;

        cl_context *GPUContext = _mCLRuntime->GetGPUContext();
        mHostBuffer.second = clCreateBuffer(*GPUContext, CL_MEM_READ_WRITE, length, NULL, NULL);
    }

    cl_mem OpenCLBackend::ConvertNHWCBufferToImage(const std::vector<int> &shape, DataFormat data_format, bool needwait, bool svmFlag)
    {
        cl_int err = 0;
        cl_context *GPUcontext = _mCLRuntime->GetGPUContext();
        cl_command_queue *commandQueue = _mCLRuntime->GetCommandQue();
        cl_image_format clImageFormat;
        clImageFormat.image_channel_order = CL_RGBA;
        clImageFormat.image_channel_data_type = CL_FLOAT;
        const std::vector<int> outputShape = TensorShapeFormat(shape, data_format);

        size_t imageWidth = outputShape[2] * UP_DIV(outputShape[3], 4), imageHeight = outputShape[0] * outputShape[1];
        cl_mem outputData = clCreateImage2D(*GPUcontext, CL_MEM_READ_WRITE, &clImageFormat, imageWidth, imageHeight, 0, NULL, &err);
        uint32_t outputGlobalWorkSize[2] = {static_cast<uint32_t>(UP_DIV(outputShape[3], 4) * outputShape[2]),
                                            static_cast<uint32_t>(outputShape[0] * outputShape[1])};
        if (mNHWCBufferToImageFloat == NULL)
        {
            std::set<std::string> buildOptions;
            buildOptions.emplace("-DBUFFER_IMAGE_IO_TRANS");
            mNHWCBufferToImageFloat = _mCLRuntime->BuildKernel("buffer_to_image", "nhwc_buffer_to_image", buildOptions);
        }
        uint32_t idx = 0;
        err |= clSetKernelArg(mNHWCBufferToImageFloat, idx++, sizeof(int), &outputGlobalWorkSize[0]);
        err |= clSetKernelArg(mNHWCBufferToImageFloat, idx++, sizeof(int), &outputGlobalWorkSize[1]);
        err |= clSetKernelArg(mNHWCBufferToImageFloat, idx++, sizeof(cl_mem), &mHostBuffer.second);
        err |= clSetKernelArg(mNHWCBufferToImageFloat, idx++, sizeof(int), &outputShape[1]);
        err |= clSetKernelArg(mNHWCBufferToImageFloat, idx++, sizeof(int), &outputShape[2]);
        err |= clSetKernelArg(mNHWCBufferToImageFloat, idx++, sizeof(int), &outputShape[3]);
        err |= clSetKernelArg(mNHWCBufferToImageFloat, idx++, sizeof(cl_mem), &outputData);
        oclCheckError(err, CL_SUCCESS);
        const uint32_t maxWorkGroupSize = static_cast<uint32_t>(_mCLRuntime->getMaxWorkGroupSize(mNHWCBufferToImageFloat));
        cl_event event;
        size_t roundUpGroupWorkSize[2] = {};
        size_t lws[2] = {16, MAX((size_t)1, maxWorkGroupSize / 16)};
        for (size_t i = 0; i < 2; ++i)
        {
            roundUpGroupWorkSize[i] = ROUND_UP(outputGlobalWorkSize[i], lws[i]);
        }
        err |= clEnqueueNDRangeKernel(commandQueue[0], mNHWCBufferToImageFloat, 2, NULL, roundUpGroupWorkSize, lws, 0, NULL, &event);
        oclCheckError(err, CL_SUCCESS);
        if (true == needwait)
        {
            clWaitForEvents(1, &event);
        }
        err |= clFinish(commandQueue[0]);
        oclCheckError(err, CL_SUCCESS);
        return outputData;
    }

    cl_mem OpenCLBackend::ConvertToDevice(const std::vector<int> &shape, DataFormat data_format, bool svmFlag)
    {
        cl_mem outputData;
        if (data_format == DATA_FORMAT_NHWC)
        {
            outputData = this->ConvertNHWCBufferToImage(shape, data_format, false, svmFlag);
        }
        else
        {
            printf("ERROR: data format not support !! \n");
            SNN_ASSERT(false);
        }
        return outputData;
    }

    void OpenCLBackend::CopyToDevice(Tensor *tensor)
    {
        cl_int err = 0;
        cl_command_queue *commandQueue = _mCLRuntime->GetCommandQue();
        const std::vector<std::vector<int>> &inputShapes = tensor->InputShape();
        SNN_ASSERT(inputShapes.size() == 1);
        const std::vector<int> &inputShape = inputShapes[0];
        // const std::vector<int> &outputShape = tensor->OutputShape();
        int buffer_sizes = inputShape[0] * inputShape[1] * inputShape[2] * inputShape[3] * sizeof(float);
        this->_AllocHostBuffer(buffer_sizes);
        // int elementSize = buffer_sizes / sizeof(float);
        // float pseudoIputData[elementSize] = {};
        // err |= clEnqueueWriteBuffer(commandQueue[0], mHostBuffer.second, CL_TRUE, 0, buffer_sizes, pseudoIputData, 0, NULL, NULL);
        // oclCheckError(err, CL_SUCCESS);
        DataFormat data_format = tensor->data_format;
        cl_mem outputData = this->ConvertToDevice(inputShapes[0], data_format, false);
        err |= clFinish(commandQueue[0]);
        oclCheckError(err, CL_SUCCESS);
        tensor->SetDeviceInputData(outputData);
        return;
    }
}