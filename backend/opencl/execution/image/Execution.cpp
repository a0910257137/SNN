#include "Execution.h"

namespace SNN
{
    Execution::Execution(OpenCLBackend *mbackend)
    {
        mOpenCLRuntime = mbackend->CLRuntime();
        GPUcontext = mOpenCLRuntime->GetGPUContext();
        commandQueue = mOpenCLRuntime->GetCommandQue();
        clImageFormat.image_channel_order = CL_RGBA;
        clImageFormat.image_channel_data_type = CL_FLOAT;
        clImageFormat.image_channel_order = CL_RGBA;
        mImageConvert = new ImageBufferConverter(mOpenCLRuntime);
    }
    bool Execution::onResize(Tensor *inputs, int *input_shape, int *output_shape)
    {
        return true;
    }
    bool Execution::onExecute(std::vector<std::shared_ptr<Tensor>> &inputs, std::vector<std::shared_ptr<Tensor>> &outputs)
    {
        return true;
    }
    float *Execution::onConvert(std::shared_ptr<Tensor> &input)
    {
        std::set<std::string> buildOptions;
        buildOptions.emplace("-DBUFFER_IMAGE_IO_TRANS");
        cl_kernel imageToBufferKernel = mOpenCLRuntime->BuildKernel("buffer_to_image", "image_to_nhwc_buffer", buildOptions);
        float *outputData = mImageConvert->ConvertImageToNHWCBuffer(input, imageToBufferKernel, mOpenCLRuntime, false, false);
        // clRetainMemObject();
        return outputData;
    }
    Execution::~Execution()
    {
        delete mImageConvert;
    }

} // SNN