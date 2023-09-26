#ifndef IMAGEBUFFERCONVERTER_H
#define IMAGEBUFFERCONVERTER_H
#include <set>
#include <iostream>
#include "include/SNN/Tensor.h"
#include "backend/opencl/core/runtime/OpenCLRuntime.h"
#include "backend/opencl/core/OpenCLSetting.h"
#include "backend/opencl/utils/coreUtils.h"
#include "include/SNN/macro.h"
namespace SNN
{
    class ImageBufferConverter
    {
    public:
        explicit ImageBufferConverter(OpenCLRuntime *opencl_runtime);
        ~ImageBufferConverter();
        ImageBufferConverter(const ImageBufferConverter &) = delete;
        ImageBufferConverter &operator=(const ImageBufferConverter &) = delete;
        bool ConvertImagetoBuffer(std::shared_ptr<Tensor> tensor, const OpenCLBufferFormat type, bool needwait = false, const std::string &buildOption = "");
        bool ConvertBuffertoImage(Tensor *inputs);

    private:
        OpenCLRuntime *mOpenCLRuntime = nullptr;
        cl_kernel mImageToBufferKernel;
        std::string mImageToBufferKernelName;
        cl_kernel mBufferToImageKernel;
        std::string mBufferToImageKernelName;
    };
}
#endif // IMAGEBUFFERCONVERTER_H