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
        bool ConvertBufferToImage(std::shared_ptr<Tensor> tensor, const OpenCLBufferFormat type, bool needwait = false, const std::string &buildOption = "");

        bool ConvertImageToNHWCBuffer(std::shared_ptr<Tensor> tensor, cl_kernel &imageToBufferKernel,
                                      OpenCLRuntime *runtime, bool needWait = false, bool svmFlag = false);

    private:
        OpenCLRuntime *mOpenCLRuntime = nullptr;
        cl_kernel mImageToBufferKernel;
        std::string mImageToBufferKernelName;
        cl_kernel mBufferToImageKernel;
        std::string mBufferToImageKernelName;
    };
}
#endif // IMAGEBUFFERCONVERTER_H