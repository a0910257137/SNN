#ifndef BUFFERCONVERTOR_H
#define BUFFERCONVERTOR_H
#include "backend/opencl/core/runtime/OpenCLRuntime.h"
#include "backend/opencl/core/OpenCLSetting.h"
#include "backend/opencl/utils/coreUtils.h"
namespace SNN
{
    enum TransType
    {
        InpTrans = 0,
        OutTrans = 1,
        NoTrans = 2
    };
    class BufferConvertor
    {
    public:
        explicit BufferConvertor(OpenCLRuntime *opencl_runtime);
        ~BufferConvertor();
        BufferConvertor(const BufferConvertor &) = delete;
        BufferConvertor &operator=(const BufferConvertor &) = delete;
        bool convertToNC4HW4Buffer(std::shared_ptr<Tensor> tensor, const OpenCLBufferFormat type,
                                   bool needTrans, bool needWait = false, bool lowMemory = false, int quantBit = 0);

    private:
        OpenCLRuntime *mOpenCLRuntime = nullptr;
        cl_kernel mImageToBufferKernel = NULL;
        cl_kernel mBufferToImageKernel = NULL;
        std::string mImageToBufferKernelName;
        std::string mBufferToImageKernelName;
    };

} // namespace SNN

#endif