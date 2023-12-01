#ifndef OPENCLBACKEND_H
#define OPENCLBACKEND_H
#include "include/SNN/Tensor.h"
#include "OpenCLSetting.h"
#include "runtime/OpenCLRuntime.h"
#include "backend/opencl/utils/coreUtils.h"
#include <iostream>
namespace SNN
{
    // future will add abstration class
    //  maintain the level of classes for buffer and image opencl execution

    class OpenCLBackend
    {
    public:
        OpenCLBackend(bool permitFloat16);
        ~OpenCLBackend();
        OpenCLBackend(const OpenCLBackend &) = delete;
        OpenCLBackend &operator=(const OpenCLBackend &) = delete;
        OpenCLRuntime *CLRuntime() const { return _mCLRuntime; };
        cl_mem ConvertToDevice(const std::vector<int> &shape, DataFormat data_format, bool svmFlag);
        cl_mem ConvertNHWCBufferToImage(const std::vector<int> &shape, DataFormat data_format, bool needwait, bool svmFlag);
        void CopyToDevice(Tensor *srcTensor);
        mutable std::pair<int, cl_mem> mHostBuffer;

    private:
        bool OnSetCache();
        void _AllocHostBuffer(int length) const;
        // cl_mem ConvertToDevice(const Tensor *srcTensor, DataFormat data_format, bool svmFlag);
        // cl_mem ConvertNHWCBufferToImage(const Tensor *tensor, bool needwait, bool svmFlag);

    protected:
        OpenCLRuntime *_mCLRuntime = nullptr;

    private:
        bool permitFloat16;

    private:
        // cl_kernel mImageToNCHWBufferFloat;
        // cl_kernel mImageToNC4HW4BufferFloat;
        // cl_kernel mNC4HW4BufferToImageFloat;
        // cl_kernel mNCHWBufferToImageFloat;
        // cl_kernel mNHWCBufferToImageInt8;
        cl_kernel mImageToNHWCBufferFloat;
        cl_kernel mNHWCBufferToImageFloat;
    };
}
#endif // OPENCLBACKEND_H