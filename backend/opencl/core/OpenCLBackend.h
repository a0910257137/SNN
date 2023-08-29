#ifndef OPENCLBACKEND_H
#define OPENCLBACKEND_H

#include <string>
#include <memory>
#include "CL/opencl.h"
#include "CL/cl_half.h"
#include "OpenCLBackend.h"
#include "OpenCLSetting.h"
#include "runtime/OpenCLRuntime.h"
#include "include/SNN/api_types.h"
#include "include/SNN/common.h"
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

    protected:
        OpenCLRuntime *_mCLRuntime = nullptr;

    private:
        bool permitFloat16;
    };
}
#endif // OPENCLBACKEND_H