#ifndef OPENCLRUNTIME_H
#define OPENCLRUNTIME_H
#include "OpenCLBackend.h"
#include "OpenCLSetting.h"
#include <string>
#include "runtime/OpenCLRuntime.h"
#include <CL/opencl.h>
#include <memory>
namespace SNN
{
    class OpenCLBackend
    {
    public:
        OpenCLBackend(bool enable_half);
        ~OpenCLBackend();
        OpenCLBackend(const OpenCLBackend &) = delete;
        OpenCLBackend &operator=(const OpenCLBackend &) = delete;

    private:
        const OpenCLRuntime *mCLRuntime;
        bool enable_half;
    };
}
#endif // OPENCLRUNTIME_H__