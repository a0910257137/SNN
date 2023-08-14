#ifndef OPENCLRUNTIME_H
#define OPENCLRUNTIME_H
#include <string>
#include <stdio.h>
#include <CL/opencl.h>
#include "../../utils/shrUtils.h"
#include "../../utils/oclUtils.h"
#define NQUEUES 1
namespace SNN
{
    class OpenCLRuntime
    {
    public:
        OpenCLRuntime();
        ~OpenCLRuntime();
        OpenCLRuntime(const OpenCLRuntime &) = delete;
        OpenCLRuntime &operator=(const OpenCLRuntime &) = delete;
        cl_int err;
        cl_context cxGPUContext;
        cl_program program;
        cl_command_queue commandQueue[NQUEUES];

    private:
        cl_platform_id platform;
        cl_device_id *device;
        cl_uint NumDevices;
        cl_event event = NULL;
    };
}
#endif // OpenCLRuntime_H__