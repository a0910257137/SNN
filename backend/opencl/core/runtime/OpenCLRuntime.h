#ifndef OpenCLRuntime_H
#define OpenCLRuntime_H
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

    private:
        cl_platform_id platform;
        cl_device_id *device;
        cl_context cxGPUContext;
        cl_uint NumDevices;
        cl_command_queue commandQueue[NQUEUES];
        cl_event event = NULL;
        cl_program program;
    };
}
#endif // OpenCLRuntime_H__