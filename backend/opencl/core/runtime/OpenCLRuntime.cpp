#include "OpenCLRuntime.h"

namespace SNN
{
    OpenCLRuntime::OpenCLRuntime()
    {

        printf("Get the devices and create context\n");
        this->err = oclGetPlatformID(&(this->platform));
    }
    OpenCLRuntime::~OpenCLRuntime()
    {
    }
}