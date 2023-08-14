#include "OpenCLRuntime.h"

namespace SNN
{
    OpenCLRuntime::OpenCLRuntime()
    {

        printf("INFO: Get the devices and create context\n");
        this->err = oclGetPlatformID(&(this->platform));
        oclCheckError((this->err), CL_SUCCESS);
        this->err = clGetDeviceIDs(this->platform, CL_DEVICE_TYPE_GPU, 0, NULL, &(this->NumDevices));
        printf("INFO: Fetch the # of %d device\n", this->NumDevices);
        this->device = (cl_device_id *)malloc(this->NumDevices * sizeof(cl_device_id));
        oclCheckError(err, CL_SUCCESS);
        this->err = clGetDeviceIDs(this->platform, CL_DEVICE_TYPE_GPU, this->NumDevices, this->device, NULL);
        oclCheckError(this->err, CL_SUCCESS);
        cxGPUContext = clCreateContext(0, this->NumDevices, this->device, NULL, NULL, &(this->err));
        oclCheckError(this->err, CL_SUCCESS);
        size_t DeviceBytes;
        cl_uint DeviceCount;
        this->err |= clGetContextInfo(this->cxGPUContext, CL_CONTEXT_DEVICES, 0, nullptr, &DeviceBytes);
        DeviceCount = (cl_uint)DeviceBytes / sizeof(cl_device_id);
        if (err != CL_SUCCESS)
            printf("ERROR: %i in clGetDeviceIDs call !!!\n\n", this->err);
        else if (DeviceCount == 0)
            printf("ERROR: There are no devices supporting OpenCL (return code %i)\n\n", this->err);

        for (int i = 0; i < DeviceCount; ++i)
        {
            this->commandQueue[i] = clCreateCommandQueue(this->cxGPUContext, this->device[i], CL_QUEUE_PROFILING_ENABLE, &this->err);
        }
        if ((this->err) != CL_SUCCESS)
            printf("ERROR:  %i in clCreateCommandQueue call !!!\n\n", (this->err));
        printf("INFO: Allocating device variables\n");
    }
    OpenCLRuntime::~OpenCLRuntime()
    {
    }
}