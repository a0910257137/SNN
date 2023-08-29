#ifndef OPENCLRUNTIME_H
#define OPENCLRUNTIME_H
#include <string>
#include <stdio.h>
#include <CL/opencl.h>
#include <map>
#include <set>
#include <dirent.h>
#include "backend/opencl/utils/shrUtils.h"
#include "backend/opencl/utils/oclUtils.h"
#include <iostream>
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
        cl_kernel BuildKernel(const std::string &programName, const std::string &kernelName, const std::set<std::string> &buildOptions);
        cl_device_id *GetDevice() { return _device; };
        cl_uint GetNumDevice() { return _num_devices; };
        cl_command_queue *GetCommandQue() { return _commandQueue; };
        cl_context &GetGPUContext() { return _GPUContext; };
        uint64_t maxAllocSize() const;
        uint64_t getMaxWorkGroupSize(const cl_kernel &kernel);
        bool isSupportedFP16() const;
        bool isWeightCpuTransHalf() const;
        bool isDeviceSupportedFP16() const;
        cl_int err;

    protected:
        cl_device_id *_device;
        cl_uint _num_devices;
        cl_context _GPUContext;
        cl_command_queue _commandQueue[NQUEUES];

    private:
        bool LoadProgram(const std::string &cl_name, cl_program &program);
        bool BuildProgramMaps();

    private:
        bool isSetWorkGroupAttribute = false;
        std::string mDefaultBuildParams = " -cl-mad-enable";
        std::map<std::string, cl_program> mProgramMaps;
        cl_platform_id platform;
        cl_event event = NULL;
        uint32_t mMaxMemAllocSize;
        bool mIsSupportedFP16 = false;
        bool mIsDeviceSupportedFP16 = false;
        float mCLVersion = 3.0f;
    };
}
#endif // OpenCLRuntime_H__