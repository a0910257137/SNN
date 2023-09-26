#ifndef OPENCLRUNTIME_H
#define OPENCLRUNTIME_H
#include <string>
#include <stdio.h>
#include <map>
#include <set>
#include <dirent.h>
#include "include/SNN/Tensor.h"
#include "backend/opencl/utils/shrUtils.h"
#include "backend/opencl/utils/oclUtils.h"
#include "include/SNN/macro.h"
#include <iostream>
#include <limits.h>
#include <utility>
#include <vector>

#define NQUEUES 1

namespace SNN
{
    enum CLTuneLevel
    {
        None = 0,
        Heavy = 1,
        Wide = 2,
        Normal = 3,
        Fast = 4
    };
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
        std::pair<std::vector<size_t>, size_t> localWS2DDefault(const std::vector<size_t> &gws,
                                                                const size_t maxWorkGroupSize,
                                                                OpenCLRuntime *runtime,
                                                                const std::string &kernelName,
                                                                const cl_kernel &mKernel);
        CLTuneLevel GetCLTuneLevel()
        {
            return mTuneLevel;
        }

        float GetCostTime(const cl_event *event);
        size_t *getMaxWorkItemSizes();
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
        std::map<std::string, bool> mProgramDirty;
        cl_platform_id platform;
        cl_event event = NULL;
        uint32_t mMaxMemAllocSize;
        bool mIsSupportedFP16 = false;
        bool mIsDeviceSupportedFP16 = false;
        float mCLVersion = 3.0f;
        CLTuneLevel mTuneLevel = Fast;
    };
}
#endif // OpenCLRuntime_H__