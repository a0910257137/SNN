#include "OpenCLRuntime.h"

namespace SNN
{
    OpenCLRuntime::OpenCLRuntime()
    {
        printf("INFO: ===================Initialization of OpenCL device===================\n");
        printf("INFO: Get the devices and create context\n");
        this->err = oclGetPlatformID(&(this->platform));
        oclCheckError((this->err), CL_SUCCESS);
        this->err = clGetDeviceIDs(this->platform, CL_DEVICE_TYPE_GPU, 0, NULL, &(this->_num_devices));
        printf("INFO: Fetch the # of %d device\n", this->_num_devices);
        this->_device = (cl_device_id *)malloc(this->_num_devices * sizeof(cl_device_id));
        oclCheckError(err, CL_SUCCESS);
        this->err = clGetDeviceIDs(this->platform, CL_DEVICE_TYPE_GPU, this->_num_devices, this->_device, NULL);
        oclCheckError(this->err, CL_SUCCESS);
        _GPUContext = clCreateContext(0, this->_num_devices, this->_device, NULL, NULL, &(this->err));
        oclCheckError(this->err, CL_SUCCESS);
        size_t DeviceBytes;
        cl_uint DeviceCount;
        this->err |= clGetContextInfo(this->_GPUContext, CL_CONTEXT_DEVICES, 0, nullptr, &DeviceBytes);
        DeviceCount = (cl_uint)DeviceBytes / sizeof(cl_device_id);
        if (err != CL_SUCCESS)
            printf("ERROR: %i in clGetDeviceIDs call !!!\n\n", this->err);
        else if (DeviceCount == 0)
            printf("ERROR: There are no devices supporting OpenCL (return code %i)\n\n", this->err);

        for (int i = 0; i < DeviceCount; ++i)
        {
            this->_commandQueue[i] = clCreateCommandQueue(this->_GPUContext, this->_device[i], CL_QUEUE_PROFILING_ENABLE, &this->err);
        }
        if ((this->err) != CL_SUCCESS)
            printf("ERROR:  %i in clCreateCommandQueue call !!!\n\n", (this->err));
        bool status = OpenCLRuntime::BuildProgramMaps();
        oclCheckError(status, true);
        cl_device_fp_config fp_config;
        clGetDeviceInfo(this->_device[0], CL_DEVICE_SINGLE_FP_CONFIG, sizeof(cl_device_fp_config), &fp_config, 0);
        mIsDeviceSupportedFP16 = fp_config > 0;
        if (mIsDeviceSupportedFP16)
            printf("INFO: Device is supported half-precision\n");
        printf("INFO: ================Finish initialization of OpenCL device================\n");
        mGpuType = OTHER;
    }
    OpenCLRuntime::~OpenCLRuntime()
    {
        mProgramMaps.clear();
        mProgramDirty.clear();
    }
    bool OpenCLRuntime::BuildProgramMaps()
    {
        printf("INFO: Fetch opencl programs from %s\n", "backend/opencl/execution/cl/*.cl");
        DIR *dir;
        struct dirent *ent;
        int count = 0;
        std::string b = ".", bb = "..";
        size_t program_length;
        char *source_path, *source;
        if ((dir = opendir("../backend/opencl/execution/cl/")) != NULL)
        {
            while ((ent = readdir(dir)) != NULL)
            {
                std::string file_name = ent->d_name;
                if (file_name == b || file_name == bb)
                    continue;
                source_path = shrFindFilePath(ent->d_name);
                // std::cout << source_path << std::endl;
                oclCheckError(source_path != NULL, shrTRUE);
                source = oclLoadProgSource(source_path, "", &program_length);
                auto programRaw = clCreateProgramWithSource(this->_GPUContext, 1, (const char **)&source, &program_length, &err);
                oclCheckError(err, CL_SUCCESS);
                if (!programRaw)
                {
                    printf("Can't load %s  load program\n", source_path);
                    return false;
                }
                mProgramMaps.insert(std::pair<std::string, cl_program>(file_name, programRaw));
                mProgramDirty.insert(std::pair<std::string, bool>(file_name, false));
            }
            closedir(dir);
        }
        else
        {
            perror("");
        }
        return true;
    }
    bool OpenCLRuntime::LoadProgram(const std::string &cl_name, cl_program &program)
    {
        auto buildProgramInter = mProgramMaps.find(cl_name);
        auto isProgramDirty = mProgramDirty.find(cl_name);

        if (buildProgramInter != mProgramMaps.end())
        {
            program = buildProgramInter->second;
            isProgramDirty->second = true;
            return true;
        }
        else
        {
            printf("ERROR: Can't load program %s source \n", cl_name.c_str());
            return false;
        }
    }
    cl_kernel OpenCLRuntime::BuildKernel(const std::string &programName, const std::string &kernelName, const std::set<std::string> &buildOptions)
    {
        std::string buildOptionsStr;
        if (mIsSupportedFP16)
            buildOptionsStr = "-DFLOAT=half -DFLOAT2=half2 -DFLOAT4=half4 -DFLOAT8=half8 -DFLOAT16=half16 -DRI_F=read_imageh -DWI_F=write_imageh -DCONVERT_FLOAT4=convert_half4 -DMNN_SUPPORT_FP16";
        else
            buildOptionsStr = "-DFLOAT=float  -DFLOAT2=float2 -DFLOAT4=float4 -DFLOAT8=float8 -DRI_F=read_imagef -DFLOAT16=float16 -DWI_F=write_imagef -DCONVERT_FLOAT4=convert_float4";
        if (isSetWorkGroupAttribute)
            buildOptionsStr += " -DSET_ATTRIBUTE=true";
        else
            buildOptionsStr += " -DSET_ATTRIBUTE=false";
        for (auto &option : buildOptions)
            buildOptionsStr += " " + option;

        buildOptionsStr += mDefaultBuildParams;
        std::string key = programName + ".cl";
        cl_program program;
        bool isDirty = this->mProgramDirty.find(key)->second;
        bool status = LoadProgram(key, program);
        oclCheckError(status, true);
        if (isDirty == false)
        {
            err = clBuildProgram(program, this->_num_devices, this->_device, buildOptionsStr.c_str(), NULL, NULL);
            oclCheckError(err, CL_SUCCESS);
        }
        cl_kernel kernel = clCreateKernel(program, kernelName.c_str(), &err);
        oclCheckError(err, CL_SUCCESS);
        return kernel;
    }

    uint64_t OpenCLRuntime::getMaxWorkGroupSize(const cl_kernel &kernel)
    {
        uint64_t maxWorkGroupSize = 0;
        clGetKernelWorkGroupInfo(kernel, *this->_device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(uint64_t), &maxWorkGroupSize, NULL);
        return maxWorkGroupSize;
    }
    size_t *OpenCLRuntime::getMaxWorkItemSizes()
    {
        cl_int err;
        cl_uint workItemDims;
        err = clGetDeviceInfo(_device[0], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(workItemDims), &workItemDims, NULL);
        size_t *p_max_work_item_sizes = NULL;
        if (workItemDims < 3)
        {
            p_max_work_item_sizes[0] = 8;
            p_max_work_item_sizes[1] = 8;
            p_max_work_item_sizes[2] = 8;
            return p_max_work_item_sizes;
        }
        size_t size;
        err = clGetDeviceInfo(_device[0], CL_DEVICE_MAX_WORK_ITEM_SIZES, 0, NULL, &size);
        oclCheckError(err, CL_SUCCESS);
        p_max_work_item_sizes = (size_t *)malloc(size);
        err = clGetDeviceInfo(_device[0], CL_DEVICE_MAX_WORK_ITEM_SIZES, size, p_max_work_item_sizes, NULL);
        // for (size_t i = 0; i < size / sizeof(size_t); i++)
        // {
        //     printf("max_work_item_size_of_work_group_dim %zu=%zu\n", i, p_max_work_item_sizes[i]);
        // }
        oclCheckError(err, CL_SUCCESS);
        return p_max_work_item_sizes;
    }
    std::map<std::pair<std::string, std::vector<size_t>>, std::pair<std::vector<size_t>, float>> &OpenCLRuntime::TunedLwsMap()
    {

        return mTunedLws;
    }
    float OpenCLRuntime::GetCostTime(const cl_event *event)
    {

        cl_int err = clWaitForEvents(1, event);
        oclCheckError(err, CL_SUCCESS);
        cl_ulong time_start, time_end;
        float total_time;
        clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
        total_time = (time_end - time_start) / 1000000.0f;
        // printf("\n Execution time in milliseconds = %0.3f ms\n", (total_time / 1000000.0f));
        return total_time;
    }
    std::pair<std::vector<size_t>, float_t> OpenCLRuntime::localWS2DDefault(const std::vector<size_t> &gws,
                                                                            const size_t maxWorkGroupSize,
                                                                            OpenCLRuntime *runtime,
                                                                            const std::string &kernelName,
                                                                            const cl_kernel &mKernel)
    {
        float min_cost = INFINITY;
        auto maxWorkItemSizes = runtime->getMaxWorkItemSizes();
        cl_command_queue *commandQueue = runtime->GetCommandQue();
        size_t lws[] = {1, 1};
        std::vector<size_t> lws_prefer(2, 1);
        auto &tunedLws = runtime->TunedLwsMap();
        std::pair<std::string, std::vector<size_t>> info = std::make_pair(kernelName, gws);
        if (tunedLws.find(info) != tunedLws.end())
        {
            return tunedLws[info];
        }
        // for (size_t i = 0; i < 3; i++)
        // {
        //     printf("max_work_item_size_of_work_group_dim %zu=%zu\n", i, maxWorkItemSizes[i]);
        // }
        err = 0;

        if (runtime->GetCLTuneLevel() == Fast)
        {
            while (lws[1] <= gws[1] && lws[1] <= 6)
            {
                lws[0] = 1;
                while (lws[0] <= gws[0] && lws[0] <= 6)
                {
                    if (lws[0] <= maxWorkItemSizes[0] && lws[1] <= maxWorkItemSizes[1] && lws[0] * lws[1] <= maxWorkGroupSize)
                    {

                        cl_event event;
                        size_t internalGlobalWS[2] = {1, 1};
                        for (size_t i = 0; i < 2; ++i)
                        {
                            internalGlobalWS[i] = ROUND_UP(gws[i], MAX((size_t)1, lws[i]));
                        }
                        // const size_t lws[2] = {16, MAX((unsigned int)1, maxWorkGroupSize / 16)};
                        err |= clEnqueueNDRangeKernel(commandQueue[0], mKernel, 2, NULL, internalGlobalWS, lws, 0, NULL, &event);
                        oclCheckError(err, CL_SUCCESS);

                        if (err != CL_SUCCESS)
                        {
                            printf("lws tune result errors %s", kernelName.c_str());
                        }
                        float cost_time = (float)this->GetCostTime(&event);
                        // std::cout << cost_time << std::endl;
                        if (cost_time < min_cost)
                        {
                            min_cost = cost_time;
                            lws_prefer[0] = lws[0];
                            lws_prefer[1] = lws[1];
                        }
                    }
                    do
                    {
                        lws[0]++;
                    } while (((2 * gws[0]) % lws[0] > 1) && (lws[0] & (lws[0] - 1)) != 0 && (lws[0] <= gws[0]) && (lws[0] <= 6));
                }
                // float *a = (float *)malloc(320 * 320 * 24 * sizeof(float));
                do
                {
                    lws[1]++;
                } while (((2 * gws[1]) % lws[1] > 1) && (lws[1] & (lws[1] - 1)) != 0 && (lws[1] <= gws[1]) && (lws[1] <= 6)); // divisible powOfTwo lessThanSix
            }
        }
        else if (runtime->GetCLTuneLevel() == None)
        {
            // define not tune method to choose lws
            lws_prefer[0] = 0;
            lws_prefer[1] = 0;
            min_cost = 0.0f;
        }

        if (runtime->GetCLTuneLevel() != None)
        {
            cl_event event;
            size_t gw_arr[2] = {gws[0], gws[1]};
            err |= clEnqueueNDRangeKernel(commandQueue[0], mKernel, 2, NULL, gw_arr, NULL, 0, NULL, &event);
            oclCheckError(err, CL_SUCCESS);

            float cost_time = runtime->GetCostTime(&event);
            if (cost_time < min_cost)
            {
                lws_prefer[0] = 0;
                lws_prefer[1] = 0;
                min_cost = cost_time;
            }
        }
        if (tunedLws.find(info) == tunedLws.end())
        {
            tunedLws.insert(std::make_pair(info, std::make_pair(lws_prefer, min_cost)));
        }
        // printf("INFO: Find Best local work item for x %lu \n", lws[0]);
        // printf("INFO: Find Best local work item for y %lu \n", lws[1]);
        return std::make_pair(lws_prefer, min_cost);
    }
    uint64_t OpenCLRuntime::maxAllocSize() const
    {
        return mMaxMemAllocSize;
    }
    bool OpenCLRuntime::isSupportedFP16() const
    {
        return mIsSupportedFP16;
    }
    bool OpenCLRuntime::isWeightCpuTransHalf() const
    {
#ifdef USE_HALF_WEIGHT_MEMORY
        return mIsSupportedFP16;
#else
        return false; // most of time
#endif
    }
    bool OpenCLRuntime::isDeviceSupportedFP16() const
    {
        return mIsDeviceSupportedFP16;
    }
}