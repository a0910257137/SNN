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
        // std::string pathDir = "../backend/opencl/execution/cl/";
        // std::string pathDir = "../backend/opencl/execution/bin/";
        std::string pathDir;
        bool status;
        if (isBinarySource)
        {
            pathDir = "../backend/opencl/execution/bin/";
            status = BuildBinaryProgramMaps(pathDir);
        }
        else
        {
            pathDir = "../backend/opencl/execution/cl/";
            status = BuildProgramMaps(pathDir);
        }
        // bool status = BuildBinaryProgramMaps(pathDir);
        oclCheckError(status, true);
        cl_device_fp_config fp_config;
        clGetDeviceInfo(this->_device[0], CL_DEVICE_SINGLE_FP_CONFIG, sizeof(cl_device_fp_config), &fp_config, 0);
        mIsDeviceSupportedFP16 = fp_config > 0;
        if (mIsDeviceSupportedFP16)
            printf("INFO: Device is supported half-precision\n");
        printf("INFO: ================Finish initialization of OpenCL device================\n");
        mGpuType = OTHER;
        // get gpu information
        clGetDeviceInfo(this->_device[0], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(uint64_t), &mGPUGlobalMemeryCacheSize, 0);
        clGetDeviceInfo(this->_device[0], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(uint32_t), &mGPUComputeUnits, 0);
        clGetDeviceInfo(this->_device[0], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(uint32_t), &mMaxFreq, 0);
        cl_device_svm_capabilities mSvmCapabilities;
        clGetDeviceInfo(this->_device[0], CL_DEVICE_SVM_CAPABILITIES, sizeof(mSvmCapabilities), &mSvmCapabilities, 0);
    }
    OpenCLRuntime::~OpenCLRuntime()
    {
        mSourceMaps.clear();
        mBuiltProgramMaps.clear();
    }

    bool OpenCLRuntime::BuildBinaryProgramMaps(std::string &folder)
    {
        printf("INFO: Fetch pre-built binary opencl programs from %s\n", "backend/opencl/execution/bin/*.bin");
        DIR *dir;
        FILE *ptr;
        struct dirent *ent;
        std::string b = ".", bb = "..", saved_name;
        char *source_path;
        char *binary;
        unsigned long length;
        cl_int binary_status;
        cl_program program;
        if ((dir = opendir(folder.c_str())) != NULL)
        {
            while ((ent = readdir(dir)) != NULL)
            {
                std::string file_name = ent->d_name;
                if (file_name == b || file_name == bb)
                    continue;
                char *filae_path = (char *)malloc(strlen(folder.c_str()) + strlen(file_name.c_str()) + 1);
                strcpy(filae_path, folder.c_str());
                strcat(filae_path, file_name.c_str());
                binary = common_read_file(filae_path, &length);
                saved_name = remove_extension(file_name);
                // program = clCreateProgramWithBinary(_GPUContext, 1, _device, &length,
                //                                     (const unsigned char **)&binary, &binary_status, &err);
                // std::cout << err << std::endl;
                oclCheckError(err, CL_SUCCESS);
                mSourceMaps[saved_name] = std::make_tuple(binary, length);
                free(filae_path);
            }
            closedir(dir);
        }
        else
        {
            perror("");
        }
        // exit(1);
        return true;
    }
    bool OpenCLRuntime::BuildProgramMaps(std::string &folder)
    {
        printf("INFO: Fetch opencl programs from %s\n", "backend/opencl/execution/cl/*.cl");
        DIR *dir;
        struct dirent *ent;
        std::string b = ".", bb = "..", saved_name;
        size_t program_length;
        char *source_path, *source;
        if ((dir = opendir(folder.c_str())) != NULL)
        {
            while ((ent = readdir(dir)) != NULL)
            {
                std::string file_name = ent->d_name;
                if (file_name == b || file_name == bb)
                    continue;
                source_path = shrFindFilePath(ent->d_name);
                oclCheckError(source_path != NULL, shrTRUE);
                source = oclLoadProgSource(source_path, "", &program_length);
                saved_name = remove_extension(file_name);
                mSourceMaps[saved_name] = std::make_tuple(source, static_cast<unsigned long>(program_length));
            }
            closedir(dir);
        }
        else
        {
            perror("");
        }
        return true;
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
        cl_program program;
        std::tuple<std::string, std::string> key;
        key = std::make_pair(programName + "_" + kernelName, buildOptionsStr);
        auto builtProgram = mBuiltProgramMaps.find(key);
        cl_kernel kernel = NULL;
        err = CL_SUCCESS;
        char *source;
        if (builtProgram != mBuiltProgramMaps.end())
        {
            program = builtProgram->second;
        }
        else
        {
            std::tuple<char *, unsigned long> source_infos = mSourceMaps[programName];
            source = std::get<0>(source_infos);
            unsigned long length = std::get<1>(source_infos);
            if (isBinarySource)
            {
                cl_int binary_status;
                // std::cout << programName << std::endl;
                // std::cout << length << std::endl;
                program = clCreateProgramWithBinary(_GPUContext, 1, _device, &length,
                                                    (const unsigned char **)&source, &binary_status, &err);
                oclCheckError(err, CL_SUCCESS);
            }
            else
                program = clCreateProgramWithSource(this->_GPUContext, 1, (const char **)&source, &length, &err);
            oclCheckError(err, CL_SUCCESS);
            if (!program)
            {
                printf("Can't load %s  load program\n", programName.c_str());
                exit(1);
            }
            err = clBuildProgram(program, this->_num_devices, this->_device, buildOptionsStr.c_str(), NULL, NULL);
            oclCheckError(err, CL_SUCCESS);
            // exit(1);
            mBuiltProgramMaps.emplace(std::make_pair(key, program));
        }
        kernel = clCreateKernel(program, kernelName.c_str(), &err);
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
        // if (err != 0)
        //     return INFINITY;
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
        // std::cout << kernelName << std::endl;
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
        // exit(1);
        err = 0;
        if (runtime->GetCLTuneLevel() == Fast)
        {
            while (lws[1] <= gws[1] && lws[1] <= 6)
            {
                lws[0] = 1;
                while (lws[0] <= gws[0] && lws[0] <= 6)
                {
                    if ((lws[0] <= maxWorkItemSizes[0]) && (lws[1] <= maxWorkItemSizes[1]) && (lws[0] * lws[1] <= maxWorkGroupSize))
                    {
                        cl_event event = NULL;
                        size_t internalGlobalWS[2] = {1, 1};
                        for (int i = 0; i < 2; ++i)
                        {
                            internalGlobalWS[i] = ROUND_UP(gws[i], MAX((int)1, lws[i]));
                        }
                        err |= clEnqueueNDRangeKernel(commandQueue[0], mKernel, 2, NULL, internalGlobalWS, lws, 0, nullptr, &event);
                        oclCheckError(err, CL_SUCCESS);
                        float cost_time = (float)this->GetCostTime(&event);
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
                    } while (((2 * gws[0]) % lws[0] > 1) && (lws[0] & (lws[0] - 1)) != 0 && (lws[0] <= gws[0]) && (lws[0] <= 6)); // divisible powOfTwo lessThanSix
                }
                do
                {
                    lws[1]++;
                } while (((2 * gws[1]) % lws[1] > 1) && (lws[1] & (lws[1] - 1)) != 0 && (lws[1] <= gws[1]) && (lws[1] <= 6)); // divisible powOfTwo lessThanSix
            }
            err |= clFinish(commandQueue[0]);
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

            if (err != CL_SUCCESS)
            {
                printf("2D lws null res %s\n", kernelName.c_str());
            }

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
        // printf("INFO: Find Best local work item for x %lu \n", lws_prefer[0]);
        // printf("INFO: Find Best local work item for y %lu \n", lws_prefer[1]);
        return std::make_pair(lws_prefer, min_cost);
    }
    std::pair<std::vector<size_t>, float_t> OpenCLRuntime::localWS3DDefault(const std::vector<size_t> &gws,
                                                                            const size_t maxWorkGroupSize,
                                                                            OpenCLRuntime *runtime,
                                                                            const std::string &kernelName,
                                                                            const cl_kernel &mKernel)
    {
        SNN_ASSERT(gws.size() == 3);

        auto maxWorkItemSizes = runtime->getMaxWorkItemSizes();
        auto &tunedLws = runtime->TunedLwsMap();
        std::pair<std::string, std::vector<size_t>> info = std::make_pair(kernelName, gws);
        if (tunedLws.find(info) != tunedLws.end())
        {
            return tunedLws[info];
        }
        cl_command_queue *commandQueue = runtime->GetCommandQue();

        size_t lws[] = {1, 1, 1};
        std::vector<size_t> lws_prefer(4, 1);
        float min_cost = INFINITY;
        if (runtime->GetCLTuneLevel() == Fast)
        {
            while (lws[2] <= gws[2] && lws[2] <= 6)
            {
                lws[1] = 1;
                while (lws[1] <= gws[1] && lws[1] <= 6)
                {
                    lws[0] = 1;
                    while (lws[0] <= gws[0] && lws[0] <= 6)
                    {
                        if ((lws[0] <= maxWorkItemSizes[0]) && (lws[1] <= maxWorkItemSizes[1]) && (lws[2] <= maxWorkItemSizes[2]) && (lws[0] * lws[1] * lws[2] <= maxWorkGroupSize))
                        {
                            cl_event event;
                            size_t internalGlobalWS[3] = {1, 1, 1};
                            for (size_t i = 0; i < 3; ++i)
                            {
                                internalGlobalWS[i] = ROUND_UP(gws[i], MAX((size_t)1, lws[i]));
                            }
                            // const size_t lws[2] = {16, MAX((unsigned int)1, maxWorkGroupSize / 16)};
                            err |= clEnqueueNDRangeKernel(commandQueue[0], mKernel, 3, NULL, internalGlobalWS, lws, 0, NULL, &event);
                            oclCheckError(err, CL_SUCCESS);

                            if (err != CL_SUCCESS)
                            {
                                printf("lws tune res %s\n", kernelName.c_str());
                            }
                            float cost_time = (float)this->GetCostTime(&event);
                            // std::cout << cost_time << std::endl;
                            if (cost_time < min_cost)
                            {
                                min_cost = cost_time;
                                lws_prefer[0] = lws[0];
                                lws_prefer[1] = lws[1];
                                lws_prefer[2] = lws[2];
                            }
                        }
                        do
                        {
                            lws[0]++;
                        } while (((2 * gws[0]) % lws[0] > 1) && (lws[0] & (lws[0] - 1)) != 0 && (lws[0] <= gws[0]) && (lws[0] <= 6)); // divisible powOfTwo lessThanSix
                    }
                    do
                    {
                        lws[1]++;
                    } while (((2 * gws[1]) % lws[1] > 1) && (lws[1] & (lws[1] - 1)) != 0 && (lws[1] <= gws[1]) && (lws[1] <= 6)); // divisible powOfTwo lessThanSix
                }
                do
                {
                    lws[2]++;
                } while (((2 * gws[2]) % lws[2] > 1) && (lws[2] & (lws[2] - 1)) != 0 && (lws[2] <= gws[2]) && (lws[2] <= 6)); // divisible powOfTwo lessThanSix
            }
        }
        else if (runtime->GetCLTuneLevel() == None)
        {
            // define not tune method to choose lws
            lws_prefer[0] = 0;
            lws_prefer[1] = 0;
            lws_prefer[2] = 0;
            min_cost = 0;
        }
        if (runtime->GetCLTuneLevel() != None)
        {
            cl_event event;
            size_t gw_arr[3] = {gws[0], gws[1], gws[2]};

            err |= clEnqueueNDRangeKernel(commandQueue[0], mKernel, 3, NULL, gw_arr, NULL, 0, NULL, &event);
            oclCheckError(err, CL_SUCCESS);

            if (err != CL_SUCCESS)
            {
                printf("3D lws null res %s\n", kernelName.c_str());
            }

            float cost_time = runtime->GetCostTime(&event);
            if (cost_time < min_cost)
            {
                lws_prefer[0] = 0;
                lws_prefer[1] = 0;
                lws_prefer[2] = 0;
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
    void OpenCLRuntime::RunKernel2D(const cl_kernel &kernel, const std::vector<size_t> &gws, const std::vector<size_t> &lws,
                                    OpenCLRuntime *runtime, cl_event *eventPtr)
    {
        SNN_ASSERT(lws.size() >= 2);
        size_t internalGlobalWS[2] = {};
        cl_command_queue *commandQueue = runtime->GetCommandQue();
        for (size_t i = 0; i < 2; ++i)
        {
            internalGlobalWS[i] = ROUND_UP(gws[i], MAX((uint32_t)1, lws[i]));
        }
        err = CL_SUCCESS;
        if (lws[0] == 0 || lws[1] == 0)
        {
            err |= clEnqueueNDRangeKernel(commandQueue[0], kernel, 2, NULL, internalGlobalWS, NULL, 0, NULL, NULL);
        }
        else
        {
            size_t internalLocalWS[2] = {lws[0], lws[1]};
            err |= clEnqueueNDRangeKernel(commandQueue[0], kernel, 2, NULL, internalGlobalWS, internalLocalWS, 0, NULL, NULL);
        }
        oclCheckError(err, CL_SUCCESS);
        unsigned int num_flush = runtime->GetQueueNum();
        if (num_flush % 10 == 0)
        {
            clFlush(commandQueue[0]);
        }
    }
    void OpenCLRuntime::RunKernel3D(const cl_kernel &kernel, const std::vector<size_t> &gws, const std::vector<size_t> &lws,
                                    OpenCLRuntime *runtime, cl_event *eventPtr)
    {

        SNN_ASSERT(lws.size() >= 3);
        err = CL_SUCCESS;
        size_t internalGlobalWS[3] = {};
        for (size_t i = 0; i < 3; ++i)
        {
            internalGlobalWS[i] = ROUND_UP(gws[i], MAX((uint32_t)1, lws[i]));
        }
        cl_command_queue *commandQueue = runtime->GetCommandQue();
        if (lws[0] == 0 || lws[1] == 0 || lws[3] == 0)
        {
            err |= clEnqueueNDRangeKernel(commandQueue[0], kernel, 2, NULL, internalGlobalWS, NULL, 0, NULL, NULL);
        }
        else
        {
            size_t internalLocalWS[3] = {lws[0], lws[1], lws[2]};
            err |= clEnqueueNDRangeKernel(commandQueue[0], kernel, 3, NULL, internalGlobalWS, internalLocalWS, 0, NULL, NULL);
        }
        oclCheckError(err, CL_SUCCESS);
        unsigned int num_flush = runtime->GetQueueNum();
        if (num_flush % 10 == 0)
        {
            clFlush(commandQueue[0]);
        }
    }
    unsigned int OpenCLRuntime::GetQueueNum()
    {
        mQueueCount++;
        return mQueueCount;
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
    uint32_t OpenCLRuntime::deviceComputeUnits() const
    {
        return mGPUComputeUnits;
    }
}