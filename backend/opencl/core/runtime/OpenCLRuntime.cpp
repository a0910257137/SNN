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

        // printf("INFO: Allocating device variables\n");
        bool status = OpenCLRuntime::BuildProgramMaps();
        oclCheckError(status, true);
        cl_device_fp_config fp_config;
        clGetDeviceInfo(this->_device[0], CL_DEVICE_SINGLE_FP_CONFIG, sizeof(cl_device_fp_config), &fp_config, 0);
        mIsDeviceSupportedFP16 = fp_config > 0;
        if (mIsDeviceSupportedFP16)
            printf("INFO: Device is supported half-precision\n");
        printf("INFO: ================Finish initialization of OpenCL device================\n");
    }
    OpenCLRuntime::~OpenCLRuntime()
    {
        mProgramMaps.clear();
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

        if (buildProgramInter != mProgramMaps.end())
        {
            program = buildProgramInter->second;
            return true;
        }
        else
        {
            printf("ERROR: Can't load program %s source \n", cl_name);
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
        bool status = LoadProgram(key, program);
        oclCheckError(status, true);
        err = clBuildProgram(program, this->_num_devices, this->_device, buildOptionsStr.c_str(), NULL, NULL);
        oclCheckError(err, CL_SUCCESS);
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