#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>
#include <dirent.h>

char *removeExtension(const char *myStr)
{
    char *retStr;
    char *lastExt;
    if (myStr == NULL)
        return NULL;
    if ((retStr = malloc(strlen(myStr) + 1)) == NULL)
        return NULL;
    strcpy(retStr, myStr);
    lastExt = strrchr(retStr, '.');
    if (lastExt != NULL)
        *lastExt = '\0';
    return retStr;
}
void GetFiles(const char *path, char *file_paths[20])
{
    DIR *dir;
    int i = 0;
    struct dirent *ent;
    const char *b = ".", *bb = "..";
    const char *file_name, *file_path;
    FILE *ptr;
    if ((dir = opendir(path)) != NULL)
    {
        while ((ent = readdir(dir)) != NULL)
        {
            file_name = ent->d_name;

            if (*file_name == *b || *file_name == *bb)
                continue;
            // printf("%s\n", file_name);
            file_paths[i] = file_name;
            i++;
        }
    }
    // for (int i = 0; i < 9; i++)
    // {
    //     printf("%s\n", file_paths[i]);
    // }
    // exit(1);
}
int read_file(char **output, size_t *size, const char *name)
{
    FILE *fp = fopen(name, "rb");
    if (!fp)
    {
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    *size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    *output = (char *)malloc(*size);
    if (!*output)
    {
        fclose(fp);
        return -1;
    }
    fread(*output, *size, 1, fp);
    fclose(fp);
    return 0;
}

int write_file(const char *name, const unsigned char *content, size_t size)
{
    FILE *fp = fopen(name, "wb+");
    if (!fp)
    {
        return -1;
    }
    fwrite(content, size, 1, fp);
    fclose(fp);
    return 0;
}

cl_int get_platform_list(cl_platform_id **platforms_out,
                         cl_uint *num_platforms_out)
{
    cl_int err;

    // Read the number of platforms
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS)
    {
        return err;
    }
    if (num_platforms == 0)
    {
        return CL_INVALID_VALUE;
    }

    // Allocate the array of cl_platform_id
    cl_platform_id *platforms =
        (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
    if (!platforms)
    {
        return CL_OUT_OF_HOST_MEMORY;
    }

    // Get the result
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (err != CL_SUCCESS)
    {
        free(platforms);
        return err;
    }

    *platforms_out = platforms;
    *num_platforms_out = num_platforms;
    return CL_SUCCESS;
}

void free_platform_list(cl_platform_id *platforms, cl_uint num_platforms)
{
    free(platforms);
}

char *get_platform_info(cl_platform_id platform, cl_platform_info param)
{
    cl_int err;

    // Read the size of the buffer for platform name
    size_t buf_size;
    err = clGetPlatformInfo(platform, param, 0, NULL, &buf_size);
    if (err != CL_SUCCESS)
    {
        return NULL;
    }
    if (buf_size == 0)
    {
        return NULL;
    }

    // Allocate the buffer for platform name
    char *buf = (char *)malloc(buf_size);
    if (!buf)
    {
        return NULL;
    }

    // Read the platform name
    err = clGetPlatformInfo(platform, param, buf_size, buf, NULL);
    if (err != CL_SUCCESS)
    {
        free(buf);
        return NULL;
    }

    return buf;
}

cl_int get_device_list(cl_device_id **devices_out, cl_uint *num_devices_out,
                       cl_platform_id platform)
{
    cl_int err;

    // Read the number of devices of the given platform
    cl_uint num_devices;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL,
                         &num_devices);
    if (err != CL_SUCCESS)
    {
        return err;
    }

    // Allocate the array of cl_device_id
    cl_device_id *devices =
        (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
    if (!devices)
    {
        return CL_OUT_OF_HOST_MEMORY;
    }

    // Read the result
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices,
                         devices, NULL);
    if (err != CL_SUCCESS)
    {
        free(devices);
        return err;
    }

    *devices_out = devices;
    *num_devices_out = num_devices;
    return CL_SUCCESS;
}

void free_device_list(cl_device_id *devices, cl_uint num_devices)
{
    cl_uint i;
    for (i = 0; i < num_devices; ++i)
    {
        clReleaseDevice(devices[i]);
    }
    free(devices);
}

// TODO:
cl_int write_binaries(cl_program program, unsigned num_devices,
                      cl_uint platform_idx, const char *filename)
{
    unsigned i;
    cl_int err = CL_SUCCESS;
    size_t *binaries_size = NULL;
    unsigned char **binaries_ptr = NULL;

    // Read the binaries size
    size_t binaries_size_alloc_size = sizeof(size_t) * num_devices;
    binaries_size = (size_t *)malloc(binaries_size_alloc_size);
    if (!binaries_size)
    {
        err = CL_OUT_OF_HOST_MEMORY;
        goto cleanup;
    }

    err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                           binaries_size_alloc_size, binaries_size, NULL);
    if (err != CL_SUCCESS)
    {
        goto cleanup;
    }

    // Read the binaries
    size_t binaries_ptr_alloc_size = sizeof(unsigned char *) * num_devices;
    binaries_ptr = (unsigned char **)malloc(binaries_ptr_alloc_size);
    if (!binaries_ptr)
    {
        err = CL_OUT_OF_HOST_MEMORY;
        goto cleanup;
    }
    memset(binaries_ptr, 0, binaries_ptr_alloc_size);
    for (i = 0; i < num_devices; ++i)
    {
        binaries_ptr[i] = (unsigned char *)malloc(binaries_size[i]);
        if (!binaries_ptr[i])
        {
            err = CL_OUT_OF_HOST_MEMORY;
            goto cleanup;
        }
    }

    err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, binaries_ptr_alloc_size,
                           binaries_ptr, NULL);
    if (err != CL_SUCCESS)
    {
        goto cleanup;
    }

    // Write the binaries to file
    for (i = 0; i < num_devices; ++i)
    {
        char *saved_name = removeExtension(filename);
        char *name_with_extension = malloc(strlen(saved_name) + strlen(".bin") + 1);
        strcpy(name_with_extension, saved_name);
        strcat(name_with_extension, ".bin");
        write_file(name_with_extension, binaries_ptr[i], binaries_size[i]);
        free(name_with_extension);
    }

cleanup:
    // Free the return value buffer
    if (binaries_ptr)
    {
        for (i = 0; i < num_devices; ++i)
        {
            free(binaries_ptr[i]);
        }
        free(binaries_ptr);
    }
    free(binaries_size);

    return err;
}

cl_int compile_program(cl_uint *num_devices_out, const char *src,
                       size_t src_size, cl_platform_id platform,
                       cl_uint platform_idx, const char *filename)
{
    cl_int err = CL_SUCCESS;

    // Get the device list
    cl_device_id *devices = NULL;
    cl_uint num_devices = 0;
    get_device_list(&devices, &num_devices, platform);
    *num_devices_out = num_devices;
    // Create context
    cl_context_properties ctx_properties[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};

    cl_context ctx = clCreateContext(ctx_properties, num_devices, devices, NULL,
                                     NULL, &err);
    if (err != CL_SUCCESS)
    {
        goto cleanup;
    }

    // Create program
    cl_program program = clCreateProgramWithSource(ctx, 1, &src, &src_size, &err);
    if (err != CL_SUCCESS)
    {
        goto cleanup;
    }
    const char *buildOptionsStr = "-DFLOAT=float -DFLOAT2=float2 -DFLOAT4=float4 -DFLOAT8=float8 -DRI_F=read_imagef -DFLOAT16=float16 -DWI_F=write_imagef -DCONVERT_FLOAT4=convert_float4  -DSET_ATTRIBUTE=false -cl-mad-enable -DBIAS -DRELU -DOPERATOR=in0+in1";
    // Compile program
    // const char *buildOptionsStr = "-cl-mad-enable";
    err = clBuildProgram(program, num_devices, devices, buildOptionsStr, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        goto cleanup_program;
    }
    write_binaries(program, num_devices, platform_idx, filename);
cleanup_program:
    // Free the built program
    clReleaseProgram(program);

cleanup:
    // Free the device list
    free_device_list(devices, num_devices);

    return err;
}

void compile_all(const char *src, size_t src_size, const char *filename)
{
    cl_uint i;

    // Get the platform list
    cl_platform_id *platforms = NULL;
    cl_uint num_platforms = 0;
    if (get_platform_list(&platforms, &num_platforms) != CL_SUCCESS)
    {
        return;
    }
    // For each platform compile binaries for each devices
    for (i = 0; i < num_platforms; ++i)
    {
        // Compile for each devices
        cl_uint num_devices = 0;
        cl_int err = compile_program(&num_devices, src, src_size, platforms[i], i, filename);
        // Print the result
        char *platform_name = get_platform_info(platforms[i], CL_PLATFORM_NAME);
        printf("PLATFORM [%s]  -->  %s (%u)\n",
               (platform_name ? platform_name : ""),
               ((err == CL_SUCCESS) ? "SUCCESS" : "FAILURE"),
               (unsigned)num_devices);

        fflush(stdout);
        free(platform_name);
    }

    // Free the platform list
    free_platform_list(platforms, num_platforms);
}
int main(int argc, char const *argv[])
{
    /* code */
    if (argc < 2)
    {
        fprintf(stderr, "USAGE: cl-compile [SOURCE]\n");
        exit(EXIT_FAILURE);
    }
    const char *root_dir = argv[1];
    // Read the source file
    char *src = NULL;
    size_t src_size = 0;
    char *file_names[20];
    GetFiles(root_dir, file_names);
    for (int i = 0; i < 11; i++)
    {
        const char *filename = file_names[i];
        printf("%s\n", filename);
        char *name_with_extension = malloc(strlen(root_dir) + strlen(filename) + 1);
        strcpy(name_with_extension, root_dir);
        strcat(name_with_extension, filename);
        if (read_file(&src, &src_size, name_with_extension) != 0)
        {

            fprintf(stderr, "ERROR: Failed to read: %s\n", filename);
            exit(EXIT_FAILURE);
        }
        compile_all(src, src_size, filename);
        free(name_with_extension);
    }
    free(src);
    return 0;
}
