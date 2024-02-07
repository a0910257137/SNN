#include "coreUtils.h"
namespace SNN
{

    void getImageShape(const std::vector<int> &shape, const OpenCLBufferFormat type, size_t *imageShape)
    {
        SNN_ASSERT(imageShape != nullptr);
        if (type == CONV2D_FILTER)
        {
            imageShape[0] = static_cast<size_t>(shape[1]);
            imageShape[1] = static_cast<size_t>(UP_DIV(shape[0], 4) * shape[2] * shape[3]);
        }
        else if (type == DW_CONV2D_FILTER)
        {
            imageShape[0] = static_cast<size_t>(shape[0] * shape[2] * shape[3]);
            imageShape[1] = static_cast<size_t>(UP_DIV(shape[1], 4));
        }

        else if (type == NHWC_BUFFER || type == NCHW_BUFFER)
        {
            imageShape[0] = static_cast<size_t>(UP_DIV(shape[3], 4) * shape[2]);
            imageShape[1] = static_cast<size_t>(shape[0] * shape[1]);
        }
        else if (type == ARGUMENT)
        {
            if (shape.size() == 4)
            {
                imageShape[0] = static_cast<size_t>(UP_DIV(shape[3], 4));
                imageShape[1] = static_cast<size_t>(1);
            }
            else
            {
                imageShape[0] = UP_DIV(shape[0], 4);
                imageShape[1] = static_cast<size_t>(1);
            }
        }
        else if (type == CONV2D1x1_OPT_FILTER)
        {
            imageShape[0] = static_cast<size_t>(UP_DIV(shape[1], 4));
            imageShape[1] = static_cast<size_t>(shape[2] * shape[3] * shape[0]);
        }
        else
        {
            printf("type not supported !!! \n");
        }
    }

    void CopyBufferToImage(OpenCLRuntime *runtime, const cl_mem &buffer, const cl_mem &image, int *w, int *h, cl_int &err)
    {
        cl_command_queue *commandQueue = runtime->GetCommandQue();
        std::set<std::string> buildOptions;
        if (runtime->isWeightCpuTransHalf() == false)
            buildOptions.emplace("-DBUFFER_INP_FP32");
        cl_kernel kernel = runtime->BuildKernel("copy_buffer_to_image2d", "copy_buffer_to_image2d", buildOptions);
        err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
        oclCheckError(err, CL_SUCCESS);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &image);
        oclCheckError(err, CL_SUCCESS);
        err |= clSetKernelArg(kernel, 2, sizeof(float) * 1, w);
        oclCheckError(err, CL_SUCCESS);
        err |= clSetKernelArg(kernel, 3, sizeof(float) * 1, h);
        oclCheckError(err, CL_SUCCESS);
        size_t GobalSize[3] = {(size_t)w[0], (size_t)h[0]};

        err |= clEnqueueNDRangeKernel(commandQueue[0], kernel, 2, NULL, GobalSize, NULL, 0, NULL, NULL);
        oclCheckError(err, CL_SUCCESS);
        err |= clFinish(commandQueue[0]);
        oclCheckError(err, CL_SUCCESS);
    }
    std::vector<int> TensorShapeFormat(const std::vector<int> &shape, DataFormat data_format)
    {
        int dimensions = shape.size();
        int iN = (0 != shape[0]) ? shape[0] : 1;
        int iH = (0 != shape[1]) ? shape[1] : 1;
        int iW = (0 != shape[2]) ? shape[2] : 1;
        int iC = (0 != shape[3]) ? shape[3] : 1;

        if (data_format == DATA_FORMAT_NHWC)
        {
            iN = (0 < shape[0]) ? shape[0] : 1;
            iH = (0 < shape[1]) ? shape[1] : 1;
            iW = (0 < shape[2]) ? shape[2] : 1;
            iC = (0 < shape[3]) ? shape[3] : 1;
        }
        if (dimensions == 2)
        {
            iN = shape[0];
            iH = 1;
            iW = 1;
            iC = shape[1];
        }
        if (dimensions == 1)
        {
            iN = 1;
            iH = 1;
            iW = 1;
            iC = shape[0];
        }
        std::vector<int> shape_vec{iN, iH, iW, iC};

        return shape_vec;
    }
} // namespace  SNN
