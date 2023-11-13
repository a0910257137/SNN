#include "ConvBaseExecution.h"

namespace SNN
{
    ConvBaseExecution::ConvBaseExecution(std::shared_ptr<Tensor> tensor, OpenCLBackend *mbackend) : Execution(mbackend)
    {

        int biasSize = tensor->bias_bytes() / sizeof(float);
        const std::shared_ptr<std::vector<std::pair<float *, float *>>> mainMemory = tensor->GetMainMemory();
        const std::vector<uint8_t> &ptrIndex = tensor->GetMemoryPtrIndex();
        std::pair<float *, float *> &weight_bias = mainMemory->at(ptrIndex[0]);
        const float *weightData = weight_bias.first;
        const float *biasData = weight_bias.second;
        int buffer_size = ALIGN_UP4(biasSize);
        if (mOpenCLRuntime->isWeightCpuTransHalf())
            buffer_size *= sizeof(cl_half);
        else
            buffer_size *= sizeof(float);
        cl_mem biasBuffer;
        cl_int err = CL_SUCCESS;
        biasBuffer = clCreateBuffer(*GPUcontext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                    buffer_size, NULL, &err);
        oclCheckError(err, CL_SUCCESS);
        float *biasPtrCL = (float *)clEnqueueMapBuffer(commandQueue[0], biasBuffer, true, CL_MAP_WRITE, 0, buffer_size, 0, NULL, NULL, &err);
        oclCheckError(err, CL_SUCCESS);
        if (biasPtrCL != nullptr && err == CL_SUCCESS)
        {

            if (mOpenCLRuntime->isWeightCpuTransHalf())
            {
                for (int i = 0; i < biasSize; ++i)
                    ((cl_half *)biasPtrCL)[i] = (cl_half)(biasData[i]);

                for (int i = biasSize; i < ALIGN_UP4(biasSize); ++i)
                    ((cl_half *)biasPtrCL)[i] = (cl_half)(0.0f);
            }
            else
            {
                memset(biasPtrCL, 0, ALIGN_UP4(biasSize) * sizeof(float));
                memcpy(biasPtrCL, biasData, tensor->bias_bytes());
            }
        }
        else
            printf("ERROR: Map memory error in biasPtrCL !! \n");
        err = clEnqueueUnmapMemObject(commandQueue[0], biasBuffer, biasPtrCL, 0, NULL, NULL);
        oclCheckError(err, CL_SUCCESS);
        int w[1] = {UP_DIV(biasSize, 4)}, h[1] = {1};
        cl_mem mBias = clCreateImage2D(*GPUcontext, CL_MEM_READ_WRITE, &clImageFormat, w[0], h[0], 0, NULL, &err);
        // oclCheckError(err, CL_SUCCESS);
        // err = clFinish(commandQueue[0]);
        oclCheckError(err, CL_SUCCESS);
        CopyBufferToImage(this->mOpenCLRuntime, biasBuffer, mBias, w, h, err);
        oclCheckError(err, CL_SUCCESS);
        err |= clFinish(commandQueue[0]);
        oclCheckError(err, CL_SUCCESS);
        tensor->SetDeviceBias(mBias);
        clReleaseMemObject(biasBuffer);
    }
} // SNN