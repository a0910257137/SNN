#include "PostProcessExecution.h"
#include <iostream>

namespace SNN
{
    PostProcessExecution::PostProcessExecution(std::vector<std::shared_ptr<Tensor>> &tensors, OpenCLBackend *mbackend) : Execution(mbackend)
    {

        this->mbackend = mbackend;
        mBuildOptions.emplace("-DBUFFER_IMAGE_IO_TRANS");
        kernelName = "genBbboxLandmarks";
        mKernel = mOpenCLRuntime->BuildKernel("postprocess", kernelName, mBuildOptions);
        mMaxWorkGroupSize = static_cast<size_t>(mOpenCLRuntime->getMaxWorkGroupSize(mKernel));
    }
    bool PostProcessExecution::onOptimizedResize(std::vector<std::shared_ptr<Tensor>> &tensors)
    {
        std::shared_ptr<Tensor> cls_maps = tensors[0];
        std::shared_ptr<Tensor> bbox_maps = tensors[1];
        std::shared_ptr<Tensor> param_maps = tensors[2];
        std::shared_ptr<Tensor> trans_maps = tensors[3];
        const std::vector<int> &clsOutputShape = cls_maps->OutputShape();
        const std::vector<int> &bboxOutputShape = bbox_maps->OutputShape();
        const std::vector<int> &paramOutputShape = param_maps->OutputShape();
        const std::vector<int> &transOutputShape = trans_maps->OutputShape();
        int in_gws[2] = {static_cast<int>(UP_DIV(clsOutputShape[3], 4)) * clsOutputShape[2],
                         static_cast<int>(clsOutputShape[0] * clsOutputShape[1])};
        int clsChannels = clsOutputShape[3], bboxChannels = bboxOutputShape[3], paramChannels = paramOutputShape[3], transChannels = transOutputShape[3];
        featHeight = clsOutputShape[1], featWidth = clsOutputShape[2];
        int bbox_size = bboxOutputShape[0] * bboxOutputShape[1] * bboxOutputShape[2] * bboxOutputShape[3] * sizeof(float);
        int params_size = paramOutputShape[0] * paramOutputShape[1] * paramOutputShape[2] * paramOutputShape[3] * sizeof(float);
        int trans_size = transOutputShape[0] * transOutputShape[1] * transOutputShape[2] * transOutputShape[3] * sizeof(float);
        uint32_t idx = 0;
        cl_int err = 0;
        const cl_mem *clsCLImage = cls_maps->GetDeviceOutputData();
        const cl_mem *bboxCLImage = bbox_maps->GetDeviceOutputData();
        const cl_mem *paramsCLImage = param_maps->GetDeviceOutputData();
        const cl_mem *transCLImage = trans_maps->GetDeviceOutputData();
        bboxHostData = (float *)malloc(bbox_size);
        paramHostData = (float *)malloc(params_size);
        transHostData = (float *)malloc(trans_size);
        // clear the host buffer
        memset(bboxHostData, 0.0f, bbox_size);
        memset(paramHostData, 0.0f, params_size);
        memset(transHostData, 0.0f, trans_size);
        cl_mem bboxCLBuffer = clCreateBuffer(*GPUcontext, CL_MEM_READ_WRITE, bbox_size, NULL, &err);
        cl_mem paramCLBuffer = clCreateBuffer(*GPUcontext, CL_MEM_READ_WRITE, params_size, NULL, &err);
        cl_mem transCLBuffer = clCreateBuffer(*GPUcontext, CL_MEM_READ_WRITE, trans_size, NULL, &err);
        oclCheckError(err, CL_SUCCESS);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &in_gws[0]);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &in_gws[1]);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &bboxCLBuffer);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &paramCLBuffer);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), &transCLBuffer);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &featHeight);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &featWidth);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &clsChannels);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &bboxChannels);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &paramChannels);
        err |= clSetKernelArg(mKernel, idx++, sizeof(int), &transChannels);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), clsCLImage);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), bboxCLImage);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), paramsCLImage);
        err |= clSetKernelArg(mKernel, idx++, sizeof(cl_mem), transCLImage);
        oclCheckError(err, CL_SUCCESS);
        const size_t lws[2] = {16, MAX((size_t)1, mMaxWorkGroupSize / 16)};
        size_t roundUpGroupWorkSize[3] = {ROUND_UP(in_gws[0], lws[0]), ROUND_UP(in_gws[1], lws[1])};
        err |= clEnqueueNDRangeKernel(commandQueue[0], mKernel, 2, NULL, roundUpGroupWorkSize, lws, 0, NULL, NULL);
        oclCheckError(err, CL_SUCCESS);
        err |= clEnqueueReadBuffer(commandQueue[0], bboxCLBuffer, CL_TRUE, 0, bbox_size, bboxHostData, 0, NULL, NULL);
        err |= clEnqueueReadBuffer(commandQueue[0], paramCLBuffer, CL_TRUE, 0, params_size, paramHostData, 0, NULL, NULL);
        err |= clEnqueueReadBuffer(commandQueue[0], transCLBuffer, CL_TRUE, 0, trans_size, transHostData, 0, NULL, NULL);
        err |= clFinish(commandQueue[0]);
        // for (int i = 0; i < bbox_size / 4; i++)
        // {
        //     if (bboxHostData[i] > 0)
        //     {
        //     }
        // }
        oclCheckError(err, CL_SUCCESS);
        clReleaseMemObject(bboxCLBuffer);
        clReleaseMemObject(paramCLBuffer);
        clReleaseMemObject(transCLBuffer);
        return true;
    }
    bool PostProcessExecution::onOptimizedExecute(std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors)
    {
        int numInput = input_tensors.size();
        int numOutput = output_tensors.size();
        this->inputCLData = *(input_tensors[numInput - 1]->GetDeviceOutputData());
        SNN_ASSERT(inputCLData != NULL);
        cl_int err = CL_SUCCESS;
        err |= clSetKernelArg(mKernel, 2, sizeof(cl_mem), &this->inputCLData);
        mOpenCLRuntime->RunKernel2D(this->mKernel, mGWS, mLWS, mOpenCLRuntime);
        output_tensors[numOutput - 1]->SetDeviceOutputData(this->outputCLData);
        // output_tensors[numOutput - 1]->SetDeviceInputData(this->inputCLData);
        bool status = true;
        if (err != CL_SUCCESS)
            return false;
        return status;
    }
} // namespace SNN
