#include "Backend.h"

namespace SNN
{
    Backend::Backend(BackendConfig &cfg)
    {
        if (cfg.precisionType == BackendConfig::Precision_FP32)
            this->permitFloat16 = true;

        if (cfg.backendType == BackendConfig::OpenCL)
        {
            printf("INFO: Enable OpenCL backend\n");
            this->_mBackend = (OpenCLBackend *)new OpenCLBackend(this->permitFloat16);
        }
        else
        {
            printf("INFO: Enable CPU backend\n");
            this->_mBackend = (void *)new CPUBackend(this->permitFloat16);
        }
    }
    Backend::~Backend()
    {
        delete (void *)this->_mBackend;
    }

    void Backend::ConvertInputBuffer(std::shared_ptr<Tensor> tensor, float *input_data)
    {
        OpenCLBackend *mbk = (OpenCLBackend *)_mBackend;
        OpenCLRuntime *_mCLRuntime = mbk->CLRuntime();
        cl_context *GPUcontext = _mCLRuntime->GetGPUContext();
        const std::vector<std::vector<int>> &inputShapes = tensor->InputShape();
        const std::vector<int> &inputShape = inputShapes[0];
        int N = inputShape[0];
        int H = inputShape[1];
        int W = inputShape[2];
        int C = inputShape[3];
        int buffer_sizes = N * H * W * C * sizeof(float);
        cl_int err = 0;
        cl_command_queue *commandQueue = _mCLRuntime->GetCommandQue();
        cl_mem mhostBuffer = clCreateBuffer(*GPUcontext, CL_MEM_READ_WRITE, buffer_sizes, NULL, &err);
        err |= clEnqueueWriteBuffer(commandQueue[0], mhostBuffer, CL_TRUE, 0, buffer_sizes, input_data, 0, NULL, NULL);
        oclCheckError(err, CL_SUCCESS);
        mbk->mHostBuffer.first = buffer_sizes;
        mbk->mHostBuffer.second = mhostBuffer;
        cl_mem inputCLData = mbk->ConvertNHWCBufferToImage(tensor->InputShape()[0], tensor->data_format, false, false);
        SNN_ASSERT(inputCLData != NULL);
        tensor->SetDeviceInputData(inputCLData);
        tensor->SetDeviceOutputData(inputCLData);
        clReleaseMemObject(mhostBuffer);
    }
    void Backend::BuildOperation(std::shared_ptr<Tensor> tensor, std::vector<std::shared_ptr<Execution>> &netOpContainer)
    {
        if (tensor->GetOpType() == DEPTHWISECONV2D)
        {
            std::shared_ptr<DepthwiseConvExecution> op(new DepthwiseConvExecution(tensor, (OpenCLBackend *)this->_mBackend));
            op->onResize(tensor);
            netOpContainer.emplace_back(op);
        }
        else if (tensor->GetOpType() == CONV2D)
        {
            std::shared_ptr<ConvExecution> op(new ConvExecution(tensor, (OpenCLBackend *)this->_mBackend));
            bool status = op->onResize(tensor);
            netOpContainer.emplace_back(op);
        }
        else if (tensor->GetOpType() == DECONV2D)
        {
            std::shared_ptr<DeconvExecution> op(new DeconvExecution(tensor, (OpenCLBackend *)this->_mBackend));
            op->onResize(tensor);
            netOpContainer.emplace_back(op);
        }
        else if (tensor->GetOpType() == AVERAGE_POOL_2D)
        {
            std::shared_ptr<PoolExecution> op(new PoolExecution(tensor, (OpenCLBackend *)this->_mBackend));
            op->onResize(tensor);
            netOpContainer.emplace_back(op);
        }
        else if (tensor->GetOpType() == MAX_POOL_2D)
        {
            std::shared_ptr<PoolExecution> op(new PoolExecution(tensor, (OpenCLBackend *)this->_mBackend));
            op->onResize(tensor);
            netOpContainer.emplace_back(op);
        }
        else if (tensor->GetOpType() == RESIZE_NEAREST_NEIGHBOR)
        {
            std::shared_ptr<InterpExecution> op(new InterpExecution(tensor, (OpenCLBackend *)this->_mBackend));
            op->onResize(tensor);
            netOpContainer.emplace_back(op);
        }
        else if (tensor->GetOpType() == CONCATENATION)
        {
            std::shared_ptr<ConcatExecution> op(new ConcatExecution(tensor, (OpenCLBackend *)this->_mBackend));
            op->onResize(tensor);
            netOpContainer.emplace_back(op);
        }
        else if (tensor->GetOpType() == ADD)
        {
            const std::string comput = "in0+in1";
            std::shared_ptr<EltwiseExecution> op(new EltwiseExecution(tensor, (OpenCLBackend *)this->_mBackend, comput));
            op->onResize(tensor);
            netOpContainer.emplace_back(op);
        }
        else if (tensor->GetOpType() == SUB)
        {
            const std::string comput = "in0-in1";
            std::shared_ptr<EltwiseExecution> op(new EltwiseExecution(tensor, (OpenCLBackend *)this->_mBackend, comput));
            op->onResize(tensor);
            netOpContainer.emplace_back(op);
        }
        else if (tensor->GetOpType() == MUL)
        {
            const std::string comput = "in0*in1";
            std::shared_ptr<EltwiseExecution> op(new EltwiseExecution(tensor, (OpenCLBackend *)this->_mBackend, comput));
            op->onResize(tensor);
            netOpContainer.emplace_back(op);
        }
        else if (tensor->GetOpType() == REALDIV)
        {
            const std::string comput = "sign(in1)*in0/(fabs(in1)>(FLOAT4)((FLOAT)0.0000001)?fabs(in1):(FLOAT4)((FLOAT)0.0000001))";
            std::shared_ptr<EltwiseExecution> op(new EltwiseExecution(tensor, (OpenCLBackend *)this->_mBackend, comput));
            op->onResize(tensor);
            netOpContainer.emplace_back(op);
        }
    }
    void Backend::ReleaseBuffer(std::shared_ptr<Tensor> tensor)
    {
        clReleaseMemObject(*tensor->GetDeviceInputData());
    }
}
