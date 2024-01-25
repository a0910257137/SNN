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
        else if (tensor->GetOpType() == INPUTDATA)
        {
            std::shared_ptr<InputExecution> op(new InputExecution(tensor, (OpenCLBackend *)this->_mBackend));
            op->onResize(tensor);
            netOpContainer.emplace_back(op);
        }
    }

    void Backend::MergedOperators(std::vector<std::shared_ptr<Tensor>> &tensors, std::vector<std::shared_ptr<Execution>> &netOpContainer)
    {
        if (tensors.size() == 2)
        {
            if ((tensors[0]->GetOpType() == INPUTDATA) && ((tensors[1]->GetOpType() == CONV2D) || (tensors[1]->GetOpType() == DEPTHWISECONV2D)))
            {
                std::shared_ptr<StemExecution> op(new StemExecution(tensors, (OpenCLBackend *)this->_mBackend));
                op->onOptimizedResize(tensors);
                netOpContainer.emplace_back(op);
            }
            else if ((tensors[0]->GetOpType() == DEPTHWISECONV2D) && ((tensors[1]->GetOpType() == CONV2D)))
            {
                std::shared_ptr<SeperableConvExecution> op(new SeperableConvExecution(tensors, (OpenCLBackend *)this->_mBackend));
                op->onOptimizedResize(tensors);
                netOpContainer.emplace_back(op);
            }
        }
        else if (tensors.size() == 3)
        {
            if (((tensors[0]->GetOpType() == DEPTHWISECONV2D) && ((tensors[1]->GetOpType() == DEPTHWISECONV2D) && (tensors[2]->GetOpType() == DEPTHWISECONV2D))) || ((tensors[0]->GetOpType() == DEPTHWISECONV2D) && ((tensors[1]->GetOpType() == CONV2D) && (tensors[2]->GetOpType() == CONV2D))))
            {
                std::shared_ptr<SeperableConvExecution> op(new SeperableConvExecution(tensors, (OpenCLBackend *)this->_mBackend));
                op->onOptimizedResize(tensors);
                netOpContainer.emplace_back(op);
            }
            else if (((tensors[0]->GetOpType() == DEPTHWISECONV2D) && (tensors[1]->GetOpType() == DEPTHWISECONV2D) && (tensors[2]->GetOpType() == ADD)) || ((tensors[0]->GetOpType() == DEPTHWISECONV2D) && (tensors[1]->GetOpType() == ADD) && (tensors[2]->GetOpType() == ADD)) || ((tensors[0]->GetOpType() == DEPTHWISECONV2D) && (tensors[1]->GetOpType() == CONV2D) && (tensors[2]->GetOpType() == ADD)))
            {
                std::shared_ptr<AddExecution> op(new AddExecution(tensors, (OpenCLBackend *)this->_mBackend));
                op->onOptimizedResize(tensors);
                netOpContainer.emplace_back(op);
            }
            else if ((tensors[0]->GetOpType() == RESIZE_NEAREST_NEIGHBOR) && (tensors[1]->GetOpType() == CONV2D) && (tensors[2]->GetOpType() == ADD))
            {
                std::shared_ptr<AddExecution> op(new AddExecution(tensors, (OpenCLBackend *)this->_mBackend));
                op->onOptimizedResize(tensors);
                netOpContainer.emplace_back(op);
            }
            else if ((tensors[0]->GetOpType() == CONV2D) && (tensors[1]->GetOpType() == CONV2D) && (tensors[2]->GetOpType() == ADD))
            {
                std::shared_ptr<AddExecution> op(new AddExecution(tensors, (OpenCLBackend *)this->_mBackend));
                op->onOptimizedResize(tensors);
                netOpContainer.emplace_back(op);
            }
        }
        else
        {
            printf("ERROR: The fusion operator are not implemented.... !!\n");
            exit(1);
        }
    }
    void Backend::ReleaseBuffer(std::shared_ptr<Tensor> tensor)
    {
        clReleaseMemObject(*tensor->GetDeviceInputData());
    }
}
