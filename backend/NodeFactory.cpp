#include "NodeFactory.h"
#include "include/SNN/common.h"

namespace SNN
{
    NodeFactory::NodeFactory(BackendConfig &cfg)
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
    NodeFactory::~NodeFactory()
    {
        delete (void *)this->_mBackend;
    }
    bool NodeFactory::BuildOperation(std::shared_ptr<Tensor> tensor)
    {
        if (tensor->GetOpType() == DepthwiseConv)
        {
            DepthwiseConvExecution op(tensor, (OpenCLBackend *)this->_mBackend);
            op.onResize(tensor);
        }
        else if (tensor->GetOpType() == Conv2D)
        {
            ConvExecution op(tensor, (OpenCLBackend *)this->_mBackend);
            op.onResize(tensor);
        }
        else if (tensor->GetOpType() == Concat)
        {
        }
        else if (tensor->GetOpType() == AVERAGE_POOL_2D)
        {
            PoolExecution op(tensor, (OpenCLBackend *)this->_mBackend);
            op.onResize(tensor);
        }
        else if (tensor->GetOpType() == MAX_POOL_2D)
        {
            PoolExecution op(tensor, (OpenCLBackend *)this->_mBackend);
            op.onResize(tensor);
        }
        else if (tensor->GetOpType() == RESIZE_NEAREST_NEIGHBOR)
        {
            InterpExecution op(tensor, (OpenCLBackend *)this->_mBackend);
            op.onResize(tensor);
        }
        else if (tensor->GetOpType() == CONCATENATION)
        {
            ConcatExecution op(tensor, (OpenCLBackend *)this->_mBackend);
            op.onResize(tensor);
        }
        else if (tensor->GetOpType() == ADD)
        {
            const std::string comput = "in0+in1";
            EltwiseExecution op(tensor, (OpenCLBackend *)this->_mBackend, comput);
            op.onResize(tensor);
            exit(1);
        }
        else if (tensor->GetOpType() == SUB)
        {
            const std::string comput = "in0-in1";
            EltwiseExecution op(tensor, (OpenCLBackend *)this->_mBackend, comput);
            op.onResize(tensor);
        }
        else if (tensor->GetOpType() == MUL)
        {
            const std::string comput = "in0*in1";
            EltwiseExecution op(tensor, (OpenCLBackend *)this->_mBackend, comput);
            op.onResize(tensor);
        }
        else if (tensor->GetOpType() == REALDIV)
        {
            const std::string comput = "sign(in1)*in0/(fabs(in1)>(FLOAT4)((FLOAT)0.0000001)?fabs(in1):(FLOAT4)((FLOAT)0.0000001))";
            EltwiseExecution op(tensor, (OpenCLBackend *)this->_mBackend, comput);
            op.onResize(tensor);
        }
        return true;
    }

}
