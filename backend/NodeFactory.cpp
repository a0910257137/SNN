
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
        else if (tensor->GetOpType() == AveragePooling2D)
        {
        }
        else if (tensor->GetOpType() == MaxPooling2D)
        {
        }
        // else if (kernelData->op_type == Relu)
        // {
        // }

        return true;
    }

}
