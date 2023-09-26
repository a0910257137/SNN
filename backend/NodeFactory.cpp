
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
        delete this->_mBackend;
    }

    bool NodeFactory::BuildOperation(std::shared_ptr<Tensor> tensor)
    {
        if (tensor->GetOpType() == DepthwiseConv)
        {
            DepthwiseConvExecution op(tensor, (OpenCLBackend *)this->_mBackend);
            op.onResize(tensor);
            exit(1);
        }
        //     // if (kernelData->op_type == DepthwiseConv)
        //     // {
        //     // }
        //     // else if (kernelData->op_type == Conv)
        //     // {
        //     // }
        //     // else if (kernelData->op_type == Concat)
        //     // {
        //     // }
        //     // else if (kernelData->op_type == Relu)
        //     // {
        //     // }
        //     // else if (kernelData->op_type == AveragePooling2D)
        //     // {
        //     // }
        //     // else if (kernelData->op_type == MaxPooling2D)
        //     // {
        //     // }

        return true;
    }

}
