
#include "BackendFactory.h"

namespace SNN
{
    BackendFactory::BackendFactory(bool enable_fp16)
    {

        this->enable_fp16 = enable_fp16;
        this->mBackend = (void *)new CPUBackend(this->enable_fp16);
    }
    BackendFactory::~BackendFactory()
    {
    }
    void BackendFactory::SetOpenCLBackend()
    {
        delete this->mBackend;
        this->mBackend = (OpenCLBackend *)new OpenCLBackend(this->enable_fp16);
    }
    void BackendFactory::BuildOperation(Tensor *inputs)
    {
        if (inputs->op_type == DepthwiseConv)
        {
            // pass to execution to convert OpenCL data type
            DepthwiseConvExecution execution(inputs);
        }
        else if (inputs->op_type == Conv)
        {
        }
        else if (inputs->op_type == Concat)
        {
        }
        else if (inputs->op_type == Relu)
        {
        }
        else if (inputs->op_type == AveragePooling2D)
        {
        }
        else if (inputs->op_type == MaxPooling2D)
        {
        }
    }
}
