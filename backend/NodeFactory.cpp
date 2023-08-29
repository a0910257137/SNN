
#include "NodeFactory.h"

namespace SNN
{
    NodeFactory::NodeFactory(bool enable_fp16)
    {

        this->permitFloat16 = enable_fp16;
        this->_mBackend = (void *)new CPUBackend(this->permitFloat16);
    }
    NodeFactory::~NodeFactory()
    {
    }
    void NodeFactory::RegistOpenCLBackend()
    {
        delete (CPUBackend *)this->_mBackend;
        this->_mBackend = (OpenCLBackend *)new OpenCLBackend(this->permitFloat16);
    }

    bool NodeFactory::BuildOperation(Tensor *inputs, int *input_shape, int *output_shape)
    {
        if (inputs->op_type == DepthwiseConv)
        {
            if (this->_mBackend != nullptr)
            {
                DepthwiseConvExecution op((OpenCLBackend *)this->_mBackend);
                op.onInit(inputs);
                op.onResize(inputs, input_shape, output_shape);
                exit(1);
            }
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

        return true;
    }

    // bool NodeFactory::AllocateMemory(Tensor *inputs)
    // {
    //     // apply onResize tensor
    //     if (inputs->op_type == DepthwiseConv)
    //     {
    //         if (this->_mBackend != nullptr)
    //             DepthwiseConvExecution op(inputs, (OpenCLBackend *)this->_mBackend);
    //     }
    // }
}
