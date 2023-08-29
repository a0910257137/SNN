#ifndef BACKENDFACTORY_H
#define BACKENDFACTORY_H
#include "backend/opencl/core/OpenCLBackend.h"
#include "backend/cpu/CPUBackend.h"
#include "backend/opencl/execution/image/DepthwiseConvExecution.h"
#include "include/SNN/common.h"
namespace SNN
{
    class NodeFactory
    {
    public:
        NodeFactory(bool enable_fp16);
        ~NodeFactory();
        NodeFactory(const NodeFactory &) = delete;
        NodeFactory &operator=(const NodeFactory &) = delete;
        bool BuildOperation(Tensor *inputs, int *input_shape, int *output_shape);
        void RegistOpenCLBackend();
        void RegistCPUBackend();
        bool AllocateMemory(Tensor *inputs);
        bool GetOperationType() const
        {
            return permitFloat16;
        }
        void *mBackend()
        {
            return _mBackend;
        }

    protected:
        void *_mBackend = nullptr;

    private:
        bool permitFloat16 = NULL;
    };
}
#endif // NODEDFACTORY_H