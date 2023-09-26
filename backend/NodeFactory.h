#ifndef BACKENDFACTORY_H
#define BACKENDFACTORY_H
#include "backend/opencl/core/OpenCLBackend.h"
#include "backend/cpu/CPUBackend.h"
#include "backend/opencl/execution/image/DepthwiseConvExecution.h"
#include "include/SNN/Tensor.h"

namespace SNN
{
    class NodeFactory
    {
    public:
        NodeFactory(BackendConfig &cfg);
        ~NodeFactory();
        NodeFactory(const NodeFactory &) = delete;
        NodeFactory &operator=(const NodeFactory &) = delete;
        bool BuildOperation(std::shared_ptr<Tensor> tensor);
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