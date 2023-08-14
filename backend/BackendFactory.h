#ifndef BACKENDFACTORY_H
#define BACKENDFACTORY_H
#include "backend/opencl/core/OpenCLBackend.h"
#include "backend/cpu/CPUBackend.h"
#include "backend/opencl/execution/image/DepthwiseConvExecution.h"
#include "include/SNN/common.h"
namespace SNN
{
    class BackendFactory
    {
    public:
        BackendFactory(bool enable_fp16);
        ~BackendFactory();
        BackendFactory(const BackendFactory &) = delete;
        BackendFactory &operator=(const BackendFactory &) = delete;
        void BuildOperation(Tensor *inputs);
        void SetOpenCLBackend();

        const bool GetOperationType() const
        {
            return enable_fp16;
        }
        const void *GetBackend() const
        {
            return mBackend;
        }

    private:
        const void *mBackend = nullptr;
        bool enable_fp16 = NULL;
    };
}
#endif // BACKENDFACTORY_H