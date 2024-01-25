#ifndef BACKEND_H
#define BACKEND_H
#include "backend/opencl/core/OpenCLBackend.h"
#include "backend/cpu/CPUBackend.h"
#include "backend/opencl/execution/image/DepthwiseConvExecution.h"
#include "backend/opencl/execution/image/ConvExecution.h"
#include "backend/opencl/execution/image/InterpExecution.h"
#include "backend/opencl/execution/image/PoolExecution.h"
#include "backend/opencl/execution/image/ConcatExecution.h"
#include "backend/opencl/execution/image/AddNExecution.h"
#include "backend/opencl/execution/image/EltwiseExecution.h"
#include "backend/opencl/execution/image/DeconvExecution.h"
#include "backend/opencl/execution/image/InputExecution.h"
#include "backend/opencl/execution/image/StemExecution.h"
#include "backend/opencl/execution/image/SeperableConvExecution.h"
#include "backend/opencl/execution/image/DoubleConvExecution.h"
#include "backend/opencl/execution/image/AddExecution.h"
#include "include/SNN/common.h"
namespace SNN
{
    class Backend
    {
    public:
        Backend(BackendConfig &cfg);
        ~Backend();
        Backend(const Backend &) = delete;
        Backend &operator=(const Backend &) = delete;
        void BuildOperation(std::shared_ptr<Tensor> tensor, std::vector<std::shared_ptr<Execution>> &netOpList);
        void ReleaseBuffer(std::shared_ptr<Tensor> tensor);
        void MergedOperators(std::vector<std::shared_ptr<Tensor>> &tensors, std::vector<std::shared_ptr<Execution>> &netOpContainer);
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
#endif // BACKEND_H