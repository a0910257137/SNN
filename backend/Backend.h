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
#include "include/SNN/Tensor.h"
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
        void ConvertInputBuffer(std::shared_ptr<Tensor> tensor, float *input_data, bool needResize);
        void ReleaseBuffer(std::shared_ptr<Tensor> tensor);
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