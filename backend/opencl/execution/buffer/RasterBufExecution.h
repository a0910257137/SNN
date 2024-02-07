#ifndef RASTERBUFEXECUTION_H
#define RASTERBUFEXECUTION_H
#include "../image/Execution.h"
#include "backend/opencl/core/OpenCLBackend.h"
#include "core/TensorUtils.h"
namespace SNN
{
    class RasterBufExecution : public Execution
    {
    public:
        RasterBufExecution(std::shared_ptr<Tensor> tensor, OpenCLBackend *mbackend);
        virtual ~RasterBufExecution() = default;
        virtual bool onResize(std::shared_ptr<Tensor> tensor);

    private:
        std::vector<std::vector<int>> _generateFilterConvertRegion();

    private:
        OpenCLBackend *mbackend;

    private:
        int mKernelHeight, mKernelWidth;
        int mInputChannel, mOutputChannel;
        cl_mem mTempInput, mTempOutput;
        bool mNeedZero = false;
        bool mFast = false;
    };

} // namespace SNN
#endif // RASTERBUFEXECUTION_H