#ifndef CONVEXECUTION_H
#define COVEXECUTION_H
#include "ConvBaseExecution.h"
#include "Execution.h"
#include "backend/opencl/core/OpenCLBackend.h"
#include "core/ConvolutionCommon.h"
#include "backend/opencl/core/ImageBufferConverter.h"

namespace SNN
{
    class ConvExecution : public ConvBaseExecution
    {
    public:
        explicit ConvExecution(std::shared_ptr<Tensor> tensor, OpenCLBackend *mbackend);
        ~ConvExecution();
        ConvExecution(const ConvExecution &) = delete;
        ConvExecution &operator=(const ConvExecution &) = delete;
        bool onResize(std::shared_ptr<Tensor> tensor);
        bool onExecute();

    private:
        OpenCLBackend *mbackend;

    private:
        std::vector<int> mStrides{1, 1};
        std::vector<int> mPaddings{0, 0};
        std::vector<int> mDilations{1, 1};
        std::vector<size_t> mGlobalWorkSize{1, 1, 1};
        std::vector<size_t> mLocalWorkSize{1, 1, 1};
        cl_kernel mKernel;
        uint32_t mMaxWorkGroupSize;
        bool mIsTurn;
        bool mConv1x1Opt{false};
        bool mUseLocalMem{false};
        cl_mem mKernelBuffer;
        cl_mem mBiasBuffer;
        std::set<std::string> mBuildOptions;
        std::shared_ptr<ConvolutionCommon> mConvCommon;
        ImageBufferConverter *mImageConvert;
    };
} // namespace SNN

#endif