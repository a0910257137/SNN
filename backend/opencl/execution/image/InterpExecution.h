#ifndef INTERPEXECUTION_H
#define INTERPEXECUTION_H
#include "Execution.h"
// #include "include/SNN/common.h"

namespace SNN
{
    class InterpExecution : public Execution
    {
    public:
        InterpExecution(std::shared_ptr<Tensor> tensor, OpenCLBackend *mbackend);
        virtual ~InterpExecution() = default;
        virtual bool onResize(std::shared_ptr<Tensor> tensor);
        virtual bool onExecute();

    private:
        OpenCLBackend *mOpenCLBackend;

    private:
        cl_kernel mKernel;
        std::vector<size_t> mLWS{0, 0, 0, 0};
        std::vector<size_t> mGWS{0, 0, 0, 0};
        size_t mMaxWorkGroupSize;
        float mCordTransform[4];
    };
}

#endif // INTERPEXECUTION_H