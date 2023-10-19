#ifndef CONCATEXECUTION_H
#define CONCATEXECUTION_H
#include "Execution.h"
namespace SNN
{

    class ConcatExecution : public Execution
    {
    public:
        ConcatExecution(std::shared_ptr<Tensor> tensor, OpenCLBackend *mbackend);
        virtual ~ConcatExecution() = default;
        virtual bool onResize(std::shared_ptr<Tensor> tensor);
        virtual bool onExecute();

    private:
        bool Concat2(std::shared_ptr<Tensor> tensor);

    private:
        OpenCLBackend *mOpenCLBackend;

    private:
        cl_kernel mKernel = NULL;
        size_t mMaxWorkGroupSize;
        int axis;
        int numInputs;
        std::vector<size_t> mGWS{1, 1, 1};
        std::vector<size_t> mLWS{1, 1, 1};
    };

} // namespace SNN
#endif // CONCAT_H