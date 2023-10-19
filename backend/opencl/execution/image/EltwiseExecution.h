#ifndef ELTWISEEXECUTION_H
#define ELTWISEEXECUTION_H
#include "Execution.h"
namespace SNN
{
    class EltwiseExecution : public Execution
    {
    public:
        EltwiseExecution(std::shared_ptr<Tensor> tensor, OpenCLBackend *mbackend, const std::string &compute);
        virtual ~EltwiseExecution() = default;
        virtual bool onResize(std::shared_ptr<Tensor> tensor);
        virtual bool onExecute();

    private:
        uint32_t RealSize(const std::vector<int> &inputShape);
        bool CheckSize(const std::vector<std::vector<int>> &inputShapes, const std::vector<int> &outputShape);

    private:
        OpenCLBackend *mOpenCLBackend;

    private:
        bool mBroadCast;
        float mOperatorData;
        std::string mCompute;
        std::set<std::string> mBuildOptions;
        cl_kernel mKernel;
        std::vector<size_t> mLWS{1, 1, 1};
        std::vector<size_t> mGWS{1, 1, 1};

        size_t mMaxWorkGroupSize;
    };
} // namespace SNN

#endif // ELTWISEEXECUTION_H