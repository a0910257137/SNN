#ifndef ELTWISEEXECUTION_H
#define ELTWISEEXECUTION_H
#include "Execution.h"
namespace SNN
{
    class EltwiseExecution : public Execution
    {
    public:
        explicit EltwiseExecution(std::shared_ptr<Tensor> tensor, OpenCLBackend *mbackend, const std::string &compute);
        virtual ~EltwiseExecution() = default;
        virtual bool onResize(std::shared_ptr<Tensor> tensor);
        virtual bool onExecute(std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
        virtual bool onOptimizedExecute(std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors) override;

    private:
        uint32_t RealSize(const std::vector<int> &inputShape);
        bool CheckSize(const std::vector<std::vector<int>> &inputShapes, const std::vector<int> &outputShape);

    private:
        OpenCLBackend *mOpenCLBackend;

    private:
        // cl_mem *inputCLData0, *inputCLData1, *outputCLData;
        cl_mem inputCLData0 = NULL, inputCLData1 = NULL, outputCLData = NULL;

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