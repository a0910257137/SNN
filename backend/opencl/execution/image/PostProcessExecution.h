#ifndef POSTPROCESSEXECUTION_H
#define POSTPROCESSEXECUTION_H
#include "Execution.h"
#include "backend/opencl/core/OpenCLBackend.h"

namespace SNN
{
    class PostProcessExecution : public Execution
    {
    public:
        explicit PostProcessExecution(std::vector<std::shared_ptr<Tensor>> &tensors, OpenCLBackend *mbackend);
        virtual ~PostProcessExecution() = default;
        PostProcessExecution(const PostProcessExecution &) = delete;
        PostProcessExecution &operator=(const PostProcessExecution &) = delete;
        virtual bool onOptimizedResize(std::vector<std::shared_ptr<Tensor>> &tensors) override;
        virtual bool onOptimizedExecute(std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors) override;

    private:
        OpenCLBackend *mbackend;

    private:
        float *bboxHostData, *paramHostData, *transHostData;
        int featWidth, featHeight;

    private:
        cl_mem inputCLData = NULL, outputCLData = NULL;

    private:
        std::vector<size_t> mGWS{1, 1, 1};
        std::vector<size_t> mLWS{1, 1, 1};
        std::string kernelName;
        cl_kernel mKernel;
        uint32_t mMaxWorkGroupSize;
        cl_mem mKernelBuffer;
        std::set<std::string> mBuildOptions;
    };
} // namespace SNN

#endif // POSTPROCESSEXECUTION_H