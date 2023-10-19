#ifndef ADDN_H
#define ADDN_H
#include "Execution.h"
namespace SNN
{
    class AddN : public Execution
    {
    public:
        AddN(std::shared_ptr<Tensor> tensor, OpenCLBackend *mbackend);
        virtual ~AddN() = default;
        virtual bool onResize(std::shared_ptr<Tensor> tensor);
        virtual bool onExecute();

    private:
        OpenCLBackend *mOpenCLBackend;

    private:
        cl_kernel mKernel;
        std::vector<size_t> mLWS{0, 0, 0, 0};
        std::vector<size_t> mGWS{0, 0, 0, 0};
        size_t mMaxWorkGroupSize;
    };
} // namespace SNN

#endif // ADDN_H