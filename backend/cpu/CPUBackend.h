#ifndef CPUBACKEND_H
#define CPUBACKEND_H
#include "backend/cpu/core/CPURuntime.h"
namespace SNN
{
    // #ifdef MNN_SUPPORT_BF16 // checkout for different platform
    class CPUBackend
    {
    public:
        CPUBackend(bool enable_fp16);
        ~CPUBackend();
        CPUBackend(const CPUBackend &) = delete;
        CPUBackend &operator=(const CPUBackend &) = delete;

    private:
        CPURuntime *mCPURuntime = nullptr;
        bool permitFloat16;
    };
}
#endif // CPUBACKEND_H