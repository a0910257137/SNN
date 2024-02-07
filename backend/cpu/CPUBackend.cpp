#include "CPUBackend.h"
namespace SNN
{
    CPUBackend::CPUBackend(bool enable_fp16)
    {
        this->permitFloat16 = enable_fp16;
        mCPURuntime = new CPURuntime();
    }
    CPUBackend::~CPUBackend()
    {
        delete mCPURuntime;
    }
}