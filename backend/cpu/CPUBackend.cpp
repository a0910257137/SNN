#include "CPUBackend.h"
namespace SNN
{
    CPUBackend::CPUBackend(bool enable_fp16)
    {
        this->enable_fp16 = enable_fp16;
        mCPURuntime = new CPURuntime();
    }
    CPUBackend::~CPUBackend()
    {
        delete mCPURuntime;
    }
}