#include "OpenCLBackend.h"

namespace SNN
{
    OpenCLBackend::OpenCLBackend(bool enable_half)
    {
        this->enable_half = enable_half;
        mCLRuntime = new OpenCLRuntime();
    }
    OpenCLBackend::~OpenCLBackend() {}
}