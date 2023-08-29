#include "OpenCLBackend.h"

namespace SNN
{
    OpenCLBackend::OpenCLBackend(bool permitFloat16)
    {
        this->permitFloat16 = permitFloat16;
        _mCLRuntime = new OpenCLRuntime();
    }
    OpenCLBackend::~OpenCLBackend()
    {
        delete _mCLRuntime;
    }

}