#if __aarch64__
#include <sys/sysctl.h>
#endif
#include "CPURuntime.h"
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <vector>

#if defined(__linux__) && defined(__aarch64__)
#include <sys/auxv.h>

#define CPUINFO_ARM_LINUX_FEATURE_FPHP UINT32_C(0x00000200)
#define CPUINFO_ARM_LINUX_FEATURE_ASIMDHP UINT32_C(0x00000400)
#define CPUINFO_ARM_LINUX_FEATURE_ASIMDDP UINT32_C(0x00100000)
#define CPUINFO_ARM_LINUX_FEATURE_I8MM UINT32_C(0x00002000)
#define CPUINFO_ARM_LINUX_FEATURE_SVE UINT32_C(0x00400000)
#define CPUINFO_ARM_LINUX_FEATURE_SVE2 UINT32_C(0x00000002)

#endif /* __linux__ && __aarch64__ */
namespace SNN
{
    CPURuntime::CPURuntime()
    {
    }
    CPURuntime::~CPURuntime()
    {
    }
}