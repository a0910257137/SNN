#ifndef CPURUNTIME_H
#define CPURUNTIME_H
namespace SNN
{
    class CPURuntime
    {
    public:
        CPURuntime();
        ~CPURuntime();
        CPURuntime(const CPURuntime &) = delete;
        CPURuntime &operator=(const CPURuntime &) = delete;
    };
}
#endif // CPURUNTIME_H