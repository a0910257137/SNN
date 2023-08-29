#ifndef BASEPROTOCOL_H
#define BASEPROTOCOL_H
namespace SNN
{
    class BaseProtocol
    {
    public:
        BaseProtocol() = default;
        ~BaseProtocol() = default;
        BaseProtocol(const BaseProtocol &) = delete;
        BaseProtocol &operator=(const BaseProtocol &) = delete;
        BaseProtocol &operator=(const BaseProtocol &&) = delete;
    };
}
#endif /* BASEPROTOCOL_H */
