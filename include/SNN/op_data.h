#ifndef OP_DATA_H
#define OP_DATA_H
#include <stdint.h>

typedef enum
{
    kActNone = 0,
    kActRelu = 1,
    kActReluN1To1 = 2, // min(max(-1, x), 1)
    kActRelu6 = 3,     // min(max(0, x), 6)
    kActTanh = 4,
    kActSignBit = 5,
    kActSigmoid = 6,
} FusedActivation;

typedef enum
{
    kPaddingUnknown = 0,
    kPaddingSame = 1,
    kPaddingValid = 2,
} Padding;
#endif