#ifndef HALIDERUNTIME_H
#define HALIDERUNTIME_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <vector>
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/opencl.h>
#include "op_data.h"
#ifdef __cplusplus
extern "C"
{
#endif

// Note that you should not use "inline" along with HALIDE_ALWAYS_INLINE;
// it is not necessary, and may produce warnings for some build configurations.
#ifdef _MSC_VER
#define HALIDE_ALWAYS_INLINE __forceinline
#define HALIDE_NEVER_INLINE __declspec(noinline)
#else
#define HALIDE_ALWAYS_INLINE __attribute__((always_inline)) inline
#define HALIDE_NEVER_INLINE __attribute__((noinline))
#endif
    // Forward-declare to suppress warnings if compiling as C.
    struct halide_buffer_t;
    struct halide_CL_buffer_t;

    /** Types in the halide type system. They can be ints, unsigned ints,
     * or floats (of various bit-widths), or a handle (which is always 64-bits).
     * Note that the int/uint/float values do not imply a specific bit width
     * (the bit width is expected to be encoded in a separate value).
     */
    typedef enum halide_type_code_t
    {
        halide_type_int = 0,   //!< signed integers
        halide_type_uint = 1,  //!< unsigned integers
        halide_type_float = 2, //!< floating point numbers
        halide_type_handle = 3 //!< opaque pointer type (void *)
    } halide_type_code_t;
    // Note that while __attribute__ can go before or after the declaration,
// __declspec apparently is only allowed before.
#ifndef HALIDE_ATTRIBUTE_ALIGN
#ifdef _MSC_VER
#define HALIDE_ATTRIBUTE_ALIGN(x) __declspec(align(x))
#else
#define HALIDE_ATTRIBUTE_ALIGN(x) __attribute__((aligned(x)))
#endif
#endif
    /** A runtime tag for a type in the halide type system. Can be ints,
     * unsigned ints, or floats of various bit-widths (the 'bits'
     * field). Can also be vectors of the same (by setting the 'lanes'
     * field to something larger than one). This struct should be
     * exactly 32-bits in size. */
    struct halide_type_t
    {
        /** The basic type code: signed integer, unsigned integer, or floating point. */
#if __cplusplus >= 201103L
        HALIDE_ATTRIBUTE_ALIGN(1)
        halide_type_code_t code; // halide_type_code_t
#else
    HALIDE_ATTRIBUTE_ALIGN(1)
    uint8_t code; // halide_type_code_t
#endif

        /** The number of bits of precision of a single scalar value of this type. */
        HALIDE_ATTRIBUTE_ALIGN(1)
        uint8_t bits;

        /** How many elements in a vector. This is 1 for scalar types. */
        HALIDE_ATTRIBUTE_ALIGN(2)
        uint16_t lanes;

#ifdef __cplusplus
        /** Construct a runtime representation of a Halide type from:
         * code: The fundamental type from an enum.
         * bits: The bit size of one element.
         * lanes: The number of vector elements in the type. */
        HALIDE_ALWAYS_INLINE halide_type_t(halide_type_code_t code, uint8_t bits, uint16_t lanes = 1)
            : code(code), bits(bits), lanes(lanes)
        {
        }

        /** Default constructor is required e.g. to declare halide_trace_event
         * instances. */
        HALIDE_ALWAYS_INLINE halide_type_t() : code((halide_type_code_t)0), bits(0), lanes(0) {}

        /** Compare two types for equality. */
        HALIDE_ALWAYS_INLINE bool operator==(const halide_type_t &other) const
        {
            return (code == other.code &&
                    bits == other.bits &&
                    lanes == other.lanes);
        }

        HALIDE_ALWAYS_INLINE bool operator!=(const halide_type_t &other) const
        {
            return !(*this == other);
        }

        /** Size in bytes for a single element, even if width is not 1, of this type. */
        HALIDE_ALWAYS_INLINE int bytes() const { return (bits + 7) / 8; }
#endif
    };

#ifdef __cplusplus
} // extern "C"
#endif
typedef enum
{
    halide_buffer_flag_host_dirty = 1,
    halide_buffer_flag_device_dirty = 2
} halide_buffer_flags;
typedef struct halide_CL_buffer_t
{
    cl_mem inputData = NULL;
    cl_mem mFilter = NULL, mBias = NULL;
} halide_CL_buffer_t;

/**
 * The raw representation of an image passed around by generated
 * Halide code. It includes some stuff to track whether the image is
 * not actually in main memory, but instead on a device (like a
 * GPU). For a more convenient C++ wrapper, use Halide::Buffer<T>. */
typedef struct halide_buffer_t
{
    /** A device-handle for e.g. GPU memory used to back this buffer. */
    uint64_t device;

    /** A pointer to the start of the data in main memory. In terms of
     * the Halide coordinate system, this is the address of the min
     * coordinates (defined below). */
    std::vector<uint8_t> hostPtr;

    /** flags with various meanings. */
    uint64_t flags;

    /** The type of each buffer element. */
    struct halide_type_t type;

    /** The dimensionality of the buffer. */
    int32_t dimensions;
    /**The kernel stride*/
    std::vector<uint8_t> strides{0, 0};
    /**The kernel dilation*/
    std::vector<uint8_t> dilations{0, 0};
    /**The kernel weight shape*/
    std::vector<uint8_t> kernelShapes{0, 0, 0, 0};
    /**The kernel bias shape*/
    std::vector<uint8_t> biasShapes{0};
    /**
     * The kernel weight and bias data bytes
     */
    uint32_t weightBytes, biasBytes;
    /**
     * Input data shape
     */
    uint32_t inputShapes[4] = {};
    /**
     * Pass to kernel the output shape is ?
     */
    uint32_t outputShapes[4] = {};
    /**
     * Hint mBuffer belong which op type for different operation
     */
    int opType;
    /**
     * Hint mBuffer belong which fused activation type
     */
    int actType;
    /**
     * Pads the buffer up to a multiple of 8 bytes
     */
    int paddingType;

    /** Support opencl buffer*/
    halide_CL_buffer_t mDeviceBuffer;
} halide_buffer_t;
#ifdef __cplusplus
namespace
{
    template <typename T>
    struct check_is_pointer;
    template <typename T>
    struct check_is_pointer<T *>
    {
    };
}

/** Construct the halide equivalent of a C type */
template <typename T>
HALIDE_ALWAYS_INLINE halide_type_t halide_type_of()
{
    // Create a compile-time error if T is not a pointer (without
    // using any includes - this code goes into the runtime).
    check_is_pointer<T> check;
    (void)check;
    return halide_type_t(halide_type_handle, 64);
}
template <>
HALIDE_ALWAYS_INLINE halide_type_t halide_type_of<float>()
{
    return halide_type_t(halide_type_float, 32);
}

template <>
HALIDE_ALWAYS_INLINE halide_type_t halide_type_of<double>()
{
    return halide_type_t(halide_type_float, 64);
}

template <>
HALIDE_ALWAYS_INLINE halide_type_t halide_type_of<bool>()
{
    return halide_type_t(halide_type_uint, 1);
}

template <>
HALIDE_ALWAYS_INLINE halide_type_t halide_type_of<uint8_t>()
{
    return halide_type_t(halide_type_uint, 8);
}

template <>
HALIDE_ALWAYS_INLINE halide_type_t halide_type_of<uint16_t>()
{
    return halide_type_t(halide_type_uint, 16);
}

template <>
HALIDE_ALWAYS_INLINE halide_type_t halide_type_of<uint32_t>()
{
    return halide_type_t(halide_type_uint, 32);
}

template <>
HALIDE_ALWAYS_INLINE halide_type_t halide_type_of<uint64_t>()
{
    return halide_type_t(halide_type_uint, 64);
}

template <>
HALIDE_ALWAYS_INLINE halide_type_t halide_type_of<int8_t>()
{
    return halide_type_t(halide_type_int, 8);
}

template <>
HALIDE_ALWAYS_INLINE halide_type_t halide_type_of<int16_t>()
{
    return halide_type_t(halide_type_int, 16);
}

template <>
HALIDE_ALWAYS_INLINE halide_type_t halide_type_of<int32_t>()
{
    return halide_type_t(halide_type_int, 32);
}

template <>
HALIDE_ALWAYS_INLINE halide_type_t halide_type_of<int64_t>()
{
    return halide_type_t(halide_type_int, 64);
}
#endif
#endif