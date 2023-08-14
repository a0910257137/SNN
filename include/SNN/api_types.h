#ifndef SNN_API_TYPES_H_
#define SNN_API_TYPES_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

// Define TFL_CAPI_EXPORT macro to export a function properly with a shared
// library.
#ifdef SWIG
#define TFL_CAPI_EXPORT
#elif defined(TFL_STATIC_LIBRARY_BUILD)
#define TFL_CAPI_EXPORT
#else // not definded TFL_STATIC_LIBRARY_BUILD
#if defined(_WIN32)
#ifdef TFL_COMPILE_LIBRARY
#define TFL_CAPI_EXPORT __declspec(dllexport)
#else
#define TFL_CAPI_EXPORT __declspec(dllimport)
#endif // TFL_COMPILE_LIBRARY
#else
#define TFL_CAPI_EXPORT __attribute__((visibility("default")))
#endif // _WIN32
#endif // SWIG

    // Note that new error status values may be added in future in order to
    // indicate more fine-grained internal states, therefore, applications should
    // not rely on status values being members of the enum.
    typedef enum SNNStatus
    {
        kSNNOk = 0,

        // Generally referring to an error in the runtime (i.e. interpreter)
        kSNNError = 1,

        // Generally referring to an error from a SNNDelegate itself.
        kSNNDelegateError = 2,

        // Generally referring to an error in applying a delegate due to
        // incompatibility between runtime and delegate, e.g., this error is returned
        // when trying to apply a TF Lite delegate onto a model graph that's already
        // immutable.
        kSNNApplicationError = 3,

        // Generally referring to serialized delegate data not being found.
        // See SNN::delegates::Serialization.
        kSNNDelegateDataNotFound = 4,

        // Generally referring to data-writing issues in delegate serialization.
        // See SNN::delegates::Serialization.
        kSNNDelegateDataWriteError = 5,

        // Generally referring to data-reading issues in delegate serialization.
        // See SNN::delegates::Serialization.
        kSNNDelegateDataReadError = 6,

        // Generally referring to issues when the TF Lite model has ops that cannot be
        // resolved at runtime. This could happen when the specific op is not
        // registered or built with the TF Lite framework.
        kSNNUnresolvedOps = 7,
    } SNNStatus;

    // Types supported by tensor

#ifdef __cplusplus
} // extern C
#endif
#endif // SNN_API_TYPES_H_
