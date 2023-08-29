#ifndef MNNDefine_h
#define MNNDefine_h
#include <stdio.h>
#define STR_IMP(x) #x
#define STR(x) STR_IMP(x)
#define SNN_VERSION_MAJOR 1
#define SNN_VERSION_MINOR 0
#define SNN_VERSION_PATCH 0
#define SNN_VERSION STR(SNN_VERSION_MAJOR) "." STR(SNN_VERSION_MINOR) "." STR(SNN_VERSION_PATCH)
#define SNN_ERROR(format, ...) printf(format, ##__VA_ARGS__)
#define SNN_ASSERT(x)                                            \
    {                                                            \
        int res = (x);                                           \
        if (!res)                                                \
        {                                                        \
            SNN_ERROR("Error for %s, %d\n", __FILE__, __LINE__); \
        }                                                        \
    }

#define SNN_CHECK_SUCCESS(x, b)                                                       \
    if (x != b)                                                                       \
    {                                                                                 \
        fprintf(stderr, "Failed status within the file %s:%d\n", __FILE__, __LINE__); \
        exit(1);                                                                      \
    }
#endif