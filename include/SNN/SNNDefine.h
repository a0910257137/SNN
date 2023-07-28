#ifndef MNNDefine_h
#define MNNDefine_h
#include <stdio.h>
#define STR_IMP(x) #x
#define STR(x) STR_IMP(x)
#define SNN_VERSION_MAJOR 1
#define SNN_VERSION_MINOR 0
#define SNN_VERSION_PATCH 0
#define SNN_VERSION STR(SNN_VERSION_MAJOR) "." STR(SNN_VERSION_MINOR) "." STR(SNN_VERSION_PATCH)
#endif