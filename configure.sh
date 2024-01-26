#! /bin/sh

cmake -DSNN_OPENCL=ON  -DSNN_8368_P_AARCH64=OFF . -B ./build

# GEMINI_TOOLCHAIN="/Gemini"
# ARMCC_PREFIX="${GEMINI_TOOLCHAIN}/build/tools/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-"
# PREBUILT="${GEMINI_TOOLCHAIN}/linux/sdk/prebuilt/common/usr/local"
# LOCAL="/usr/local"
# cmake \
#     -DCMAKE_C_COMPILER="${ARMCC_PREFIX}gcc" \
#     -DCMAKE_CXX_COMPILER="${ARMCC_PREFIX}g++" \
#     -DCMAKE_CXX_FLAGS=" -I${PREBUILT}/include/opencv4 -L${LOCAL}/lib/opencv4" \
#     -DSNN_OPENCL=ON \
#     -DSNN_8368_P_AARCH64=ON \
#     -DCMAKE_SYSTEM_NAME=Linux \
#     -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
#     -DCMAKE_BUILD_TYPE=Debug \
#     . -B ./build