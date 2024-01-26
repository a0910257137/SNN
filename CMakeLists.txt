
cmake_minimum_required(VERSION 3.0)
message(DEPRECATION ${PROJECT_SOURCE_DIR})
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/include/SNN/SNNDefine.h" SNN_DEFINE)

string(REGEX MATCH "SNN_VERSION_MAJOR [0-9]+" SNN_VERSION_MAJOR_DEFINE ${SNN_DEFINE})
string(REGEX MATCH "[0-9]+" SNN_VERSION_MAJOR ${SNN_VERSION_MAJOR_DEFINE})
string(REGEX MATCH "SNN_VERSION_MINOR [0-9]+" SNN_VERSION_MINOR_DEFINE ${SNN_DEFINE})
string(REGEX MATCH "[0-9]+" SNN_VERSION_MINOR ${SNN_VERSION_MINOR_DEFINE})
string(REGEX MATCH "SNN_VERSION_PATCH [0-9]+" SNN_VERSION_PATCH_DEFINE ${SNN_DEFINE})
string(REGEX MATCH "[0-9]+" SNN_VERSION_PATCH ${SNN_VERSION_PATCH_DEFINE})
set(SNN_VERSION ${SNN_VERSION_MAJOR}.${SNN_VERSION_MINOR}.${SNN_VERSION_PATCH})
message(STATUS "SUNPLUS NEURAL NETWORK VERSION: " ${SNN_VERSION})

project(SNN VERSION ${SNN_VERSION} LANGUAGES C CXX)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set (CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${CMAKE_CURRENT_LIST_DIR}/cmake"
)

set(SNN_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include/SNN")
set(THIRD_PARTY_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include/third_party")
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
file(GLOB SRC_LIST
            "${PROJECT_SOURCE_DIR}/tools/inference.cpp"
) 
# backend options
option(SNN_OPENCL "Enable OpenCL" OFF)
option(SNN_8368_P_AARCH64 "Compile arm board" OFF)
message(STATUS "\tOpenCL: ${SNN_OPENCL}")
message(STATUS "\tARCH64: ${SNN_8368_P_AARCH64}")

IF(SNN_8368_P_AARCH64)
  set(GEMINI_TOOLCHAIN "/Gemini")
  set(PREBUILT "${GEMINI_TOOLCHAIN}/linux/sdk/prebuilt/common/usr/local")
  set(LOCAL "/usr/local")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -funsafe-math-optimizations -I${PREBUILT}/include -L${PREBUILT}/lib")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -funsafe-math-optimizations -I${PREBUILT}/include -L${PREBUILT}/lib")
  message(STATUS "\tThe 8368-P toolchain path at " ${PREBUILT})
  # OpenCL
  IF(SNN_OPENCL)
    set(OPENCL_INCLUDE_DIR "${PREBUILT}/include/CL")
    set(OPENCL_LIBRARY "${PREBUILT}/lib")
  ENDIF()
ELSE()
# OpenCL
  IF(SNN_OPENCL)
    set(OPENCL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/OpenCL")
  ENDIF()
ENDIF()
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/backend/opencl)
find_package(OpenCV REQUIRED)
IF(OpenCV_FOUND)
    # find_package(OpenCV REQUIRED)
    message(STATUS "The Opecv lib is found!") 
ELSE()
    message(STATUS "\tManually set opencv path!")     
    set(OpenCV_INCLUDE_DIRS  "${PREBUILT}/include/opencv4")
    set(OpenCV_LIBRARIES_DIRS "${LOCAL}/lib/opencv4")
    file(GLOB OpenCV_LIBS "${LOCAL}/lib/opencv4/*.so")
    IF((NOT EXISTS ${OpenCV_INCLUDE_DIRS}) OR (NOT EXISTS ${OpenCV_LIBRARIES_DIRS}))
      message(STATUS "\tManually set failed!!!")     
      RETURN()
    ENDIF()
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIBRARIES_DIRS})
ENDIF()
find_package(Threads REQUIRED)
link_directories(
                ${GOOGLE_INCLUDE_DIR}
                ${SNN_INCLUDE_DIR})

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/misc)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/core)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/backend)
add_executable(infer.exe ${SRC_LIST})
target_include_directories(infer.exe  
                            PRIVATE 
                              ${SNN_INCLUDE_DIR}
                              ${OPENCL_INCLUDE_DIR}
                            PUBLIC  
                              ${CMAKE_SOURCE_DIR})
IF(SNN_8368_P_AARCH64)
  target_link_libraries(infer.exe 
                        PRIVATE 
                            ${OpenCV_LIBS}
                            ${Threads_LIBS}
                            tensorflow-lite
                            CORE_LIBS
                            UTIL_LIBS
                            OpenCL GAL 
                            VSC CLC 
                            EGL GLESv2 
                            SPIRV_viv 
                            wayland-server wayland-client 
                            drm gbm 
                            gbm_viv ffi
                            pthread)
ELSE()
  target_link_libraries(infer.exe 
                      PRIVATE 
                          ${OpenCV_LIBS}
                          ${Threads_LIBS}
                          tensorflow-lite
                          CORE_LIBS
                          UTIL_LIBS
                          pthread)
ENDIF()