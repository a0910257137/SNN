
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
message(STATUS "\tOpenCL: ${SNN_OPENCL}")
# OpenCL
IF(SNN_OPENCL)
  set(OPENCL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/OpenCL")
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/backend/opencl)
ENDIF()
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
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

target_link_libraries(infer.exe 
                    PRIVATE 
                        ${OpenCV_LIBS}
                        ${Threads_LIBS}
                        tensorflow-lite
                        CORE_LIBS
                        UTIL_LIBS)
                        