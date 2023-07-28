
file(GLOB_RECURSE OpenCL_SRC 
                  ${CMAKE_CURRENT_LIST_DIR}/core/OpenCL*.cpp
                  ${CMAKE_CURRENT_LIST_DIR}/core/runtime/OpenCL*.cpp
                  ${CMAKE_CURRENT_LIST_DIR}/utils/*.cpp)
set(OPENCL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/OpenCL")
set(OPENCL_LIB_DIR "${CMAKE_SOURCE_DIR}/third_party/OpenCL")
set(OPENCL_BACKEND_DIR "${CMAKE_SOURCE_DIR}/backend")
# message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")

add_definitions(-DCL_TARGET_OPENCL_VERSION=300)
# by using cuda toolkit
# find_package(CUDA 11.0 REQUIRED)
# find_path(
#   OpenCL_INCLUDE_DIR CL/opencl.h
#   HINTS "${CUDA_TOOLKIT_ROOT_DIR}/include"
#   NO_DEFAULT_PATH
# )
# if (CMAKE_SIZEOF_VOID_P EQUAL 8)
#   set(opencl_lib64 "${CUDA_TOOLKIT_ROOT_DIR}/lib64" )
# endif()
# find_library(
#   OpenCL_LIBRARY OpenCL
#   HINTS ${opencl_lib64} "${CUDA_TOOLKIT_ROOT_DIR}/lib"
#   PATH_SUFFIXES "${CMAKE_LIBRARY_ARCHITECTURE}"
#   NO_DEFAULT_PATH
# )
# link_directories(${opencl_lib64})
# find_package(OpenCL REQUIRED)
message(STATUS "Get opencl library from local third part" ${OPENCL_INCLUDE_DIR})
# message(${opencl_lib64})
add_library(OCL_LIBS SHARED ${OpenCL_SRC})
target_link_libraries(OCL_LIBS
    PRIVATE 
        OpenCL)
target_include_directories(OCL_LIBS  
      PUBLIC
            ${OPENCL_INCLUDE_DIR}
            ${CMAKE_SOURCE_DIR})