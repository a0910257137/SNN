
set(TENSORFLOW_SOURCE_DIR "/home2/anders/proj_c/side_proj/tensorflow")
link_directories(
                ${TENSORFLOW_SOURCE_DIR})
add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
  EXCLUDE_FROM_ALL
)
add_library(CORE_LIBS SHARED 
                Tensor.cpp
                Interpreter.cpp
                Pipeline.cpp
                ConvolutionCommon.cpp)
target_link_libraries(CORE_LIBS
      PRIVATE 
          tensorflow-lite
          UTIL_LIBS
          OCL_LIBS
          NodeFactory_LIBS)
# message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
# message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
# message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
target_include_directories(CORE_LIBS
      PUBLIC
          ${CMAKE_SOURCE_DIR})
      