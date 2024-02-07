
set(TENSORFLOW_SOURCE_DIR "/aidata/anders/thirdy_party/tensorflow")
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
                ConvolutionCommon.cpp
                Model.cpp
                PostProcessor.cpp
                TensorUtils.cpp
                )
target_link_libraries(CORE_LIBS
      PRIVATE 
          tensorflow-lite
          UTIL_LIBS
          OCL_LIBS
          Backend_LIBS
          ${OpenCV_LIBS})
target_include_directories(CORE_LIBS
      PUBLIC
          ${CMAKE_SOURCE_DIR})
      