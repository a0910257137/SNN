
file(GLOB SRC_LIST ${CMAKE_CURRENT_LIST_DIR}/Backend.cpp)
add_library(Backend_LIBS SHARED ${SRC_LIST} )
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/cpu)
# message(STATUS "HERE")
# message(STATUS ${OCL_LIBS})
# set(SNN_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include/SNN")
# message(STATUS "${CMAKE_SOURCE_DIR}/backend/opencl/execution/image/cl/image")
set(openclBackend "${CMAKE_SOURCE_DIR}/backend/opencl/execution/image/cl")
message(STATUS ${openclBackend})
target_link_libraries(
    Backend_LIBS
    PRIVATE 
            CPU_LIBS
            ${OpenCV_LIBS})
target_include_directories(Backend_LIBS
        PRIVATE
            ${OPENCL_INCLUDE_DIR}      
            # ${openclBackend}
        PUBLIC
            ${CMAKE_SOURCE_DIR})