
file(GLOB SRC_LIST ${CMAKE_CURRENT_LIST_DIR}/Backend.cpp)
add_library(Backend_LIBS SHARED ${SRC_LIST} )
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/cpu)
target_link_libraries(
    Backend_LIBS
    PRIVATE 
            CPU_LIBS)
target_include_directories(Backend_LIBS
        PRIVATE
            ${OPENCL_INCLUDE_DIR}      
        PUBLIC
            ${CMAKE_SOURCE_DIR})