
file(GLOB SRC_LIST ${CMAKE_CURRENT_LIST_DIR}/BackendFactory.cpp)
add_library(BackendFactory_LIBS SHARED ${SRC_LIST} )
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/cpu)
target_link_libraries(BackendFactory_LIBS
    PRIVATE 
        CPU_LIBS)
target_include_directories(BackendFactory_LIBS
        PRIVATE
            ${OPENCL_INCLUDE_DIR}      
        PUBLIC
            ${CMAKE_SOURCE_DIR})