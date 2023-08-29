
file(GLOB SRC_LIST ${CMAKE_CURRENT_LIST_DIR}/NodeFactory.cpp)
add_library(NodeFactory_LIBS SHARED ${SRC_LIST} )
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/cpu)
target_link_libraries(NodeFactory_LIBS
    PRIVATE 
        CPU_LIBS)
target_include_directories(NodeFactory_LIBS
        PRIVATE
            ${OPENCL_INCLUDE_DIR}      
        PUBLIC
            ${CMAKE_SOURCE_DIR})