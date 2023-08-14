
file(GLOB_RECURSE CPU_SRC                   
                ${CMAKE_CURRENT_LIST_DIR}/core/CPU*.cpp
                ${CMAKE_CURRENT_LIST_DIR}/CPU*.cpp)
set(CPU_BACKEND_DIR "${CMAKE_SOURCE_DIR}/backend")
add_library(CPU_LIBS SHARED ${CPU_SRC})

target_include_directories(CPU_LIBS  
      PUBLIC
            ${CMAKE_SOURCE_DIR})