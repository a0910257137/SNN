add_library(UTIL_LIBS SHARED utils.cpp utils.h nms.cpp nms.h)


target_link_libraries(UTIL_LIBS
                    PUBLIC 
                        ${OpenCV_LIBS})