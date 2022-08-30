cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project("TrtExec" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS}")

# CUDA
find_package(CUDA REQUIRED)
set(LIB ${LIB} ${CUDA_LIBRARIES})
set(INC ${INC} ${CUDA_INCLUDE_DIRS})

# OpenCV
find_package(OpenCV REQUIRED)
set(LIB ${LIB} ${OpenCV_LIBRARIES})
set(INC ${INC} ${OpenCV_INCLUDE_DIRS})

# TensorRT
# find_package(TensorRT REQUIRED)
# set(LIB ${LIB} ${TensorRT_LIBRARIES})
# set(INC ${INC} ${TENSORRT_INCLUDE_DIR})
set(TENSORRT_ROOT /usr/local/TensorRT)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} PATH_SUFFIXES include/)
find_library(TENSORRT_LIBRARY_INFER nvinfer
    HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
    HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_ONNXPARSER})
set(LIB ${LIB} ${TENSORRT_LIBRARY})
set(INC ${INC} ${TENSORRT_INCLUDE_DIR})

# # TensorRT executable file
# set(SRC ${SRC} Trtexec.cpp)

# # main.cpp
# set(SRC ${SRC} main.cpp)
set(INC ${INC} ${CMAKE_CURRENT_SOURCE_DIR})
set(INC ${INC} ${PROJECT_SOURCE_DIR}/vizgard)

# Target
add_executable("TrtExec-bin" main.cpp Trtexec.cpp)
target_include_directories("TrtExec-bin" PUBLIC ${INC})
target_link_libraries("TrtExec-bin" PRIVATE ${LIB})

add_library(${PROJECT_NAME} SHARED Trtexec.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC ${INC})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIB})
target_compile_definitions(${PROJECT_NAME} PUBLIC TRTEXEC_VERSION=1)

# Install
install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})
