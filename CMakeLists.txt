cmake_minimum_required(VERSION 3.10.2)

project(vizgard LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" ${CMAKE_MODULE_PATH})
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA
find_package(CUDA REQUIRED)
set(LIB ${LIB} ${CUDA_LIBRARIES})
set(INC ${INC} ${CUDA_INCLUDE_DIRS})

# OpenCV
# set(OpenCV_DIR /usr/local/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
set(INC ${INC} ${OpenCV_INCLUDE_DIRS})
set(LIB ${LIB} ${OpenCV_LIBS})

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

# ...
add_subdirectory(TrtExec)
add_subdirectory(EasyOCR)
add_subdirectory(glob)
set(LIB ${LIB} TrtExec EasyOCR glob)
set(INC ${INC} ${PROJECT_SOURCE_DIR}/vizgard)

message(STATUS "Source directories:  ${SRC}")
message(STATUS "Include directories: ${INC}")
message(STATUS "Source directories:  ${LIB}")

# Add sources
add_executable(${PROJECT_NAME} main.cpp)
target_sources(${PROJECT_NAME} PRIVATE ${SRC})
include_directories(${INC})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIB})
