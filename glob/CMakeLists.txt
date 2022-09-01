cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project("glob" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS}")

# GLOB
set(INC ${INC} ${CMAKE_CURRENT_SOURCE_DIR})
set(SRC ${SRC} c_glob.cpp)

# Target
add_library(${PROJECT_NAME} SHARED ${SRC})
target_include_directories(${PROJECT_NAME} PUBLIC ${INC})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIB})
target_compile_definitions(${PROJECT_NAME} PUBLIC GLOB_VERSION=1)

# Install
install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})
