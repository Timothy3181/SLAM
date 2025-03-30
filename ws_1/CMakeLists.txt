cmake_minimum_required(VERSION 3.20)
project(demo)

find_package(OpenCV REQUIRED)

add_executable(main ${PROJECT_SOURCE_DIR}/1.cpp)

target_link_libraries(main ${OpenCV_LIBS})
