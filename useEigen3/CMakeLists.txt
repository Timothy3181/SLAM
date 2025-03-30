cmake_minimum_required(VERSION 3.20)
project(useEigen3)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(main ${PROJECT_SOURCE_DIR}/useEigen3.cpp)

target_link_libraries(main ${OpenCV_LIBS} Eigen3::Eigen)
