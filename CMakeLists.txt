cmake_minimum_required(VERSION 3.10)
project(sfm)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(include)

add_executable(sfm src/main.cpp src/utils.cpp src/feature_extractor.cpp)

target_include_directories(sfm PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(sfm ${OpenCV_LIBS})