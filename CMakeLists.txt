cmake_minimum_required(VERSION 3.10)
project(sfm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)

include_directories(include)

add_executable(
    sfm src/main.cpp 
    src/camera.cpp 
    src/featurematching.cpp 
    src/epipolargeometry.cpp)

target_link_libraries(sfm ${OpenCV_LIBS})