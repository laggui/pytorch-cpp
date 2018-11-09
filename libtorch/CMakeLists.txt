cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(vgg-predict)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(vgg-predict ${PROJECT_SOURCE_DIR}/predict.cpp)
target_link_libraries(vgg-predict "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
set_property(TARGET vgg-predict PROPERTY CXX_STANDARD 11)
