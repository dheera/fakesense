cmake_minimum_required(VERSION 3.10)
project(FakeSense)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

add_executable(fakesense main.cpp)
target_link_libraries(fakesense "${OpenCV_LIBS}" "${TORCH_LIBRARIES}")
set_property(TARGET fakesense PROPERTY CXX_STANDARD 14)
