# CMakeList.txt : CMake project for MNIST, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("MNIST")

set(HEADERS
    "stb_image_write.h"
    "load_mnist_dataset.h"
    "Network.h")

set(SRC
    "load_mnist_dataset.cpp"
    "main.cpp"
    "Network.cpp")

# Add source to this project's executable.
add_executable (MNIST_run ${SRC} ${HEADERS})

find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(MNIST_run PRIVATE Eigen3::Eigen)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/train-labels.idx1-ubyte" "${CMAKE_CURRENT_BINARY_DIR}/train-labels.idx1-ubyte" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/train-images.idx3-ubyte" "${CMAKE_CURRENT_BINARY_DIR}/train-images.idx3-ubyte" COPYONLY)