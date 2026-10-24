cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_ARCHITECTURES "86")

project(hairProject LANGUAGES CXX CUDA VERSION 0.1.0)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

file(GLOB SOURCES HairRemoval/*.cpp HairRemoval/*.cu)
add_executable(hairRemoval ${SOURCES})

set_target_properties(hairRemoval PROPERTIES
    CXX_STANDARD 17
)

target_compile_definitions(hairRemoval PRIVATE
    SAMPLE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/sample"
)

# Using AVX, AVX2 intrinsic set
target_compile_options(hairRemoval PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-mavx -mavx2>
)

target_link_libraries(hairRemoval PRIVATE
    ${OpenCV_LIBS}
    CUDA::cudart
    CUDA::cufft
    OpenMP::OpenMP_CXX
)

target_include_directories(hairRemoval PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}/cccl
)
