cmake_minimum_required(VERSION 3.0.0)
project(hairProject LANGUAGES CXX CUDA VERSION 0.1.0)

include(CTest)
enable_testing()


set(CUFFT_INCLUDE_DIRS "/usr/local/cuda/include")

# Using AVX, AVX intrinsic set
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")

file(GLOB hairRemovalLib STATIC
  "HairRemoval/*.h"
  "HairRemoval/*.cpp"
  "HairRemoval/*.cuh"
  "HairRemoval/*.cu"
)

add_executable(hairRemoval ${hairRemovalLib})

find_package(CUDA REQUIRED)
if (CUDA_FOUND)
  message("Find CUDA")
  include_directories(${CUFFT_INCLUDE_DIRS})
  # link_directories("/usr/local/cuda/lib")
  # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lcufft")
else(CUDA_FOUND)
  message("CUDA is not installed on this system.")
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    message("Find OpenMp")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

target_link_libraries( hairRemoval ${OpenCV_LIBS} cufft)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
