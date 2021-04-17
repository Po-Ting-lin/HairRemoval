cmake_minimum_required(VERSION 3.0.0)
project(hair LANGUAGES CXX CUDA VERSION 0.1.0)

include(CTest)
enable_testing()

# Using AVX, AVX intrinsic set
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")

add_library(hairRemovalLib STATIC
  HairRemoval/cuda_error.cuh
  HairRemoval/entropyThreshold.cu
  HairRemoval/entropyThreshold.cuh
  HairRemoval/hairRemoval.cu
  HairRemoval/hairRemoval.cuh
  HairRemoval/hairRemovalEngine.cu
  HairRemoval/hairRemovalEngine.cuh
  HairRemoval/hairRemovalStruct.h
  HairRemoval/timer.cuh
  HairRemoval/parameters.h
  HairRemoval/Source.cpp
  HairRemoval/utils.h
  )

target_compile_features(hairRemovalLib PUBLIC cxx_std_17)
add_executable(hairR HairRemoval/Source.cpp)

# Using Cufft
include_directories("/usr/local/cuda/include")
# link_directories("/usr/local/cuda/lib")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lcufft")

# Using OpenCV Lib
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( hairR ${OpenCV_LIBS} ${CUDA_CUFFT_LIBRARIES} hairRemovalLib cufft)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
