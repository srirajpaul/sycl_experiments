
set(CMAKE_CXX_COMPILER icpx)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")

add_executable(gpu_comm_atomics_kernel gpu_comm_atomics_kernel.cpp)
add_executable(gpu_comm_write_flush gpu_comm_write_flush.cpp)

