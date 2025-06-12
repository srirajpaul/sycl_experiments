#!/bin/bash

export ZE_AFFINITY_MASK=0,2,4,6

is_write=1
if [ ! -z "$1" ]
then
    is_write=$1
fi

#SYCL program :
icpx -fsycl -O3 -o gpu_copy_kernel_sycl gpu_copy_kernel_sycl.cpp
#xelink 4 ranks
ZE_AFFINITY_MASK=0,2,4,6 ./gpu_copy_kernel_sycl 33554432 $is_write
#mdfi 2 ranks
ZE_AFFINITY_MASK=0,1 ./gpu_copy_kernel_sycl 33554432 $is_write 2

#OpenCL program :
ocloc compile -file kernel.cl -output kernel -output_no_suffix -device pvc -spv_only -q -options "-cl-std=CL2.0"
icpx -O3 -o gpu_copy_kernel_l0 gpu_copy_kernel_l0.cpp -lze_loader -lpthread
#xelink 4 ranks
ZE_AFFINITY_MASK=0,2,4,6 ./gpu_copy_kernel_l0 33554432 $is_write
#mdfi 2 ranks
ZE_AFFINITY_MASK=0,1 ./gpu_copy_kernel_l0 33554432 $is_write 2
