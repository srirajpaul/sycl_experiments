all: gpu_copy_kernel_sycl gpu_copy_kernel_l0

gpu_copy_kernel_sycl: gpu_copy_kernel_sycl.cpp
	icpx -fsycl -O3 -o gpu_copy_kernel_sycl gpu_copy_kernel_sycl.cpp


gpu_copy_kernel_l0: gpu_copy_kernel_l0.cpp  kernel.spv
	icpx -O3 -o gpu_copy_kernel_l0 gpu_copy_kernel_l0.cpp -lze_loader -lpthread

kernel.spv: kernel.cl
	ocloc compile -file kernel.cl -output kernel -output_no_suffix -device pvc -spv_only -q -options "-cl-std=CL2.0"
