
/*
    On a multi GPU system, allocate data on each GPU and copy from one to other.
    This data transfer will make use of Xelink.
*/

#include<sycl/sycl.hpp>
#include<iostream>
#include<chrono>

int main(int argc, char **argv) {

    //This will list the GPU device twice, i.e. for OpenCL and Level Zero
    //for (auto device : sycl::device::get_devices(sycl::info::device_type::gpu)) {
    //    std::cout << "  Device: " << device.get_info<sycl::info::device::name>()
    //              << std::endl;
    //}

    //find all the GPUS using level zero platform
    //https://sycl.readthedocs.io/en/latest/iface/platform.html#platform-example
    auto platforms = sycl::platform::get_platforms();
    std::vector<sycl::device> devices;

    for (auto &platform : platforms) {
        auto backend = platform.get_backend();

        //if(backend == sycl::backend::ext_oneapi_level_zero) {
        //    std::cout << "Backend: " << b << std::endl;
        //}

        //std::cout << "Platform: " << platform.get_info<sycl::info::platform::name>()
        //      << std::endl;

        if(backend == sycl::backend::ext_oneapi_level_zero) {
            auto devices_tmp = platform.get_devices();
            for (auto &device : devices_tmp) {
                //std::cout << "  Device: " << device.get_info<sycl::info::device::name>()
                //    << std::endl;
                devices.push_back(device);
            }
        }
    }

    constexpr int dev_count = 4;

    size_t N = 32*1024*1024;
    int write = 1;

    if (argc > 1) {
        N = atol(argv[1]);
    }

    if (argc > 2) {
        write = atoi(argv[2]);
    }
    int num_devices = 4;
    if (argc > 3) {
        num_devices = atoi(argv[3]);
    }
    int use_copy = 0;
    if (argc > 4) {
        use_copy = atoi(argv[4]);
    }

    std::vector<sycl::queue> q_vec;
    for (int i=0; i<num_devices; i++) {
        q_vec.emplace_back(sycl::queue(devices[i]));
    }

    assert(devices.size() >= num_devices);

    assert(sizeof(long) == 8);
    long *out_buffs[dev_count], *in_buffs[dev_count];
    for (int i=0; i<num_devices; i++) {
        in_buffs[i] = sycl::malloc_device<long>(N,q_vec[i]);
        out_buffs[i] = sycl::malloc_device<long>(N * num_devices,q_vec[i]);
        q_vec[i].submit([&](sycl::handler &h) {
            h.parallel_for(N, [=](sycl::id<1> idx) {
                in_buffs[i][idx] = 23;
                for (int j = 0; j < num_devices; j++) {
                    out_buffs[i][N * j + idx] = 24;
                }
            });
        }).wait();
    }

    const int rep = 10;

    std::chrono::time_point<std::chrono::high_resolution_clock> start;

    for(int n=0;n<rep+5;n++) {

        if (n == 5) {
            start = std::chrono::high_resolution_clock::now();
        }

        std::vector<sycl::event> e_vec;
        //constexpr int work_group_size = 32;

        for (int i=0; i<num_devices; i++) {
            sycl::event e;
            if (!use_copy) {
                e = q_vec[i].submit([=](sycl::handler &h) {
                    h.parallel_for(N, [=](sycl::id<1> idx) {
                    //h.parallel_for(sycl::nd_range(sycl::range{N}, sycl::range{work_group_size}), [=](sycl::nd_item<1> it) {
                        //const size_t idx = it.get_global_linear_id();
                        if (write) {
                            long data = in_buffs[i][idx];
                            #pragma unroll
                            for (int j = 0; j < num_devices; j++) {
                                out_buffs[j][i * N + idx] = data;
                            }
                            /*
                            out_buffs[0][i * N + idx] = data;
                            out_buffs[1][i * N + idx] = data;
                            out_buffs[2][i * N + idx] = data;
                            out_buffs[3][i * N + idx] = data;
                            */
                        }
                        else {
                            #pragma unroll
                            for (int j = 0; j < num_devices; j++) {
                                out_buffs[i][j * N + idx] = in_buffs[j][idx];
                            }
                            /*
                            out_buffs[i][0 * N + idx] = in_buffs[0][idx];
                            out_buffs[i][1 * N + idx] = in_buffs[1][idx];
                            out_buffs[i][2 * N + idx] = in_buffs[2][idx];
                            out_buffs[i][3 * N + idx] = in_buffs[3][idx];
                            */

                        }
                    });
                });
            } else {
                for (int j = 0; j < num_devices; j++) {
                    if (write) {
                        e = q_vec[i].submit([=](sycl::handler &h) {
                            h.depends_on(e);
                            h.memcpy(out_buffs[j] + i * N, in_buffs[i], N * sizeof(long));
                        });
                    }
                    else {
                        e = q_vec[i].submit([=](sycl::handler &h) {
                            h.depends_on(e);
                            h.memcpy(out_buffs[i] + j * N, in_buffs[j], N * sizeof(long));
                        });
                    }
                }
            }
            e_vec.push_back(e);
        }

        for (int i=0; i<num_devices; i++) {
            e_vec[i].wait();
        }

    }

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "size : "<<sizeof(long) * N;
    std::cout << "  time : " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count()/rep<< " µs\n";

    long* host_buff = sycl::malloc_host<long>(N * num_devices, q_vec[0]);
    for (int j=0; j<num_devices; j++) {
        q_vec[j].memcpy(host_buff, out_buffs[j], N * num_devices * sizeof(long)).wait();
        for(long i=0; i < N * num_devices; i++) {
            if(host_buff[i] != 23) {
                std::cout<<"Error on device : " << j <<" at index : "<<i<<" with value : "<<host_buff[i]<<std::endl;
                break;
            }
        }
    }

    return 0;
}

