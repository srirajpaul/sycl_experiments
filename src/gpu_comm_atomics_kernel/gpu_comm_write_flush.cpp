#include<sycl/sycl.hpp>
#include<iostream>
#include<unistd.h>

#if 0
constexpr int NT = 4;
using AT = sycl::vec<int, NT>;
#else
constexpr int NT = 1;
using AT = int;
#endif

int main(int argc, char *argv[]) {

    auto platforms = sycl::platform::get_platforms();
    std::vector<sycl::device> devices;

    for (auto &platform : platforms) {
        auto backend = platform.get_backend();

        //if(backend == sycl::backend::ext_oneapi_level_zero) {
            std::cout << "Backend: " << backend << std::endl;
        //}

        std::cout << "Platform: " << platform.get_info<sycl::info::platform::name>()
              << std::endl;

        if(backend == sycl::backend::ext_oneapi_level_zero) {
            auto devices_tmp = platform.get_devices();
            for (auto &device : devices_tmp) {
                std::cout << "  Device: " << device.get_info<sycl::info::device::name>()
                    << std::endl;
                devices.push_back(device);
            }
        }
    }

    constexpr int num_gpus = 4;
    const int N = (argc >= 2) ? atol(argv[1]) : 4;

    std::cout<<"num GPUs "<<devices.size()<<"\n";
    assert(devices.size() >= num_gpus);

    // create queue on different gpus
    std::vector<sycl::queue> q;
    for(int i=0 ; i < num_gpus; i++) {
        q.push_back(sycl::queue(devices[i], sycl::property_list{sycl::property::queue::in_order{}}));
    }

    std::cout<<"selected device : "<<q[0].get_device().get_info<sycl::info::device::name>() <<"\n";
    std::cout<<"device vendor : "<<q[0].get_device().get_info<sycl::info::device::vendor>() <<"\n";

    // create buffers
    std::array<int*, num_gpus> src_ptrs, dst_ptrs, src_host_ptrs, dst_host_ptrs;
    std::array<std::array<int*, num_gpus>, num_gpus> tmp_ptrs;
    for(int i=0 ; i < num_gpus; i++) {
        src_ptrs[i] = sycl::malloc_device<int>(N, q[i]);
        dst_ptrs[i] = sycl::malloc_device<int>(N, q[i]);
        src_host_ptrs[i] = sycl::malloc_host<int>(N, q[i]);
        dst_host_ptrs[i] = sycl::malloc_host<int>(N, q[i]);
        int *tmp_ptr = sycl::malloc_device<int>(N * num_gpus, q[i]);
        for(int j=0 ; j < num_gpus; j++) {
            tmp_ptrs[i][j] = tmp_ptr + j * N;
        }

        q[i].memset(src_ptrs[i], 0, N*sizeof(int)).wait();
        q[i].memset(dst_ptrs[i], 0, N*sizeof(int)).wait();
        q[i].memset(src_host_ptrs[i], 0, N*sizeof(int)).wait();
        q[i].memset(dst_host_ptrs[i], 0, N*sizeof(int)).wait();
        q[i].memset(tmp_ptr, 0, N*num_gpus*sizeof(int)).wait();
    }

    // init values
    for(int i=0 ; i < num_gpus; i++) {
        q[i].parallel_for(N, [=](sycl::id<1> it) {
            const size_t idx = it;
            src_ptrs[i][idx] = i + 1;
        });
        // copy to first tmp
        q[i].memcpy(tmp_ptrs[i][0], src_ptrs[i], N * sizeof(int));
        // copy to dest
        q[i].memcpy(dst_ptrs[i], src_ptrs[i], N * sizeof(int));
    }
    for(int i=0 ; i < num_gpus; i++) {
        q[i].wait();
    }

    for (int n = 0; n < num_gpus - 1; n++) {
        // gpu i writes to gpu i+1
        std::array<sycl::event, num_gpus> evts;
        for(int i=0 ; i < num_gpus; i++) {
            int dest = (i + 1) % num_gpus;
            evts[i] = q[i].parallel_for(N/NT, [=](sycl::id<1> it) {
                const size_t idx = it;
                // remote write
                ((AT*)(tmp_ptrs[dest][n+1]))[idx] = ((AT*)(tmp_ptrs[i][n]))[idx];
            });
        }

        // gpu i depends on gpu i-1
        for(int i=0 ; i < num_gpus; i++) {
            int src = (i - 1 + num_gpus) % num_gpus;
            q[i].submit([=](sycl::handler &h) {
                h.depends_on(evts[src]);
                h.parallel_for(N/NT, [=](sycl::item<1> it) {
                    const size_t idx = it.get_id();
                    // local reduce
                    ((AT*)(dst_ptrs[i]))[idx] += ((AT*)(tmp_ptrs[i][n+1]))[idx];
                });
            });
        }
    }

    // copy output to host
    for(int i=0 ; i < num_gpus; i++) {
        q[i].memcpy(dst_host_ptrs[i], dst_ptrs[i], N * sizeof(int));
        q[i].memcpy(src_host_ptrs[i], src_ptrs[i], N * sizeof(int));
    }
    for(int i=0 ; i < num_gpus; i++) {
        q[i].wait();
    }

    // check output
    size_t expected_value = num_gpus * (num_gpus - 1) / 2 + num_gpus;
    bool print_all = false;
    for(int i = 0; i < num_gpus; i++) {
        int src = (i - 1 + num_gpus) % num_gpus;
        q[i].submit([=](sycl::handler &h) {
          h.host_task([=]() {
            for(size_t idx = 0; idx < N; idx++) {
              if (print_all) {
                std::cout<<"gpu "<<i<<" index "<<idx<<" val "<<dst_host_ptrs[i][idx]<<" exp "<<expected_value<<"\n";
              }
              else if (src_host_ptrs[src][idx] != dst_host_ptrs[i][idx]) {
                std::cout<<"gpu "<<i<<" index "<<idx<<" val "<<dst_host_ptrs[i][idx]<<" exp "<<expected_value<<"\n";
                break;
              }
            }
          });
        });
    }

    return 0;
}
