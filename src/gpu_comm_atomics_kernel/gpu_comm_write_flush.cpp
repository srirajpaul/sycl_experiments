#include<sycl/sycl.hpp>
#include<iostream>
#include<unistd.h>

constexpr int NT = 4;
using AT = sycl::vec<int, NT>;

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
    std::array<int*, num_gpus> src_ptrs, dst_ptrs, tmp_ptrs, host_ptrs, src_host_ptrs;
    for(int i=0 ; i < num_gpus; i++) {
        src_ptrs[i] = sycl::malloc_device<int>(N, q[i]);
        dst_ptrs[i] = sycl::malloc_device<int>(N, q[i]);
        tmp_ptrs[i] = sycl::malloc_device<int>(N, q[i]);
        host_ptrs[i] = sycl::malloc_host<int>(N, q[i]);
        src_host_ptrs[i] = sycl::malloc_host<int>(N, q[i]);

        q[i].memset(src_ptrs[i], 0, N*sizeof(int)).wait();
        q[i].memset(dst_ptrs[i], 0, N*sizeof(int)).wait();
        q[i].memset(tmp_ptrs[i], 0, N*sizeof(int)).wait();
        q[i].memset(host_ptrs[i], 0, N*sizeof(int)).wait();
        q[i].memset(src_host_ptrs[i], 0, N*sizeof(int)).wait();
    }

    // init values
    for(int i=0 ; i < num_gpus; i++) {
        q[i].parallel_for(N, [=](sycl::id<1> it) {
            const size_t idx = it;
            src_ptrs[i][idx] = idx + 3 * (i + 1);
        });
    }
    for(int i=0 ; i < num_gpus; i++) {
        q[i].wait();
    }

    // gpu i writes to gpu i+1
    std::array<sycl::event, num_gpus> evts;
    for(int i=0 ; i < num_gpus; i++) {
        int dest = (i + 1) % num_gpus;
        evts[i] = q[i].parallel_for(N/NT, [=](sycl::id<1> it) {
            const size_t idx = it;
            ((AT*)(tmp_ptrs[dest]))[idx] = ((AT*)(src_ptrs[i]))[idx];
        });
    }

    // gpu i depends on gpu i-1
    for(int i=0 ; i < num_gpus; i++) {
        int src = (i - 1 + num_gpus) % num_gpus;
        q[i].submit([=](sycl::handler &h) {
            h.depends_on(evts[src]);
            h.parallel_for(N/NT, [=](sycl::item<1> it) {
                const size_t idx = it.get_id();
                ((AT*)(dst_ptrs[i]))[idx] = ((AT*)(tmp_ptrs[i]))[idx];
            });
        });
    }

    // copy output to host
    for(int i=0 ; i < num_gpus; i++) {
        q[i].memcpy(host_ptrs[i], dst_ptrs[i], N * sizeof(int));
        q[i].memcpy(src_host_ptrs[i], src_ptrs[i], N * sizeof(int));
    }
    for(int i=0 ; i < num_gpus; i++) {
        q[i].wait();
    }

    // check output
    for(int i = 1; i < num_gpus; i++) {
        int src = (i - 1 + num_gpus) % num_gpus;
        q[i].submit([=](sycl::handler &h) {
        h.host_task([=]() {
            for(size_t idx = 0; idx < N; idx++) {
                if (src_host_ptrs[src][idx] != host_ptrs[i][idx]) {
                    std::cout<<"gpu "<<i<<" index "<<idx<<" val "<<host_ptrs[i][idx]<<" exp "<<src_host_ptrs[src][idx]<<"\n";
                    break;
                }
            }
        });
        });
    }

    return 0;
}
