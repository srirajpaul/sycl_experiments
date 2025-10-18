#include<sycl/sycl.hpp>
#include<iostream>
#include<unistd.h>

int main() {

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

    assert(devices.size() >= 2);
    // create queue on different gpus
    sycl::queue q(devices[0]);
    sycl::queue q1(devices[1]);

    std::cout<<"selected device : "<<q.get_device().get_info<sycl::info::device::name>() <<"\n";
    std::cout<<"device vendor : "<<q.get_device().get_info<sycl::info::device::vendor>() <<"\n";

    int *buf = sycl::malloc_device<int>(1, q);
    q.memset(buf, 0, sizeof(int)).wait();
    //int *buf = sycl::malloc_device<int>(1, q1);
    //q1.memset(buf, 0, sizeof(int)).wait();

    // one gpu waits for buffer to become 1
    sycl::event e = q.submit([=](sycl::handler &h) {
        h.parallel_for(1, [=](sycl::item<1> idx) {
            sycl::atomic_ref<int, sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space>
                atomic_p(*buf);

            size_t val = atomic_p.load();
            while(val < 1) {
                val = atomic_p.load();
            }
        });
    });
    sleep(1);

    // other gpu sets buffer to 1
    q1.submit([=](sycl::handler &h) {
        h.parallel_for(1, [=](sycl::item<1> idx) {
            sycl::atomic_ref<int, sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space>
                atomic_p(*buf);
            atomic_p.store(1);
        });
    }).wait();

    e.wait();

    return 0;
}
