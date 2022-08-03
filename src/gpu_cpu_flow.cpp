#include<CL/sycl.hpp>
#include<unistd.h>
#include<thread>
#include<iostream>

#define USE_SLEEP
void do_sleep() {
#ifdef USE_SLEEP
    sleep(1); std::cout<<"inside host proxy looping\n";
#endif
}

constexpr size_t T = 1;
int main() {

    sycl::queue Q;
    std::cout<<"selected device : "<<Q.get_device().get_info<sycl::info::device::name>() <<"\n";
    std::cout<<"device vendor : "<<Q.get_device().get_info<sycl::info::device::vendor>() <<"\n";

    size_t * val = sycl::malloc_host<size_t>(1, Q);
    val[0] = 0;

    int64_t * temp = sycl::malloc_shared<int64_t>(1, Q);
    temp[0] = 0;

    auto host_thread_func = [=]() {
    //auto e_h = Q.submit([=](sycl::handler &h) {
    //    h.codeplay_host_task([=]() {
            std::cout<<"inside host proxy initial val "<<val[0]<<"\n";

#ifdef USE_ATOMIC_READ_WR
            sycl::ext::oneapi::atomic_ref<size_t, sycl::ext::oneapi::memory_order::acq_rel, sycl::ext::oneapi::memory_scope::system, sycl::access::address_space::global_space> val_atomic(val[0]);
            while(val_atomic.load() == 0) { do_sleep(); }
#else
            while(val[0] == 0) { do_sleep(); }
#endif

            std::cout<<"inside host proxy later val "<<val[0]<<"\n";
    //    });
    //});
    };

    std::thread host_thread = std::thread(host_thread_func);

    std::cout<<"kernel going to launch\n";

    auto e = Q.submit([&](sycl::handler &h) {
        sycl::stream os(1024, 128, h);
        //h.single_task([=]() {
            h.parallel_for(sycl::nd_range<1>{{T}, {T}}, [=](sycl::nd_item<1> idx) {
#ifdef USE_ATOMIC_READ_WR
                sycl::ext::oneapi::atomic_ref<size_t, sycl::ext::oneapi::memory_order::acq_rel, sycl::ext::oneapi::memory_scope::system, sycl::access::address_space::global_space> val_atomic(val[0]);
                val_atomic.store(22);
#else
                val[0] = 22;
#endif
                
                double delay = 0;
                constexpr long D = 1000000000;
                //constexpr long D = 0;

                //add a delay to interleave gpu kernel and cpu thread
                for(int j=0;j<D;j++) delay+=j;

                //os<<"kernel sum: "<<delay<<"\n";
                temp[0] = delay;
        });
    });
    std::cout<<"kernel launched\n";

    e.wait_and_throw();
    std::cout<<"kernel over. temp: "<<temp[0]<<"\n";
    //e_h.wait_and_throw();
    host_thread.join();
    sycl::free(val, Q);
    return 0;
}
