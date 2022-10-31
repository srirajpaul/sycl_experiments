#include <CL/sycl.hpp>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <getopt.h>
#include "level_zero/ze_api.h"
#include <sys/mman.h>
#include"../common_includes/rdtsc.h"
#include <time.h>
#include <sys/stat.h>
#include <immintrin.h>
#include <chrono>

constexpr size_t BUFSIZE = (1<<26);   // 64 MB
constexpr double NSEC_IN_SEC = 1000000000.0;

#ifdef __SYCL_DEVICE_ONLY__
extern "C" {
  SYCL_EXTERNAL ulong intel_get_cycle_counter() __attribute__((overloadable));
}

ulong get_cycle_counter() {
  return intel_get_cycle_counter();
}
#else
ulong get_cycle_counter() {
  return 0xDEADBEEF;
}
#endif // __SYCL_DEVICE_ONLY__

/* command line mode arguments:
 *
 * storeflush iter setflagloc pollflagloc tocpumode fromcpumode
 *   tocpuloc 0 = device, 1 = host
 *   togpuloc 0 = device, 1 = host
 *   mode  0 = pointer, 1 = atomic, 2 = uncached
 */



void printduration(const char* name, sycl::event e)
  {
    ulong start =
      e.get_profiling_info<sycl::info::event_profiling::command_start>();
    ulong end =
      e.get_profiling_info<sycl::info::event_profiling::command_end>();
    double duration = static_cast<double>(end - start) / NSEC_IN_SEC;
    std::cout << name << " execution time: " << duration << " sec" << std::endl;
  }


// ############################

int main(int argc, char *argv[]) {
  unsigned long start_host_time, end_host_time;
  unsigned long start_device_time, end_device_time;
  struct timespec ts_start, ts_end;
  sycl::property_list prop_list{sycl::property::queue::enable_profiling()};

  // modelled after Jeff Hammond code in PRK repo
  std::vector<sycl::queue> qs;
  
  auto platforms = sycl::platform::get_platforms();
  for (auto & p : platforms) {
    auto pname = p.get_info<sycl::info::platform::name>();
    std::cout << "*Platform: " << pname << std::endl;
    if ( pname.find("Level-Zero") == std::string::npos) {
        std::cout << "non Level Zero GPU skipped" << std::endl;
        continue;
    }
    auto devices = p.get_devices();
    for (auto & d : devices ) {
        std::cout << "**Device: " << d.get_info<sycl::info::device::name>() << std::endl;
        if ( d.is_gpu() ) {
            std::cout << "**Device is GPU - adding to vector of queues" << std::endl;
            qs.push_back(sycl::queue(d, prop_list));
        }
    }
  }

  int haz_ngpu = qs.size();
  std::cout << "Number of GPUs found  = " << haz_ngpu << std::endl;

  int ngpu = 3;   // make a command line argument
  assert(ngpu <= haz_ngpu);
  std::vector<uint64_t *> dev_src(ngpu);
  std::vector<uint64_t *> dev_dest(ngpu);
  std::vector<uint64_t *> host_src(ngpu);
  std::vector<uint64_t *> host_dest(ngpu);

  for (int i = 0; i < ngpu; i += 1) {

    std::cout<<"selected device : "<<qs[i].get_device().get_info<sycl::info::device::name>() << std::endl;
    std::cout<<"device vendor : "<<qs[i].get_device().get_info<sycl::info::device::vendor>() << std::endl;
    uint device_frequency = (uint)qs[i].get_device().get_info<sycl::info::device::max_clock_frequency>();
    std::cout << "device frequency " << device_frequency << std::endl;
    dev_src[i] = (uint64_t *) sycl::aligned_alloc_device(4096, BUFSIZE, qs[i]);
    dev_dest[i] = (uint64_t *) sycl::aligned_alloc_device(4096, BUFSIZE, qs[i]);

  host_src[i] = (ulong *) sycl::aligned_alloc_host(4096, BUFSIZE, qs[i]);
  host_dest[i] = (ulong *) sycl::aligned_alloc_host(4096, BUFSIZE, qs[i]);
    std::cout << " dev_src[" << i << "] " << dev_src[i] << std::endl;
    std::cout << " dev_dest[" << i << "] " << dev_dest[i] << std::endl;
    std::cout << " host_src[" << i << "] " << host_src[i] << std::endl;
    std::cout << " host_dest[" << i << "] " << host_dest[i] << std::endl;
  }

  int iter = 0;
  for (int d = 0; d < ngpu; d += 1) {
    for (int i = 0; i < BUFSIZE/sizeof(ulong); i += 1) host_src[d][i] = ((ulong) d << 32) + i;
    memset(host_dest[d], 0xff, BUFSIZE);
  // initialize device_mem
    qs[d].memcpy(dev_src[d], host_src[d], BUFSIZE);
    qs[d].wait_and_throw();
    qs[d].memcpy(dev_dest[d], host_dest[d], BUFSIZE);
    qs[d].wait_and_throw();
  }

  std::cout << " memory initialized " << std::endl;

  // Measure time for an empty kernel (with size 1)
  printf("csv,mode,size,count,duration,bandwidth\n");
    // size is in bytes
  for (int mode = 0; mode < 5; mode += 1) {
    ulong *s, *d;
    switch (mode) {
    case 0:  // push
      s = dev_src[0];
      d = dev_dest[1];
      break;
    case 1:  // pull
      s = dev_src[1];
      d = dev_dest[0];
      break;
    case 2:  // same
      s = dev_src[0];
      d = dev_dest[2];
      break;
    case 3:  // pull
      s = dev_src[2];
      d = dev_dest[0];
      break;
    case 4:  // same
      s = dev_src[0];
      d = dev_dest[0];
      break;
     
    default:
      assert(0);
    }
    for (size_t size = 1; size < BUFSIZE; size <<= 1) {
      int iterations = size / sizeof(ulong);
      double duration;
      int count;
      // run for more and more counts until it takes more than 0.1 sec
      for (count = 1; count < (1 << 20); count <<= 1) {
	clock_gettime(CLOCK_REALTIME, &ts_start);
	for (int iter = 0; iter < count; iter += 1) {
	  auto e = qs[0].submit([&](sycl::handler &h) {
	      h.parallel_for(sycl::range(iterations), [=](sycl::id<1> idx) {
		  d[idx] = s[idx];
		});
	    });
	  e.wait_and_throw();
	}
	// sequential
	clock_gettime(CLOCK_REALTIME, &ts_end);
	duration = ((double) (ts_end.tv_sec - ts_start.tv_sec)) * 1000000000.0 +
	  ((double) (ts_end.tv_nsec - ts_start.tv_nsec));
	
	duration /= 1000000000.0;
	if (duration > 0.1) {
	  break;
	}
      }
      double per_iter = duration / count;
      double bw = size / (per_iter);
      double bw_mb = bw / 1000000.0;
      printf("csv,%d, %ld,%d,%f,%f\n", mode, size, count, duration, bw_mb);
    }
  }
  // check destination buffer
  
  std::cout<<"kernel returned" << std::endl;
  return 0;
}

