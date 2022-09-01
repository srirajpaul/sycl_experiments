#include <CL/sycl.hpp>
#include <stdlib.h>
#include <unistd.h>
#include <thread>
#include <getopt.h>
#include <iostream>
#include "level_zero/ze_api.h"
#include <CL/sycl/backend/level_zero.hpp>
#include <sys/mman.h>
#include"../common_includes/rdtsc.h"
#include <time.h>
#include <sys/stat.h>
#include <immintrin.h>

constexpr int use_atomic_flag_load = 0;
constexpr int use_atomic_flag_store = 1;

#define NSEC_IN_SEC 1000000000.0

constexpr size_t BUFSIZE = (1L << 21);  //allocate 2 MB, to permit double buffering up to 1 MB
constexpr size_t CTLSIZE = (4096);
constexpr uint64_t cpu_to_gpu_index = 0;
constexpr uint64_t gpu_to_cpu_index = 8;
#define cpu_relax() asm volatile("rep; nop")
#define nullptr NULL
/* option codes */
#define CMD_COUNT 1001
#define CMD_READ 1002
#define CMD_WRITE 1003
#define CMD_HELP 1004
#define CMD_VALIDATE 1005
#define CMD_CPUTOGPU 1006
#define CMD_GPUTOCPU 1007
#define CMD_DEVICEDATA 1008
#define CMD_HOSTDATA 1009
#define CMD_DEVICECTL 1010
#define CMD_HOSTCTL 1011
#define CMD_SPLITCTL 1012
#define CMD_HOSTLOOP 1013
#define CMD_HOSTMEMCPY 1014
#define CMD_HOSTAVX 1015
#define CMD_DEVICELOOP 1016
#define CMD_DEVICEMEMCPY 1017
#define CMD_GPUTHREADS 1018
/* option global variables */
// specify defaults
uint64_t glob_count = 1;
uint64_t glob_readsize = 32768;
uint64_t glob_writesize = 32768;
int glob_validate = 0;
int glob_cputogpu = 0;
int glob_gputocpu = 0;
int glob_devicedata = 0;
int glob_hostdata = 0;
int glob_devicectl = 0;
int glob_hostctl = 0;
int glob_splitctl = 0;
int glob_hostloop = 0;
int glob_hostmemcpy = 0;
int glob_hostavx = 0;
int glob_deviceloop = 0;
int glob_devicememcpy = 0;
uint64_t glob_gputhreads = 1;

void Usage()
{
  std::cout <<
    "--count <n>            set number of iterations\n"
    "--read <n>             set read size\n"
    "--write <n>            set write size\n"
    "--help                 usage message\n"
    "--validate             set and check data\n"
    "--cputogpu | --gputocpu   direction of data transfer\n"
    "--devicedata | --hostdata location of data buffer\n"
    "--devicectl | --hostctl | --splitctl    location of control flags\n"
    "--hostloop | --hostmemcpy | --hostavx   code for host\n"
    "--deviceloop | --devicememcpy           code for device\n";
  std::cout << std::endl;
  exit(1);
}



void ProcessArgs(int argc, char **argv)
{
  const char* short_opts = "c:r:w:vhDHALM";
  const option long_opts[] = {
    {"count", required_argument, nullptr, CMD_COUNT},
    {"read", required_argument, nullptr, CMD_READ},
    {"write", required_argument, nullptr, CMD_WRITE},
    {"help", no_argument, nullptr, CMD_HELP},
    {"validate", no_argument, nullptr, CMD_VALIDATE},
    {"cputogpu", no_argument, nullptr, CMD_CPUTOGPU},
    {"gputocpu", no_argument, nullptr, CMD_GPUTOCPU},
    {"devicedata", no_argument, nullptr, CMD_DEVICEDATA},
    {"hostdata", no_argument, nullptr, CMD_HOSTDATA},
    {"devicectl", no_argument, nullptr, CMD_DEVICECTL},
    {"hostctl", no_argument, nullptr, CMD_HOSTCTL},
    {"splitctl", no_argument, nullptr, CMD_SPLITCTL},

    {"hostloop", no_argument, nullptr, CMD_HOSTLOOP},
    {"hostmemcpy", no_argument, nullptr, CMD_HOSTMEMCPY},
    {"hostavx", no_argument, nullptr, CMD_HOSTAVX},
    {"deviceloop", no_argument, nullptr, CMD_DEVICELOOP},
    {"devicememcpy", no_argument, nullptr, CMD_DEVICEMEMCPY},
    {"gputhreads", required_argument, nullptr, CMD_GPUTHREADS},
    {nullptr, no_argument, nullptr, 0}
  };
  while (true)
    {
      const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);
      if (-1 == opt)
	break;
      switch (opt)
	{
	case CMD_COUNT: {
	  glob_count = std::stoi(optarg);
	  std::cout << "count " << glob_count << std::endl;
	  break;
	}
	case CMD_READ: {
	  glob_readsize = std::stoi(optarg);
	  std::cout << "read " << glob_readsize << std::endl;
	  break;
	}
	case CMD_WRITE: {
	  glob_writesize = std::stoi(optarg);
	  std::cout << "write " << glob_writesize << std::endl;
	  break;
	}
	case CMD_HELP: {
	  Usage();
	  break;
	}
	case CMD_VALIDATE: {
	  glob_validate = true;
	  break;
	}
	case CMD_CPUTOGPU: {
	  glob_cputogpu = true;
	  glob_gputocpu = false;
	  break;
	}
	case CMD_GPUTOCPU: {
	  glob_cputogpu = false;
	  glob_gputocpu = true;
	  break;
	}
	case CMD_DEVICEDATA: {
	  glob_devicedata = true;
	  glob_hostdata = false;
	  break;
	}
	case CMD_HOSTDATA: {
	  glob_devicedata = false;
	  glob_hostdata = true;
	  break;
	}
	case CMD_DEVICECTL: {
	  glob_devicectl = true;
	  glob_hostctl = false;
	  glob_splitctl = false;
	  break;
	}
	case CMD_HOSTCTL: {
	  glob_devicectl = false;
	  glob_hostctl = true;
	  glob_splitctl = false;
	  break;
	}
	case CMD_SPLITCTL: {
	  glob_devicectl = false;
	  glob_hostctl = false;
	  glob_splitctl = true;
	  break;
	}

	case CMD_HOSTLOOP: {
	  glob_hostloop = true;
	  glob_hostmemcpy = false;
	  glob_hostavx = false;
	  break;
	}
	case CMD_HOSTMEMCPY: {
	  glob_hostloop = false;
	  glob_hostmemcpy = true;
	  glob_hostavx = false;
	  break;
	}
	case CMD_HOSTAVX: {
	  glob_hostloop = false;
	  glob_hostmemcpy = false;
	  glob_hostavx = true;
	  break;
	}
	case CMD_DEVICELOOP: {
	  glob_deviceloop = true;
	  glob_devicememcpy = false;
	  break;
	}
	case CMD_DEVICEMEMCPY: {
	  glob_deviceloop = false;
	  glob_devicememcpy = true;
	  break;
	}
	case CMD_GPUTHREADS: {
	  glob_gputhreads = std::stoi(optarg);
	  std::cout << "gputhreads " << glob_gputhreads << std::endl;
	  break;
	}

    };
    }
}

void printduration(const char* name, sycl::event e)
  {
    uint64_t start =
      e.get_profiling_info<sycl::info::event_profiling::command_start>();
    uint64_t end =
      e.get_profiling_info<sycl::info::event_profiling::command_end>();
    double duration = static_cast<double>(end - start) / NSEC_IN_SEC;
    std::cout << name << " execution time: " << duration << " sec" << std::endl;
  }

void fillbuf(uint64_t *p, uint64_t size, int iter)
{
  for (size_t j = 0; j < size >> 3; j += 1) {
    p[j] = (((long) iter) << 32) + j;
  }
}

long unsigned checkbuf(uint64_t *p, uint64_t size, int iter)
{
  long unsigned errors = 0;
  for (size_t j = 0; j < size >> 3; j += 1) {
    if (p[j] != (((long) iter) << 32) + j) errors += 1;
  }
  return (errors);
}

template<typename T>
T *get_mmap_address(T * device_ptr, size_t size, sycl::queue Q) {
    sycl::context ctx = Q.get_context();
    ze_ipc_mem_handle_t ze_ipc_handle;
    ze_result_t ret = zeMemGetIpcHandle(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx), device_ptr, &ze_ipc_handle);
    std::cout<<"zeMemGetIpcHandle return : " << ret << std::endl;
    assert(ret == ZE_RESULT_SUCCESS);
    int fd;
    memcpy(&fd, &ze_ipc_handle, sizeof(fd));
    std::cout << " fd " << fd << std::endl;
    struct stat statbuf;
    fstat(fd, &statbuf);
    std::cout << "requested size " << size << std::endl;
    std::cout << "fd size " << statbuf.st_size << std::endl;

    void *base = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == (void *) -1) {
      std::cout << "mmap returned -1" << std::endl;
      std::cout << strerror(errno) << std::endl;  
    }
    assert(base != (void *) -1);
    return (T*)base;
}


int main(int argc, char *argv[]) {
  ProcessArgs(argc, argv);
  uint64_t loc_count = glob_count;
  uint64_t loc_readsize = glob_readsize;
  uint64_t loc_writesize = glob_writesize;
  int loc_validate = glob_validate;
  int loc_cputogpu = glob_cputogpu;
  int loc_gputocpu = glob_gputocpu;
  int loc_devicedata = glob_devicedata;
  int loc_hostdata = glob_hostdata;
  int loc_devicectl = glob_devicectl;
  int loc_hostctl = glob_hostctl;
  int loc_splitctl = glob_splitctl;
  int loc_hostloop = glob_hostloop;
  int loc_hostmemcpy = glob_hostmemcpy;
  int loc_hostavx = glob_hostavx;
  int loc_deviceloop = glob_deviceloop;
  int loc_devicememcpy = glob_devicememcpy;
  uint64_t loc_gputhreads = glob_gputhreads;

  if (loc_readsize > BUFSIZE/4) loc_readsize = BUFSIZE/4;
  if (loc_writesize > BUFSIZE/4) loc_writesize = BUFSIZE/4;
  sycl::property_list prop_list{sycl::property::queue::enable_profiling()};
  sycl::queue Q(sycl::gpu_selector{}, prop_list);
  std::cout<<"selected device : "<<Q.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout<<"device vendor : "<<Q.get_device().get_info<sycl::info::device::vendor>() << std::endl;

  // allocate device memory

  uint64_t *device_data_mem = (uint64_t *) sycl::aligned_alloc_device(4096, BUFSIZE, Q);
  std::cout << " device_data_mem " << device_data_mem << std::endl;

  uint64_t *extra_device_mem = (uint64_t *)sycl::aligned_alloc_device(4096, BUFSIZE, Q);
  std::cout << " extra_device_mem " << extra_device_mem << std::endl;

  uint64_t *device_ctl_mem = (uint64_t *) sycl::aligned_alloc_device(4096, CTLSIZE, Q);
  std::cout << " device_ctl_mem " << device_ctl_mem << std::endl;

  //allocate host mamory
  uint64_t *host_data_mem = (uint64_t *) sycl::aligned_alloc_host(4096, BUFSIZE, Q);
  std::cout << " host_data_mem " << host_data_mem << std::endl;
  memset(&host_data_mem[0], 0, BUFSIZE);

  std::array<uint64_t, BUFSIZE> host_data_array;
  memset(&host_data_array[0], 0, BUFSIZE);
  
  std::array<uint64_t, CTLSIZE> host_ctl_array;
  memset(&host_ctl_array[0], 0, CTLSIZE);
 
  uint64_t * host_ctl_mem = (uint64_t *) sycl::aligned_alloc_host(4096, CTLSIZE, Q);
  std::cout << " host_ctl_mem " << host_ctl_mem << std::endl;
  memset(&host_ctl_mem[0], 0, CTLSIZE);


  //create mmap mapping of usm device memory on host
  sycl::context ctx = Q.get_context();

  uint64_t *host_data_map;   // pointer for host to use
  uint64_t *device_mem;   // pointer for device to use
  uint64_t *host_mem;   // pointer for host to use

  std::cout << "About to call mmap" << std::endl;

  host_data_map = get_mmap_address(device_data_mem, BUFSIZE / 2, Q);

  if (loc_devicedata) {
    device_mem = device_data_mem;
    host_mem = host_data_map;
    std::cout << " host_data_map " << host_data_map << std::endl;
  } else {
    device_mem = host_data_mem;
    host_mem = host_data_mem;
  }

  volatile uint64_t *host_cputogpu;
  volatile uint64_t *host_gputocpu;
  volatile uint64_t *device_cputogpu;
  volatile uint64_t *device_gputocpu;

  uint64_t *host_ctl_map;

  host_ctl_map = get_mmap_address(device_ctl_mem, CTLSIZE, Q);
  std::cout << " host_ctl_map " << host_ctl_map << std::endl;
  // both flags are in host memory

  if (loc_hostctl) {
    // to gpu
    host_cputogpu = &host_ctl_mem[cpu_to_gpu_index];
    device_cputogpu = &host_ctl_mem[cpu_to_gpu_index];
    // to host
    host_gputocpu = &host_ctl_mem[gpu_to_cpu_index];
    device_gputocpu = &host_ctl_mem[gpu_to_cpu_index];
    std::cout << "setting hostctl" << std::endl;
  }
  // both flags are in device memory
  if (loc_devicectl) {
    // to gpu
    host_cputogpu = &host_ctl_map[cpu_to_gpu_index];
    device_cputogpu = &device_ctl_mem[cpu_to_gpu_index];
    // to host
    host_gputocpu = &host_ctl_map[gpu_to_cpu_index];
    device_gputocpu = &device_ctl_mem[gpu_to_cpu_index];
    std::cout << "setting devicectl" << std::endl;
  }
  // to gpu flag is in device memory
  // to cpu flag is in host memory
  if (loc_splitctl) {
    // to gpu
    host_cputogpu = &host_ctl_map[cpu_to_gpu_index];
    device_cputogpu = &device_ctl_mem[cpu_to_gpu_index];
    //host_cputogpu = &host_ctl_mem[cpu_to_gpu_index];
    //device_cputogpu = &host_ctl_mem[cpu_to_gpu_index];
    // to host
    host_gputocpu = &host_ctl_mem[gpu_to_cpu_index];
    device_gputocpu = &host_ctl_mem[gpu_to_cpu_index];
    std::cout << "setting splitctl" << std::endl;
  }

  uint64_t *nv_device_cputogpu = (uint64_t *) device_cputogpu;
  uint64_t *nv_device_gputocpu = (uint64_t *) device_gputocpu;
  // initialize device memory  
  auto e = Q.submit([&](sycl::handler &h) {
      h.memcpy(device_mem, &host_data_array[0], BUFSIZE/2);
    });
  e.wait_and_throw();
  printduration("memcpy kernel ", e);

  e = Q.submit([&](sycl::handler &h) {
      h.memcpy((void *) device_cputogpu, &host_ctl_array[0], sizeof(uint64_t));
    });
  e.wait_and_throw();
  e = Q.submit([&](sycl::handler &h) {
      h.memcpy((void *) device_gputocpu, &host_ctl_array[0], sizeof(uint64_t));
    });
  e.wait_and_throw();
  std::cout << "kernel going to launch" << std::endl;
  unsigned long start_time, end_time;
  struct timespec ts_start, ts_end;
  uint64_t loc_globalsize = loc_readsize;
  if (loc_writesize > loc_readsize) loc_globalsize = loc_writesize;
  loc_globalsize /= sizeof(uint64_t);
  uint64_t loc_loop = loc_globalsize / loc_gputhreads;
  if (loc_cputogpu) {
    // initialize the flag
    *host_cputogpu = -1L;  /* initial non-value */
    *host_gputocpu = -1L;  /* initial non-value */
    e = Q.submit([&](sycl::handler &h) {
	//   sycl::stream os(1024, 128, h);
	//h.single_task([=]() {
	h.parallel_for_work_group(sycl::range(1), sycl::range(loc_gputhreads), [=](sycl::group<1> grp) {
	    //  os<<"kernel start\n";
	    uint64_t prev = (uint64_t) -1L;
	    sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::system, sycl::access::address_space::global_space> cpu_to_gpu(nv_device_cputogpu[0]);
	    sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::system, sycl::access::address_space::global_space> gpu_to_cpu(nv_device_gputocpu[0]);
	    uint64_t i;
	    do {
	      uint64_t err = 0;
	      for (;;) {
		if (use_atomic_flag_load) {
		  i = cpu_to_gpu.load();
		} else {
		  sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
		  i = *device_cputogpu;
		}
		if (i != prev) break;
	      }
	      prev = i;
	      if (loc_readsize > 0) {
		uint64_t *p;
		if (i & 1) {
		  p = &device_mem[loc_readsize >> 3];
		} else {
		  p = &device_mem[0];
		}
		if (loc_devicememcpy) {
		  memcpy(extra_device_mem, p, loc_readsize);
		}
	        if (loc_deviceloop) {
		  grp.parallel_for_work_item([&] (sycl::h_item<1> it) {
		      int j = it.get_global_id()[0];
		      for (int k = j * loc_loop; k < (j+1) * loc_loop; k += 1)
			extra_device_mem[k] = p[k];
		    });
		}
		if (loc_validate) {
		  err = checkbuf(extra_device_mem, loc_readsize, i);
		}
	      }
	      if (use_atomic_flag_store) {
		gpu_to_cpu.store((err << 32) + i);
	      } else {
		*device_gputocpu = (err << 32) + i;
		sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
	      }
	    } while (i < (loc_count-1) );
	    // os<<"kernel exit\n";
	  });
      });
    std::cout<<"kernel launched" << std::endl;
    { // start of host block for cputogpu
      clock_gettime(CLOCK_REALTIME, &ts_start);
      start_time = rdtsc();
      // loop start code to initialize initial message
      fillbuf(&host_data_array[0], loc_writesize, 0);
      memcpy(host_mem, &host_data_array[0], loc_writesize);
      clock_gettime(CLOCK_REALTIME, &ts_start);
      start_time = rdtsc();
      for (int i = 0; i < loc_count; i += 1) {
	*host_cputogpu = i;
	if (loc_writesize > 0) {
	  // this block is preparing the <next> message while we wait for the current one
	  uint64_t *p;
	  if (i & 1) {
	    p = &host_mem[0];
	  } else {
	    p = &host_mem[loc_writesize >> 3];
	  }
	  if (loc_validate) {
	    fillbuf(&host_data_array[0], loc_writesize, i + 1);
	    if (checkbuf(&host_data_array[0], loc_writesize, i + 1) != 0)
	      std::cout << "fillbuf failed" << std::endl;
	  }
	  if (loc_hostmemcpy) {
	    memcpy(p, &host_data_array[0], loc_writesize);
	  }
	  if (loc_hostloop) {
	    for (int j = 0; j < loc_writesize >> 3; j += 1) {
	      p[j] = host_data_array[j];
	    }
	  }
	  if (loc_hostavx) {
	    for (int j = 0; j < loc_writesize>>3; j += 8) {
	      __m512i temp = _mm512_load_epi32((void *) &host_data_array[j]);
	      _mm512_store_si512((void *) &p[j], temp);
	    }
	  }
	}
	uint64_t temp;
	for (;;) {   // poll for the ack for the prev message
	  temp = *host_gputocpu;
	  if ((temp & 0xffffffff) == i) break;
	  cpu_relax();
	} 
	if (loc_validate && ((temp >> 32) != 0)) {
	  std::cout << "iteration " << i << " errors " << (temp >> 32) << std::endl;
	}
      }
      end_time = rdtscp();
      clock_gettime(CLOCK_REALTIME, &ts_end);
    }  // end of host block for cputogpu
    e.wait_and_throw();
    } else {  // gputocpu
      uint64_t prev = -1L;
      *host_gputocpu = prev;
      *host_cputogpu = prev;
    e = Q.submit([&](sycl::handler &h) {
	//   sycl::stream os(1024, 128, h);
	//h.single_task([=]() {
	h.parallel_for_work_group(sycl::range(1), sycl::range(loc_gputhreads), [=](sycl::group<1> grp) {
	    //  os<<"kernel start\n";
	    sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::system, sycl::access::address_space::global_space> cpu_to_gpu(nv_device_cputogpu[0]);
	    sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::system, sycl::access::address_space::global_space> gpu_to_cpu(nv_device_gputocpu[0]);
	    /* preload first message */
	    fillbuf(extra_device_mem, loc_writesize, 0);
	    memcpy(device_mem, extra_device_mem, loc_writesize);
	    for (int i = 0; i < loc_count; i += 1) {
	      uint64_t *p;
	      if (use_atomic_flag_store) {
		gpu_to_cpu.store(i);
	      } else {
		*device_gputocpu = i;
		sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
	      }
	      if (loc_writesize > 0) {  // write the next message 
		if (i & 1) {
		  p = &device_mem[0];
		} else {
		  p = &device_mem[loc_writesize >> 3];
		}
		if (loc_validate) {
		  fillbuf(extra_device_mem, loc_writesize, i+1);
		}
		if (loc_devicememcpy) {
		  memcpy(p, extra_device_mem, loc_writesize);
		}
	        if (loc_deviceloop) {
		  grp.parallel_for_work_item([&] (sycl::h_item<1> it) {
		      int j = it.get_global_id()[0];
		      for (int k = j * loc_loop; k < (j+1) * loc_loop; k += 1)
			p[k] = extra_device_mem[k];
		    });
		}
	      }
	      for (;;) {
		uint64_t temp;
		if (use_atomic_flag_load) {
		  temp = cpu_to_gpu.load();
		} else {
		  sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
		  temp = *device_cputogpu;
		}
		if (temp == i) break;
	      }
	    }
	    // os<<"kernel exit\n";
	  });
      });
    std::cout<<"kernel launched" << std::endl;
    {  // start of host block for gputocpu
      clock_gettime(CLOCK_REALTIME, &ts_start);
      start_time = rdtsc();
      uint64_t i = 0;
      do {
	for (;;) {
	  i = *host_gputocpu;
	  if (prev != i) break;
	}
	prev = i;
	if (loc_readsize > 0) {
	  uint64_t *p;
	  //std::cout << " received count " << i << std::endl;
	  if (i & 1) {
	    p = &host_mem[loc_readsize >> 3];
	  } else {
	    p = &host_mem[0];
	  }
	  if (loc_hostmemcpy) {
	    memcpy(&host_data_array[0], p, loc_readsize);
	  }
	  if (loc_hostloop) {
	    for (int j = 0; j < loc_readsize >> 3; j += 1) {
	      host_data_array[j] = p[j];
	    }
	  }
	  if (loc_hostavx) {
	    for (int j = 0; j < loc_readsize>>3; j += 8) {
	      __m512i temp = _mm512_load_epi32((void *) (&p[j]));
	      _mm512_store_si512((void *) &host_data_array[j], temp);
	    }
	  }
	  if (loc_validate) {
	    int err = checkbuf(&host_data_array[0], loc_readsize, i);
	    if (err != 0) std::cout << "iteration " << i << " errors " << err << std::endl;
	  }
	}
	*host_cputogpu = i;
      } while (i < (loc_count - 1));
      end_time = rdtscp();
      clock_gettime(CLOCK_REALTIME, &ts_end);
    }  // end of host block for gputocpu
    e.wait_and_throw();
  }
  printduration("gpu kernel ", e);
    /* common cleanup */
    double elapsed = ((double) (ts_end.tv_sec - ts_start.tv_sec)) * 1000000000.0 +
	((double) (ts_end.tv_nsec - ts_start.tv_nsec));
    //      std::cout << "count " << loc_count << " tsc each " << (end_time - start_time) / loc_count << std::endl;
    std::cout << argv[0];
    if (loc_cputogpu) std::cout << " --cputogpu ";
    if (loc_gputocpu) std::cout << " --gputocpu ";
    if (loc_devicedata) std::cout << "--devicedata ";
    if (loc_hostdata) std::cout << "--hostdata ";

    if (loc_devicectl) std::cout << "--devicectl ";
    if (loc_hostctl) std::cout << "--hostctl ";
    if (loc_splitctl) std::cout << "--splitctl ";

    if (loc_hostavx) std::cout << "--hostavx ";
    if (loc_hostmemcpy) std::cout << "--hostmemcpy ";
    if (loc_hostloop) std::cout << "--hostloop ";
    
    if (loc_devicememcpy) std::cout << "--devicememcpy ";
    if (loc_deviceloop) std::cout << "--deviceloop ";
    std::cout << "--count " << loc_count << " ";
    std::cout << "--read " << loc_readsize << " ";
    std::cout << "--write " << loc_writesize << " ";
    std::cout << "--gputhreads " << loc_gputhreads << " ";

    double mbps = (loc_readsize > loc_writesize) ? loc_readsize : loc_writesize;
    mbps = (mbps * 1000) / (elapsed / ((double) loc_count));
    double nsec = elapsed / ((double) loc_count);
    std::cout << " nsec each " << nsec << " MB/s " << mbps << std::endl;

    if (loc_devicedata)
      munmap(host_data_map, BUFSIZE);
    if (loc_devicectl || loc_splitctl) 
      munmap(host_ctl_map, CTLSIZE);
    sycl::free(device_data_mem, Q);
    sycl::free(extra_device_mem, Q);
    sycl::free(device_ctl_mem, Q);
    return 0;
}
