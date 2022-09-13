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
#include <stdarg.h>

#define DEBUG 0
void dbprintf(int line, const char *format, ...)
{
  va_list arglist;
  printf("line %d: ", line);
  va_start(arglist, format);
  vprintf(format, arglist);
  va_end(arglist);
}

#define DP(...) if (DEBUG) dbprintf(__LINE__, __VA_ARGS__)

#define HERE()
#define CHERE(name) if (DEBUG) std::cout << name << " " <<  __FUNCTION__ << ": " << __LINE__ << std::endl; 

/* option global variables */
// specify defaults
uint64_t glob_cpu_to_gpu_count = 0;
uint64_t glob_gpu_to_cpu_count = 0;
uint64_t glob_size = 0;
int glob_validate = 0;

int glob_use_atomic_load = 0;
int glob_use_atomic_store = 0;
int glob_latency = 0;


#define NSEC_IN_SEC 1000000000.0
/* how many messages per buffer */

constexpr size_t BUFSIZE = (1L << 20);  //allocate 1 MB usable

constexpr int N = 1024;
constexpr int NOP_THRESHOLD = 256;   /* N/4 ? See analysis */


/* message types, all non-zero */
#define MSG_IDLE 0
#define MSG_NOP 1
#define MSG_PUT 2
#define MSG_GET 3

/* should the length be implicit or explicit? */
struct RingMessage {
  int32_t header;
  int32_t next_receive;
  uint64_t data[7];
};
constexpr int MaxData = sizeof(uint64_t) * 7;

/* when peer_next_read and next_write pointers are equal, the buffer is empty.
 * when next_write pointer is one less than peer_next_read pointer, 
 * the buffer is full
 *
 * must not send last message without returning credit
 * should return NOP with credit when half of credits waiting to return
 *
 * need to know how many credits waiting to return
 *   next_receive - next_received_sent
 */

/* each end of the link has a RingState record */

struct RingState {
  int32_t next_send;        // next slot in sendbuf
  int32_t next_receive;     // next slot in recvbuf
  int32_t peer_next_receive;  // last msg read by peer in sendbuf
  int32_t peer_next_receive_sent; // last time next_receive was sent
  struct RingMessage *sendbuf;   // remote buffer
  struct RingMessage *recvbuf;   // local buffer
  int32_t total_received;     // for accounting
  int32_t total_sent;
  int32_t total_nop;
  int32_t send_count;
  int32_t recv_count;
  const char *name;
};

void initstate(struct RingState *s, struct RingMessage *sendbuf, struct RingMessage *recvbuf, int send_count, int recv_count, const char *name)
{
  s->next_send = 0;
  s->next_receive = 0;
  s->peer_next_receive = 0;
  s->peer_next_receive_sent = 0;
  s->sendbuf = sendbuf;
  s->recvbuf = recvbuf;
  s->total_received = 0;
  s->total_sent = 0;
  s->total_nop = 0;
  s->send_count = send_count;
  s->recv_count = recv_count;
  s->name = name;
}

/* how many messages can we send? */
int ring_send_space_available(struct RingState *s)
{
  int space = ((N - 1) + s->peer_next_receive - s->next_send) % N;
  return (space);
}

/* how many credits can we return? 
 * s->next_receive is the next message we haven't received yet
 * s->peer_next_receive_sent is the last next_receive we told the peer about
 * s->next_receive - s->peer_next_receive is the number of messages the peer can
 * send but we haven't told them yet.
 * The idea is that if this number gets larger than N/2 we should send an
 * otherwise empty message to update the peer
 *
 * if there is traffic in both directions, this won't trigger.  If it is needed
 * the trigger threshold should be set to N/2 or some fraction of N such that
 * the update reaches the peer before they run out of credits
 * 
 * The number of slots in the ring buffer should be enough to cover about 2 x
 * the round trip latency so that we can return credits before the peer runs out
 * when all the traffic is one-way
 */
void ring_cpu_process_message(struct RingState *s, struct RingMessage *msg)
{
  if (DEBUG) printf("process %s %d (credit %d\n)\n", s->name, s->total_received, s->peer_next_receive);
  if (msg->header != MSG_NOP) {
    s->total_received += 1;
  }
  /* do what the message says */
}

/* called by ring_cpu_send when we need space to send a message */
void ring_internal_cpu_receive(struct RingState *s)
{
  /* volatile */ struct RingMessage *msg = &(s->recvbuf[s->next_receive]);
  if (msg->header != MSG_IDLE) {
    s->peer_next_receive = msg->next_receive;
    ring_cpu_process_message(s, (struct RingMessage *) msg);
    msg->header = MSG_IDLE;
    s->next_receive = (s->next_receive + 1) % N;
  }
}

void ring_cpu_send_nop(struct RingState *s)
{
  do {
    ring_internal_cpu_receive(s);
  } while (ring_send_space_available(s) == 0);
  struct RingMessage msg;  // local composition of message
  msg.header = MSG_NOP;
  msg.next_receive = s->next_receive;
  s->peer_next_receive_sent = s->next_receive;
  void *mp = (void *) &(s->sendbuf[s->next_send]);
  __m512i temp = _mm512_load_epi32((void *) &msg);
  _mm512_store_si512(mp, temp);
  //  _movdir64b(&(s->sendbuf[s->next_send]), &msg);   // send message (atomic!)
  s->next_send = (s->next_send + 1) % N;
  s->total_nop += 1;
}

int ring_internal_cpu_send_nop_p(struct RingState *s)
{
  int cr_to_send =  (N + s->next_receive - s->peer_next_receive_sent) % N;
  return (cr_to_send > NOP_THRESHOLD);
}

/* called by users to see if any receives messages are available */
void ring_cpu_poll(struct RingState *s)
{
  /* volatile */ struct RingMessage *msg = &(s->recvbuf[s->next_receive]);
  if (msg->header != MSG_IDLE) {
    s->peer_next_receive = msg->next_receive;
    ring_cpu_process_message(s, (struct RingMessage *) msg);
    msg->header = MSG_IDLE;
    s->next_receive = (s->next_receive + 1) % N;
    if (ring_internal_cpu_send_nop_p(s)) ring_cpu_send_nop(s);
  }
}

void ring_cpu_send(struct RingState *s, int type, int length, void *data)
{
  if (DEBUG) printf("send %s next %d\n", s->name, s->next_send);
  do {
    ring_internal_cpu_receive(s);
  } while (ring_send_space_available(s) == 0);
  struct RingMessage msg;  // local composition of message
  msg.header = type;
  msg.next_receive = s->next_receive;
  s->peer_next_receive_sent = s->next_receive;
  assert (length <= MaxData);
  memcpy(&msg.data, data, length);  // local copy
  void *mp = (void *) &(s->sendbuf[s->next_send]);
  __m512i temp = _mm512_load_epi32((void *) &msg);
  _mm512_store_si512(mp, temp);
  //  _movdir64b(&(s->sendbuf[s->next_send]), &msg);   // send message (atomic!)
  s->next_send = (s->next_send + 1) % N;
  s->total_sent += 1;
}

void ring_cpu_drain(struct RingState *s)
{
  CHERE(s->name);
  while (s->total_received < s->recv_count) ring_cpu_poll(s);
}




/* a duplicate set of functions specialized for GPU */


void ring_gpu_process_message(struct RingState *s, struct RingMessage *msg)
{
  if (msg->header != MSG_NOP) {
    s->total_received += 1;
  }
  /* do what the message says */
}

/* called by ring_cpu_send when we need space to send a message */
void ring_internal_gpu_receive(struct RingState *s)
{
  /* volatile */ struct RingMessage *msg = &(s->recvbuf[s->next_receive]);
  if (msg->header != MSG_IDLE) {
    s->peer_next_receive = msg->next_receive;
    ring_gpu_process_message(s, (struct RingMessage *) msg);
    msg->header = MSG_IDLE;
    s->next_receive = (s->next_receive + 1) % N;
  }
}


void ring_gpu_send_nop(struct RingState *s)
{
  do {
    ring_internal_gpu_receive(s);
  } while (ring_send_space_available(s) == 0);
  struct RingMessage msg;  // local composition of message
  msg.header = MSG_NOP;
  msg.next_receive = s->next_receive;
  s->peer_next_receive_sent = s->next_receive;

  s->sendbuf[s->next_send] = msg;  // how to do this?

  //  _movdir64b(&(s->sendbuf[s->next_send]), &msg);   // send message (atomic!)
  s->next_send = (s->next_send + 1) % N;
}

int ring_internal_gpu_send_nop_p(struct RingState *s)
{
  int cr_to_send =  (N + s->next_receive - s->peer_next_receive_sent) % N;
  return (cr_to_send > NOP_THRESHOLD);
}

void ring_gpu_poll(struct RingState *s)
{
  /* volatile */ struct RingMessage *msg = &(s->recvbuf[s->next_receive]);
  if (msg->header != MSG_IDLE) {
    s->peer_next_receive = msg->next_receive;
    ring_gpu_process_message(s, (struct RingMessage *) msg);
    msg->header = MSG_IDLE;
    s->next_receive = (s->next_receive + 1) % N;
    if (ring_internal_gpu_send_nop_p(s)) ring_gpu_send_nop(s);
  }
}

void ring_gpu_send(struct RingState *s, int type, int length, void *data)
{
  do {
    ring_internal_gpu_receive(s);
  } while (ring_send_space_available(s) == 0);
  struct RingMessage msg;  // local composition of message
  msg.header = type;
  msg.next_receive = s->next_receive;
  s->peer_next_receive_sent = s->next_receive;


  assert (length <= MaxData);

  memcpy(&msg.data, data, length);  // local copy
  s->sendbuf[s->next_send] = msg;   // send message (atomic!)
  s->next_send = (s->next_send + 1) % N;
}

void ring_gpu_drain(struct RingState *s)
{
  while (s->total_received < s->recv_count) ring_gpu_poll(s);
}

#define cpu_relax() asm volatile("rep; nop")
#define nullptr NULL

/* option codes */
#define CMD_CPU_TO_GPU_COUNT 1001
#define CMD_GPU_TO_CPU_COUNT 1002
#define CMD_SIZE 1003
#define CMD_HELP 1004
#define CMD_VALIDATE 1005
#define CMD_ATOMICSTORE 1019
#define CMD_ATOMICLOAD 1020
#define CMD_LATENCY 1021


void Usage()
{
  std::cout <<
    "--cpu_to_gpu_count <n>            set number of iterations\n"
    "--gpu_to_cpu_count <n>            set number of iterations\n"
    "--size <n>             set size\n"
    "--help                 usage message\n"
    "--validate             set and check data\n"
    "--atomicload=0/1 | --useatomicstore=0/1         method of flag access on device\n";
  std::cout << std::endl;
  exit(1);
}



void ProcessArgs(int argc, char **argv)
{
  const char* short_opts = "c:r:w:vhDHALM";
  const option long_opts[] = {
    {"cputogpucount", required_argument, nullptr, CMD_CPU_TO_GPU_COUNT},
    {"gputocpucount", required_argument, nullptr, CMD_GPU_TO_CPU_COUNT},
    {"size", required_argument, nullptr, CMD_SIZE},
    {"help", no_argument, nullptr, CMD_HELP},
    {"validate", no_argument, nullptr, CMD_VALIDATE},
    {"atomicload", required_argument, nullptr, CMD_ATOMICLOAD},
    {"atomicstore", required_argument, nullptr, CMD_ATOMICSTORE},
    {"latency", required_argument, nullptr, CMD_LATENCY},
    {nullptr, no_argument, nullptr, 0}
  };
  while (true)
    {
      const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);
      if (-1 == opt)
	break;
      switch (opt)
	{
	  case CMD_CPU_TO_GPU_COUNT: {
	  glob_cpu_to_gpu_count = std::stoi(optarg);
	  std::cout << "cputogpucount " << glob_cpu_to_gpu_count << std::endl;
	  break;
	}
	  case CMD_GPU_TO_CPU_COUNT: {
	  glob_gpu_to_cpu_count = std::stoi(optarg);
	  std::cout << "gputocpucount " << glob_gpu_to_cpu_count << std::endl;
	  break;
	}
	case CMD_SIZE: {
	  glob_size = std::stoi(optarg);
	  std::cout << "size " << glob_size << std::endl;
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
	case CMD_ATOMICSTORE: {
	  glob_use_atomic_store = std::stoi(optarg);
	  std::cout << "atomicstore " << glob_use_atomic_store << std::endl;
	  break;
	}
	case CMD_ATOMICLOAD: {
	  glob_use_atomic_load = std::stoi(optarg);
	  std::cout << "atomicload " << glob_use_atomic_load << std::endl;
	  break;
	}
	case CMD_LATENCY: {
	  glob_latency = std::stoi(optarg);
	  std::cout << "latency " << glob_latency << std::endl;
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

constexpr int cpuonly = 1;

void printstats(struct RingState *s)
{
  std::cout << s->name << " sent " << s->total_nop << " recv " << s->total_sent << " nop " << s->total_nop << std::endl;
}

int main(int argc, char *argv[]) {
  ProcessArgs(argc, argv);
  uint64_t loc_cpu_to_gpu_count = glob_cpu_to_gpu_count;
  uint64_t loc_gpu_to_cpu_count = glob_gpu_to_cpu_count;
  uint64_t loc_size = glob_size;
  int loc_validate = glob_validate;
  int loc_use_atomic_load = glob_use_atomic_load;
  int loc_use_atomic_store = glob_use_atomic_store;
  int loc_latency = glob_latency;
  
  sycl::property_list prop_list{sycl::property::queue::enable_profiling()};
  sycl::queue Q(sycl::gpu_selector{}, prop_list);
  std::cout<<"selected device : "<<Q.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout<<"device vendor : "<<Q.get_device().get_info<sycl::info::device::vendor>() << std::endl;

  // allocate device memory

  struct RingMessage *device_data_mem = (struct RingMessage *) sycl::aligned_alloc_device(4096, BUFSIZE * 2, Q);

  //allocate host mamory
  struct RingMessage *host_data_mem = (struct RingMessage *) sycl::aligned_alloc_host(4096, BUFSIZE, Q);
  std::cout << " host_data_mem " << host_data_mem << std::endl;

  //create mmap mapping of usm device memory on host
  sycl::context ctx = Q.get_context();

  struct RingMessage *device_data_hostmap;   // pointer for host to use


  device_data_hostmap = get_mmap_address(device_data_mem, BUFSIZE, Q);

  // initialize host mamory
  memset(&host_data_mem[0], 0, BUFSIZE);

  // initialize device memory  
  auto e = Q.submit([&](sycl::handler &h) {
      h.memcpy(device_data_mem, &host_data_mem[0], BUFSIZE);
    });
  e.wait_and_throw();
  printduration("memcpy kernel ", e);

  struct RingState *cpu = (struct RingState *) sycl::aligned_alloc_host(4096, sizeof(struct RingState), Q);
  struct RingState *gpu = (struct RingState *) sycl::aligned_alloc_device(4096, sizeof(struct RingState), Q);

  /* initialize state records */
  memset(cpu, 0, sizeof(struct RingState));
// initialize device memory  
  e = Q.submit([&](sycl::handler &h) {
      h.memcpy(gpu, cpu, sizeof(struct RingState));
    });
  e.wait_and_throw();
  initstate(cpu, device_data_hostmap, host_data_mem, loc_cpu_to_gpu_count, loc_gpu_to_cpu_count, "cpu");
  e = Q.submit([&](sycl::handler &h) {
      h.parallel_for_work_group(sycl::range(1), sycl::range(1), [=](sycl::group<1> grp) {
	  initstate(gpu, host_data_mem, device_data_mem, loc_gpu_to_cpu_count, loc_cpu_to_gpu_count, "gpu");
	});
    });
  e.wait_and_throw();
  
  

  std::cout << "kernel going to launch" << std::endl;
  unsigned long start_time, end_time;
  struct timespec ts_start, ts_end;
      clock_gettime(CLOCK_REALTIME, &ts_start);
      start_time = rdtsc();

  e = Q.submit([&](sycl::handler &h) {
      h.parallel_for_work_group(sycl::range(1), sycl::range(1), [=](sycl::group<1> grp) {
	  uint64_t msgdata[7] = {0xdeadbeef, 0, 0, 0, 0, 0, 0};
	  for (int i = 0; i < gpu->send_count; i += 1) {
	    msgdata[1] = i;
	    ring_gpu_send(gpu, MSG_NOP, loc_size, msgdata);
 	    if (loc_latency) {
	      while (gpu->total_received != (i+1)) {
		ring_gpu_poll(gpu);
	      }
	    }
	  }
	  ring_gpu_drain(gpu);
	});
    });

    {  // cpu code
      uint64_t msgdata[7] = {0xfeedface, 0, 0, 0, 0, 0, 0};
      for (int i = 0; i < cpu->send_count; i += 1) {
	msgdata[1] = i;
	ring_cpu_send(cpu, MSG_NOP, loc_size, msgdata);
	if (loc_latency) {
	  while (cpu->total_received != (i+1)) {
	    cpu_relax();
	    ring_cpu_poll(cpu);
	  }
	}
      }
      ring_cpu_drain(cpu);
    }
    
  e.wait_and_throw();
      clock_gettime(CLOCK_REALTIME, &ts_end);
      end_time = rdtsc();
  printduration("gpu kernel ", e);
    /* common cleanup */
    double elapsed = ((double) (ts_end.tv_sec - ts_start.tv_sec)) * 1000000000.0 +
	((double) (ts_end.tv_nsec - ts_start.tv_nsec));

    double fcount;
    if (loc_cpu_to_gpu_count > loc_gpu_to_cpu_count)
      fcount = loc_cpu_to_gpu_count;
    else
      fcount = loc_gpu_to_cpu_count;
    double size = loc_size;
    double mbps = (size * 1000) / (elapsed / fcount);
    double nsec = elapsed / fcount;

    std::cout << "elapsed " << elapsed << " fcount " << fcount << std::endl;
    std::cout << argv[0];
    std::cout << "--gpu_to_cpu_count " << loc_gpu_to_cpu_count << " ";
    std::cout << "--cpu_to_gpu_count " << loc_cpu_to_gpu_count << " ";
    std::cout << "--size " << loc_size << " ";
    std::cout << "  each " << nsec << " nsec " << mbps << "MB/s" << std::endl;
    printstats(cpu);
    printstats(gpu);

    munmap(device_data_hostmap, BUFSIZE);
    sycl::free(device_data_mem, Q);
    sycl::free(gpu, Q);
    return 0;
}

