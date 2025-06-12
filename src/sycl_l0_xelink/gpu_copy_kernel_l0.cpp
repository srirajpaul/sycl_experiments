#include <level_zero/ze_api.h>
#include <atomic>
#include <cassert>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <list>
#include <mutex>
#include <numeric>
#include <string>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

#define zeCall(myZeCall)                               \
    do {                                               \
        ze_result_t result = (myZeCall);            \
        if (result != ZE_RESULT_SUCCESS) {           \
            std::cout << "Error at "                   \
                      << #myZeCall << ": "             \
                      << __FUNCTION__ << ": "          \
                      << std::dec << __LINE__ << "\n"; \
            std::terminate();                          \
        }                                              \
    } while (0);

class rank_entry;

size_t num_threads = 4;
size_t buffer_count = 32 * 1024 * 1024;
int is_write = 1;
int use_copy = 0;
bool print_result_buffer = false;
bool verbose = false;
constexpr size_t warmup_iter = 15;
constexpr size_t max_bench_iter = 10000;
size_t bench_iter = max_bench_iter;

std::vector<std::thread> threads;
std::vector<rank_entry> ranks;

std::vector<void*> send_bufs;
std::vector<void*> recv_bufs;

std::atomic<size_t> barrier0;
std::atomic<size_t> barrier1;
std::atomic<size_t> barrier2;
std::atomic<size_t> barrier3;

ze_driver_handle_t driver;
std::vector<ze_device_handle_t> devices;

std::vector<std::vector<long double>> total_timers;

using ze_queue_properties_t = std::vector<ze_command_queue_group_properties_t>;

void get_num_queue_groups(ze_device_handle_t device, uint32_t* num) {
    *num = 0;
    zeCall(zeDeviceGetCommandQueueGroupProperties(device, num, nullptr));
    if (*num == 0) {
        throw std::runtime_error("No queue groups found");
    }
}

void get_queues_properties(ze_device_handle_t device,
                           uint32_t num_queue_groups,
                           ze_queue_properties_t* props) {
    props->resize(num_queue_groups);
    zeCall(zeDeviceGetCommandQueueGroupProperties(device, &num_queue_groups, props->data()));
}

void get_comp_queue_ordinal(ze_device_handle_t device,
                            const ze_queue_properties_t& props,
                            uint32_t* ordinal) {
    uint32_t comp_ordinal = std::numeric_limits<uint32_t>::max();

    for (uint32_t i = 0; i < props.size(); ++i) {
        if (props[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
            comp_ordinal = i;
            break;
        }
    }

    if (comp_ordinal != std::numeric_limits<uint32_t>::max()) {
        *ordinal = comp_ordinal;
    }
    else {
        throw std::runtime_error("could not find queue compute ordinal");
        *ordinal = 0;
    }
}

void get_copy_queue_ordinal(ze_device_handle_t device,
                            const ze_queue_properties_t& props,
                            uint32_t* ordinal) {
    uint32_t copy_ordinal = std::numeric_limits<uint32_t>::max();

    for (uint32_t i = 0; i < props.size(); ++i) {
        if (((props[i].flags &
                ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) == 0) &&
            (props[i].flags &
            ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) && props[i].numQueues >= 1) {
            copy_ordinal = i;
            break;
        }
    }

    if (copy_ordinal != std::numeric_limits<uint32_t>::max()) {
        *ordinal = copy_ordinal;
    }
    else {
        throw std::runtime_error("could not find queue copy ordinal");
        *ordinal = 0;
    }
}

size_t get_iter_count(size_t bytes, size_t max_iter_count) {
    size_t res = max_iter_count;

    size_t n = bytes >> 18;
    while (n) {
        res >>= 1;
        n >>= 1;
    }

    if (!res && max_iter_count) {
        res = 1;
    }

    return res * 2;
}

void print_buffer(size_t rank_id, const void* buf) {
    std::stringstream ss;
    ss << "rank " << rank_id << ":";
    for (size_t i = 0; i < buffer_count; ++i) {
        ss << " " << ((long*)buf)[i];
    }
    ss << std::endl;
    std::cout << ss.str();
}

class rank_entry {
public:
    rank_entry() = default;

    void init(size_t rank_id) {
        this->rank_id = rank_id;
        uint32_t dev_idx = rank_id % devices.size();
        this->device = devices.at(dev_idx);
        if (verbose) {
            printf("rank %zu use device %d\n", rank_id, dev_idx);
        }
        copy_lists.resize(ranks.size());

        ze_context_desc_t context_desc{};
        zeCall(zeContextCreate(driver, &context_desc, &context));

        uint32_t num_queue_groups;
        get_num_queue_groups(device, &num_queue_groups);
        get_queues_properties(device, num_queue_groups, &queue_props);
        get_copy_queue_ordinal(device, queue_props, &copy_ordinal);
        get_comp_queue_ordinal(device, queue_props, &comp_ordinal);

        ze_event_pool_desc_t pool_desc = {
            ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
            nullptr,
            ZE_EVENT_POOL_FLAG_HOST_VISIBLE | ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP,
            event_pool_size // count of events
        };
        zeCall(zeEventPoolCreate(context, &pool_desc, 0, nullptr, &event_pool));
        events.reserve(event_pool_size);

        buffer_bytes = buffer_count * sizeof(long);
        ze_host_mem_alloc_desc_t host_alloc_desc{};
        zeCall(zeMemAllocHost(context, &host_alloc_desc, buffer_bytes, 0, &host_send_buf));

        ze_device_mem_alloc_desc_t send_buf_alloc_desc{};
        zeCall(zeMemAllocDevice(context, &send_buf_alloc_desc, buffer_bytes, 0, device, &send_bufs.at(rank_id)));

        ze_device_mem_alloc_desc_t recv_buf_alloc_desc{};
        zeCall(zeMemAllocDevice(context, &recv_buf_alloc_desc, buffer_bytes * ranks.size(), 0, device, &recv_bufs.at(rank_id)));

        host_alloc_desc = {};
        zeCall(zeMemAllocHost(context, &host_alloc_desc, buffer_bytes * ranks.size(), 0, &host_recv_buf));

        createModule();
        ze_kernel_desc_t kernel_desc{};
        std::string kernel_name = "reduce_scatter_kernel_" + std::to_string(num_threads);
        kernel_desc.pKernelName = kernel_name.c_str();
        zeCall(zeKernelCreate(module, &kernel_desc, &kernel));
    }

    void run() {
        // init host buffer values
        for (size_t i = 0; i < buffer_count; ++i) {
            ((long*)host_send_buf)[i] = 23;
        }
        for (size_t i = 0; i < buffer_count * ranks.size(); ++i) {
            ((long*)host_recv_buf)[i] = 24;
        }

        // copy from my host buffer to my device buffer
        {
            void* src = host_send_buf;
            void* dst = send_bufs.at(rank_id);
            zeCall(zeCommandListAppendMemoryCopy(get_local_comp_list(), dst, src, buffer_bytes, nullptr, 0, nullptr));
            zeCall(zeCommandListClose(get_local_comp_list()));
            auto list = get_local_comp_list();
            zeCall(zeCommandQueueExecuteCommandLists(comp_queue_list.first, 1, &list, nullptr));
            zeCall(zeCommandQueueSynchronize(comp_queue_list.first, std::numeric_limits<uint64_t>::max()));
        }

        barrier1++;
        while(barrier1.load() != ranks.size());

        zeCall(zeCommandListReset(get_local_comp_list()));

        std::vector<const void*> peer_bufs;
        for (size_t i = 0; i < ranks.size(); i++) {
            void* ptr = is_write ? recv_bufs[i] : send_bufs[i];
            peer_bufs.push_back(ptr);
        }

        ze_event_handle_t kernel_event;

        if (!use_copy) {
            uint32_t groupSizeX = 1u;
            uint32_t groupSizeY = 1u;
            uint32_t groupSizeZ = 1u;

            ze_group_count_t threadGroupCount{};
            threadGroupCount.groupCountX = 1u;
            threadGroupCount.groupCountY = 1u;
            threadGroupCount.groupCountZ = 1u;

            zeCall(zeKernelSuggestGroupSize(kernel, buffer_count, 1u, 1u, &groupSizeX, &groupSizeY, &groupSizeZ));
            threadGroupCount.groupCountX = buffer_count / groupSizeX;

            zeCall(zeKernelSetGroupSize(kernel, groupSizeX, groupSizeY, groupSizeZ));
            for (size_t i = 0; i < peer_bufs.size(); i++) {
                zeCall(zeKernelSetArgumentValue(kernel, i, sizeof(peer_bufs[i]), &peer_bufs[i]));
            }
            void* local_buf = is_write ? send_bufs.at(rank_id) : recv_bufs.at(rank_id);
            zeCall(zeKernelSetArgumentValue(kernel, peer_bufs.size(), sizeof(local_buf), &local_buf));

            zeCall(zeKernelSetArgumentValue(kernel, peer_bufs.size()+1, sizeof(buffer_count), &buffer_count));

            zeCall(zeKernelSetArgumentValue(kernel, peer_bufs.size()+2, sizeof(is_write), &is_write));

            zeCall(zeKernelSetArgumentValue(kernel, peer_bufs.size()+3, sizeof(rank_id), &rank_id));

            kernel_event = create_event();
            zeCall(zeCommandListAppendLaunchKernel(get_local_comp_list(), kernel, &threadGroupCount, kernel_event,
                    0, nullptr));
        }
        else {
            ze_event_handle_t prev_event = nullptr;
            for (size_t i = 0; i < ranks.size(); i++) {
                kernel_event = create_event();
                if (is_write) {
                    zeCall(zeCommandListAppendMemoryCopy(get_local_copy_list(0), (long *)recv_bufs[i] + rank_id * buffer_count, send_bufs[rank_id], buffer_bytes, kernel_event,
                        i?1:0, i?&prev_event:nullptr));
                }
                else {
                    zeCall(zeCommandListAppendMemoryCopy(get_local_copy_list(0), (long *)recv_bufs.at(rank_id) + i * buffer_count, send_bufs.at(i), buffer_bytes, kernel_event,
                        i?1:0, i?&prev_event:nullptr));
                }
                prev_event = kernel_event;
            }
        }

        auto entry_event = kernel_event;

        for (auto& queue_list : execution_order) {
            auto list = queue_list->second;
            zeCall(zeCommandListClose(list));
        }
        zeCall(zeCommandListClose(get_local_comp_list()));

        // warmup
        for (size_t i = 0; i < warmup_iter; ++i) {
            execute();
            zeCall(zeEventHostSynchronize(entry_event, std::numeric_limits<uint64_t>::max()));
            reset_events();
        }

        for (size_t i = 0; i < bench_iter; ++i) {
            auto start_total_time = std::chrono::high_resolution_clock::now();
            execute();
            zeCall(zeEventHostSynchronize(entry_event, std::numeric_limits<uint64_t>::max()));
            auto end_total_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<long double, std::micro> total_time_span = end_total_time - start_total_time;
            total_timers.at(rank_id).push_back(total_time_span.count());
            reset_events();
        }

        barrier2++;
        while(barrier2.load() != ranks.size()) {};

        if (rank_id == 0) {
            print_timings();
        }

        // copy from my recv buffer to my result buffer
        void* src = recv_bufs.at(rank_id);
        void* dst = host_recv_buf;
        zeCall(zeCommandListReset(get_local_comp_list()));
        zeCall(zeCommandListAppendMemoryCopy(get_local_comp_list(), dst, src, buffer_bytes * ranks.size(), nullptr, 0, nullptr));
        zeCall(zeCommandListClose(get_local_comp_list()));
        auto list = get_local_comp_list();
        zeCall(zeCommandQueueExecuteCommandLists(comp_queue_list.first, 1, &list, nullptr));
        zeCall(zeCommandQueueSynchronize(comp_queue_list.first, std::numeric_limits<uint64_t>::max()));

        if (print_result_buffer) {
            print_buffer(rank_id, host_recv_buf);
        }
        validate_buffer();
    }

    void finalize() {
        zeCall(zeKernelDestroy(kernel));
        zeCall(zeModuleDestroy(module));
        for (auto& event : events) {
            zeCall(zeEventDestroy(event));
        }
        events.clear();
        zeCall(zeEventPoolDestroy(event_pool));
        for (auto& queue_list : execution_order) {
            auto queue = queue_list->first;
            auto list = queue_list->second;
            zeCall(zeCommandQueueDestroy(queue));
            zeCall(zeCommandListDestroy(list));
        }
        if (host_recv_buf) {
            zeCall(zeMemFree(context, host_recv_buf));
        }
        if (host_send_buf) {
            zeCall(zeMemFree(context, host_send_buf));
        }
        zeCall(zeContextDestroy(context));
    }

private:
    size_t rank_id;
    ze_device_handle_t device;
    ze_context_handle_t context;
    ze_module_handle_t module;
    ze_kernel_handle_t kernel;
    ze_event_pool_handle_t event_pool;
    uint32_t event_pool_size = 100;
    std::vector<ze_event_handle_t> events;
    size_t buffer_bytes;
    void *host_send_buf, *host_recv_buf;
    uint32_t comp_ordinal = 100;
    uint32_t copy_ordinal = 100;
    ze_queue_properties_t queue_props;

    using queue_list_t = typename std::pair<ze_command_queue_handle_t, ze_command_list_handle_t>;

    std::vector<std::unordered_map<uint32_t, queue_list_t>> copy_lists;

    ze_event_handle_t create_event() {
        if (events.size() >= event_pool_size) {
            throw std::runtime_error("limit of events");
        }
        ze_event_desc_t desc = {};
        desc.signal = ZE_EVENT_SCOPE_FLAG_DEVICE;
        desc.wait = ZE_EVENT_SCOPE_FLAG_DEVICE;
        desc.index = events.size();
        ze_event_handle_t event{};
        zeCall(zeEventCreate(event_pool, &desc, &event));
        events.push_back(event);
        return event;
    }

    void reset_events() {
        for (auto& event : events) {
            zeCall(zeEventHostReset(event));
        }
    }

    ze_command_list_handle_t create_list(ze_device_handle_t device, uint32_t ordinal) {
        ze_command_list_handle_t list{};
        ze_command_list_desc_t list_desc{};
        list_desc.commandQueueGroupOrdinal = ordinal;
        zeCall(zeCommandListCreate(context, device, &list_desc, &list));
        return list;
    }

    ze_command_queue_handle_t create_queue(ze_device_handle_t device, uint32_t ordinal, uint32_t queue_idx) {
        ze_command_queue_handle_t queue{};
        ze_command_queue_desc_t queue_desc{};
        queue_desc.ordinal = ordinal;
        queue_desc.index = queue_idx;
        queue_desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
        queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
        zeCall(zeCommandQueueCreate(context, device, &queue_desc, &queue));
        return queue;
    }

    queue_list_t create_queue_and_list(ze_device_handle_t device, uint32_t ordinal, uint32_t queue_idx) {
        ze_command_queue_handle_t queue = create_queue(device, ordinal, queue_idx);
        ze_command_list_handle_t list = create_list(device, ordinal);
        return std::make_pair(queue, list);
    }

    std::list<queue_list_t*> execution_order;
    queue_list_t comp_queue_list{};

    ze_command_list_handle_t get_copy_list(uint32_t rank_id, uint32_t list_idx) {
        auto& map = copy_lists[rank_id];

        uint32_t idx = list_idx % queue_props.at(copy_ordinal).numQueues;
        auto found_in_remote_map = map.find(idx);
        if (found_in_remote_map == map.end()) {
            uint32_t dev_idx = rank_id % devices.size();
            auto device = devices.at(dev_idx);
            if (verbose) {
                printf("rank %zu: create queue with ordinal %d index %d on device %d\n", this->rank_id, copy_ordinal, idx, dev_idx);
            }
            queue_list_t queue_list = create_queue_and_list(device, copy_ordinal, idx);
            map[idx] = queue_list;
            execution_order.push_back(&map[idx]);
            return map[idx].second;
        }
        else {
            return found_in_remote_map->second.second;
        }
    }

    ze_command_list_handle_t get_local_copy_list(uint32_t list_idx) {
        return get_copy_list(rank_id, list_idx);
    }

    ze_command_list_handle_t get_local_comp_list() {
        if (!comp_queue_list.second) {
            comp_queue_list = create_queue_and_list(device, 0, 0);
            execution_order.push_back(&comp_queue_list);
        }
        return comp_queue_list.second;
    }

    void execute() {
        for (auto& queue_list : execution_order) {
            auto queue = queue_list->first;
            auto list = queue_list->second;
            zeCall(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
        }
    }

    void createModule() {
        std::string kernelFile = "kernel.spv";
        ze_module_format_t kernelFormat = ZE_MODULE_FORMAT_IL_SPIRV;

        std::ifstream file(kernelFile, std::ios_base::in | std::ios_base::binary);
        if (false == file.good()) {
            std::cout << kernelFile << " file not found\n";
            std::terminate();
        }

        uint32_t spirvSize = 0;
        file.seekg(0, file.end);
        spirvSize = static_cast<size_t>(file.tellg());
        file.seekg(0, file.beg);

        auto spirvModule = std::make_unique<char[]>(spirvSize);
        file.read(spirvModule.get(), spirvSize);

        ze_module_desc_t moduleDesc = {};
        moduleDesc.format = kernelFormat;
        moduleDesc.pInputModule = reinterpret_cast<const uint8_t *>(spirvModule.get());
        moduleDesc.inputSize = spirvSize;
        zeCall(zeModuleCreate(context, device, &moduleDesc, &module, nullptr));
    }

    void print_timings() {
        std::vector<long double> total_timers_sum;
        for (const auto& timers : total_timers) {
            long double sum = std::accumulate(timers.begin(), timers.end(), 0);
            total_timers_sum.push_back(sum);
        }

        long double total_avg_time = std::accumulate(total_timers_sum.begin(), total_timers_sum.end(), 0);
        total_avg_time /= bench_iter * ranks.size();

        long double sum = 0;
        for (const long double& timer : total_timers_sum) {
            long double latency = timer / bench_iter;
            sum += (latency - total_avg_time) * (latency - total_avg_time);
        }
        long double stddev = std::sqrt(sum / ranks.size()) / total_avg_time * 100;

        printf("%zu,%zu,%zu,%Lf,%Lf\n",
            ranks.size(), bench_iter, buffer_count * sizeof(long), total_avg_time, stddev);
    }

    uint64_t adjust_device_timestamp(uint64_t timestamp, const ze_device_properties_t& props) {
        const uint64_t min_mask = std::min(props.kernelTimestampValidBits, props.timestampValidBits);
        const uint64_t mask = (1ull << min_mask) - 1ull;
        return (timestamp & mask) * props.timerResolution;
    }

    uint64_t calculate_global_time(ze_device_handle_t device) {
        uint64_t host_timestamp = 0;
        uint64_t device_timestamp = 0;
        zeCall(zeDeviceGetGlobalTimestamps(device, &host_timestamp, &device_timestamp));

        ze_device_properties_t device_props = {};
        device_props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        zeCall(zeDeviceGetProperties(device, &device_props));

        return adjust_device_timestamp(device_timestamp, device_props);
    }

    // returns start and end values for the provided event(measured in ns)
    std::pair<uint64_t, uint64_t> calculate_event_time(ze_event_handle_t event,
                                                    ze_device_handle_t device) {
        ze_kernel_timestamp_result_t timestamp = {};
        zeCall(zeEventQueryKernelTimestamp(event, &timestamp));

        ze_device_properties_t device_props = {};
        device_props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        zeCall(zeDeviceGetProperties(device, &device_props));

        // use global counter as we calculate value across different contexts
        uint64_t start = timestamp.global.kernelStart;
        uint64_t end = timestamp.global.kernelEnd;

        // gpu counters might be limited to 32-bit, so we need to handle a potential overlap
        if (end <= start) {
            uint64_t timestamp_max_value = (1LL << device_props.kernelTimestampValidBits) - 1;
            end += timestamp_max_value - start;
        }

        start = adjust_device_timestamp(start, device_props);
        end = adjust_device_timestamp(end, device_props);

        return { start, end };
    }

    void validate_buffer() {
        bool is_ok = true;
        for (size_t buf_pos = 0; buf_pos < buffer_count; ++buf_pos) {
            int exp = 23;
            int value = ((long*)host_recv_buf)[buf_pos];
            if (exp != value) {
                is_ok = false;
                printf("ERROR: unexpected value: rank %zu, buf_pos %zu, value %d, expected %d\n",
                    rank_id, buf_pos, value, exp);
                break;
            }
        }
        if (is_ok && verbose) {
            printf("rank %zu: validation passed\n", rank_id);
        }
    }
};

void thread_entry(size_t thread_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(thread_id, &cpuset);
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
    }

    ranks.at(thread_id).init(thread_id);
    barrier0++;
    while(barrier0.load() != num_threads);
    ranks.at(thread_id).run();
    barrier3++;
    while(barrier3.load() != num_threads);
    ranks.at(thread_id).finalize();
}

int main(int argc, char *argv[]) {
    if (argc > 1) {
        buffer_count = atoi(argv[1]);
    }
    if (argc > 2) {
        is_write = atoi(argv[2]);
    }
    if (argc > 3) {
        num_threads = atoi(argv[3]);
    }
    if (argc > 4) {
        use_copy = atoi(argv[4]);
    }

    bench_iter = get_iter_count(buffer_count * sizeof(long), max_bench_iter);
    if (verbose) {
        printf("buffer count %zu , is_write %d\n", buffer_count, is_write);
    }

    zeInit(ZE_INIT_FLAG_GPU_ONLY);

    if (verbose) {
        printf("number of ranks: %zu\n", num_threads);
    }

    uint32_t driver_count = 0;
    zeCall(zeDriverGet(&driver_count, nullptr));
    zeCall(zeDriverGet(&driver_count, &driver));

    uint32_t device_count = 0;
    zeCall(zeDeviceGet(driver, &device_count, nullptr));
    devices.resize(device_count);
    zeCall(zeDeviceGet(driver, &device_count, devices.data()));
    if (verbose) {
        printf("found %zu devices\n", devices.size());
    }

    ranks.resize(num_threads);
    send_bufs.resize(num_threads, nullptr);
    recv_bufs.resize(num_threads, nullptr);
    total_timers.resize(num_threads);
    std::vector<std::thread> threads;
    for (size_t i = 0; i < num_threads; ++i) {
        threads.emplace_back(thread_entry, i);
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

