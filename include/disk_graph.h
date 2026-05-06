#pragma once
#include <fcntl.h>
#include <lz4.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <iostream>
#include <mutex>
#include <numeric>  // accumulate
#include <shared_mutex>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "fileout.h"
#include "uni.h"

#define PREFETCH_THREADPOOL_SIZE 4
#define CHECKPOINT_INTERVAL_SECONDS 30

namespace efanna2e {

struct alignas(16) Slot {
    int node_id = -1;
    int deg = 0;
    int nbrs[64];
    float dists[MAX_DEGREE];  // 到每个邻居的距离

    Slot() = default;
    Slot(const Slot& other) {
        node_id = other.node_id;
        deg = other.deg;
        memcpy(nbrs, other.nbrs, sizeof(int) * MAX_DEGREE * 2);
        memcpy(dists, other.dists, sizeof(float) * MAX_DEGREE);
    }
    inline void reset(int id) {
        node_id = id;
        deg = 0;
    }
    // void search_copy(const Slot& other) {
    //     node_id = other.node_id;
    //     deg = other.deg;
    //     memcpy(nbrs, other.nbrs, sizeof(int) * MAX_DEGREE * 2);
    // }
};

template <typename T>
struct CPU_Big_Slots {
    CPU_Big_Slots(size_t total_n, size_t block_n = GPU_N) : total_n(total_n), block_n(block_n) {
        // fo.print("Allocating " + TOS(total_n) + " slots for CPU big slots...");
        size_t remaining = total_n;
        try {
            while (remaining > 0) {
                size_t this_block_size = std::min(remaining, block_n);
                slots.emplace_back(std::make_unique<T[]>(this_block_size));
                remaining -= this_block_size;
                // fo.print("remaining: " + TOS(remaining));
            }
        } catch (const std::bad_alloc& e) {
            fo.eprint("Failed to allocate blocks for CPU big slots with remaining=" +
                      TOS(remaining) + ": " + std::string(e.what()));
        }
        // fo.iprint("Allocated " + TOS(slots.size()) + " blocks for CPU big slots");
    }
    inline T& operator[](size_t idx) { return slots[idx / block_n][idx % block_n]; }
    inline const T& operator[](size_t idx) const { return slots[idx / block_n][idx % block_n]; }
    std::vector<std::unique_ptr<T[]>> slots;
    size_t total_n, block_n;
};

enum DataPos { BASE, APPEND };

struct NodeDiskIndex {
    uint64_t offset;
    uint32_t size;
    DataPos pos = DataPos::BASE;
};

struct PartitionView {
    int data_fd;
    int append_fd;
    std::vector<NodeDiskIndex> index;
};

class DiskPartition {
   public:
    DiskPartition(const fpath& base_path, int pid, int partition_start_id,
                  int partition_node_count, int dim, bool construct);
    ~DiskPartition();
    void load_node(int global_node_id, Slot& slot_out);
    void write_node_append(int global_node_id, const Slot& slot, bool want_flush);
    void checkpoint() { persist_index(active_view.load()->index); }  // 供外部触发 checkpoint
    // 获取 partition 的 vec 读取偏移（vectors 单独放置为一个大文件）
    inline uint64_t vector_offset(int global_node_id) const {
        return (uint64_t)8 + (uint64_t)(global_node_id - start_id) * (uint64_t)vec_size_bytes;
    }
    // 导出索引表引用（只读）
    // inline const std::vector<NodeDiskIndex>& get_index_table() const { return index_table; }

   private:
    bool construct;
    fpath base_path;
    int partition_id;
    int start_id;
    int node_count;
    int dim;
    int vec_size_bytes;

    fpath data_path, idx_path, append_path;

    // int data_fd = -1;
    // int append_fd = -1;

    // std::vector<NodeDiskIndex> index_table;
    // std::shared_mutex part_mutex;
    std::atomic<PartitionView*> active_view;  // 当前读视图
    std::shared_mutex write_mutex;            // 只保护写 & swap

    // background
    std::thread compaction_thread;
    std::atomic<bool> stop_background;

    std::atomic<uint64_t> append_tail, write_time = 0;

    void persist_index(std::vector<NodeDiskIndex>& index_table);
    void load_index(std::vector<NodeDiskIndex>& index_table);
    void compaction_loop();
    void compact_once();

    /**
     * 检查给定的全局节点ID是否在当前分区中
     * @param global_node_id 要检查的全局节点ID
     * @return 如果节点在当前分区中返回true，否则返回false
     */
    inline bool node_in_partition(int global_node_id) const {  // 是否包含 node id
        return (global_node_id >= start_id) && (global_node_id < start_id + node_count);
    }
};

class DiskStorage {
   public:
    TaskExecutor executor;
    DiskStorage() : use(false) {}
    DiskStorage(const fpath& base_file, const fpath& graph_file, int node_num, int dim,
                int max_degree, bool construct);
    ~DiskStorage();
    inline bool is_use() { return use; }
    void load_node_slot(int node_id, Slot& slot_out);
    void load_node_vec(int node_id, float* vec_out);
    void write_vector(int node_id, const float* vec);
    void write_node_append(int node_id, const Slot& slot, bool want_flush = false);
    void write_batch_id(int batch_id);
    int read_batch_id();
    void checkpoint_all();
    void load_batch_slots(const std::vector<int>& node_ids, CPU_Big_Slots<Slot>& out_slots) {
#pragma omp parallel for
        for (size_t i = 0; i < node_ids.size(); ++i) load_node_slot(node_ids[i], out_slots[i]);
    }
    void load_batch_slots(const int* node_ids, const int* free_slots, int n,
                          CPU_Big_Slots<Slot>& cpu_slots) {
#pragma omp parallel for
        for (size_t i = 0; i < n; ++i) load_node_slot(node_ids[i], cpu_slots[free_slots[i]]);
    }
    void load_batch_vecs(const int* node_ids, const int* free_slots, int n, float* cpu_vecs) {
#pragma omp parallel for
        for (size_t i = 0; i < n; ++i) load_node_vec(node_ids[i], cpu_vecs + free_slots[i] * dim);
    }
    inline void load_slot_async(int node_id, Slot& slot_out) {
        executor.submit([this, node_id, &slot_out]() { load_node_slot(node_id, slot_out); });
    }
    void load_slots_sync(const int* node_ids, const int* free_slots, int n,
                         CPU_Big_Slots<Slot>& cpu_slots) {
        auto fut = executor.submit_with_future([this, node_ids, free_slots, n, &cpu_slots]() {
            load_batch_slots(node_ids, free_slots, n, cpu_slots);
        });
        fut.get();
    }
    void load_vecs_sync(const int* node_ids, const int* free_slots, int n, float* cpu_vecs) {
        auto fut = executor.submit_with_future([this, node_ids, free_slots, n, cpu_vecs]() {
            load_batch_vecs(node_ids, free_slots, n, cpu_vecs);
        });
        fut.get();
    }
    inline void write_slot_async(int node_id, const Slot& slot, bool want_flush = false) {
        Slot* slot_copy = new Slot(slot);
        executor.submit([this, node_id, slot_copy, want_flush]() {
            write_node_append(node_id, *slot_copy, want_flush);
            delete slot_copy;
        });
    }
    inline void load_vec_async(int node_id, float* vec_out) {
        executor.submit([this, node_id, vec_out]() { load_node_vec(node_id, vec_out); });
    }
    inline void load_sync() { executor.finish(); }
    // void prefetch_and_wait_vec(const std::vector<int>& node_ids, std::vector<float>& out_vecs);
    // void prefetch_slot_async(const std::vector<int>& node_ids,
    //                          std::function<void(bool, std::vector<Slot>)> callback);
    // void prefetch_vec_async(const std::vector<int>& node_ids,
    //                         std::function<void(bool, std::vector<float>)> callback);

    // 计算 partition id 与局部 id from global node id
    inline int get_partition_id(int node_id) const { return node_id / nodes_per_partition; }
    inline DiskPartition* get_partition(int node_id) {
        int pid = get_partition_id(node_id);
        if (pid < 0 || pid >= (int)partitions.size()) return nullptr;
        return partitions[pid];
    }
    inline fpath get_graph_dir() const { return base_path; }

   private:
    int vec_fd;
    fpath base_path, base_vec_file, batch_id_path;
    int node_num, dim, max_degree, partition_count, nodes_per_partition;
    bool construct, use;
    std::vector<DiskPartition*> partitions;
    std::thread checkpoint_thread;
    std::atomic<bool> stop_all, update_flag{false};
    std::mutex ckpt_mtx;
    std::condition_variable ckpt_cv;

    // prefetch threadpool
    std::mutex prefetch_vec_mutex, prefetch_slot_mutex;
    std::condition_variable prefetch_slot_cv, prefetch_vec_cv;
    std::vector<std::thread> prefetch_slot_threads, prefetch_vec_threads;
    std::vector<std::pair<std::vector<int>, std::function<void(bool, std::vector<Slot>)>>>
        prefetch_slot_queue;
    std::vector<std::pair<std::vector<int>, std::function<void(bool, std::vector<float>)>>>
        prefetch_vec_queue;

    void slot_prefetch_worker_loop();
    void vec_prefetch_worker_loop();
    void checkpoint_loop() {
        while (!stop_all) {
            // std::this_thread::sleep_for(std::chrono::seconds(CHECKPOINT_INTERVAL_SECONDS));
            // checkpoint_all();
            std::unique_lock<std::mutex> lk(ckpt_mtx);
            ckpt_cv.wait_for(lk, std::chrono::seconds(CHECKPOINT_INTERVAL_SECONDS));
            if (stop_all) break;
            if (update_flag.exchange(false, std::memory_order_acq_rel)) {
                checkpoint_all();
            }
        }
    }
};
};  // namespace efanna2e