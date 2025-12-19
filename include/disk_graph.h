#pragma once
#include <fcntl.h>
// #include <lz4.h>
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
#define CHECKPOINT_INTERVAL_SECONDS 5

namespace efanna2e {

struct HostSlot {
    int node_id, deg = 0;
    int nbrs[MAX_DEGREE];
    float dists[MAX_DEGREE];
};

class VecSlot {
   public:
    int node_id;
    float vec[DIM];
    VecSlot() = default;
    VecSlot(int node_id, const float* vec_) : node_id(node_id) {
        memcpy(vec, vec_, DIM * sizeof(float));
    }
    void copy(int node_id_, const float* vec_) {
        node_id = node_id_;
        memcpy(vec, vec_, DIM * sizeof(float));
    }
};

struct NodeDiskIndex {
    uint64_t offset;
    uint32_t size;
};

class DiskPartition {
   public:
    DiskPartition(const std::string& base_path, int pid, int partition_start_id,
                  int partition_node_count, int dim, bool construct);
    ~DiskPartition();
    void load_node(int global_node_id, HostSlot& slot_out);
    void write_node_append(int global_node_id, const HostSlot& slot, bool want_flush);
    void checkpoint() { persist_index(); }  // 供外部触发 checkpoint
    // 获取 partition 的 vec 读取偏移（vectors 单独放置为一个大文件）
    inline uint64_t vector_offset(int global_node_id) const {
        return (uint64_t)8 + (uint64_t)(global_node_id - start_id) * (uint64_t)vec_size_bytes;
    }
    // 导出索引表引用（只读）
    inline const std::vector<NodeDiskIndex>& get_index_table() const { return index_table; }

   private:
    bool construct;
    std::string base_path;
    int partition_id;
    int start_id;
    int node_count;
    int dim;
    int vec_size_bytes;

    std::string data_path, idx_path, append_path;

    int data_fd = -1;
    int append_fd = -1;

    std::vector<NodeDiskIndex> index_table;
    std::mutex part_mutex;

    // background
    std::thread compaction_thread;
    std::atomic<bool> stop_background;

    void persist_index();
    void load_index();
    void compaction_loop();
    void compact_once();

    inline bool node_in_partition(int global_node_id) const {  // 是否包含 node id
        return (global_node_id >= start_id) && (global_node_id < start_id + node_count);
    }
};

class DiskStorage {
   public:
    DiskStorage() : use(false) {}
    DiskStorage(const std::string& base_file, const std::string& graph_file, int node_num, int dim,
                int max_degree, bool construct);
    ~DiskStorage();
    inline bool is_use() { return use; }
    void load_node_slot(int node_id, HostSlot& slot_out);
    void load_node_vec(int node_id, VecSlot& vec_slot_out);
    void write_vector(int node_id, const float* vec);
    void write_node_append(int node_id, const HostSlot& slot, bool want_flush);
    void write_batch_id(int batch_id);
    int read_batch_id();
    void checkpoint_all();
    void load_batch_slots(const std::vector<int>& node_ids, HostSlot* out_slots);
    void prefetch_and_wait_vec(const std::vector<int>& node_ids, std::vector<float>& out_vecs);
    void prefetch_slot_async(const std::vector<int>& node_ids,
                             std::function<void(bool, std::vector<HostSlot>)> callback);
    void prefetch_vec_async(const std::vector<int>& node_ids,
                            std::function<void(bool, std::vector<float>)> callback);

    // 计算 partition id 与局部 id from global node id
    inline int get_partition_id(int node_id) const { return node_id / nodes_per_partition; }
    inline DiskPartition* get_partition(int node_id) {
        int pid = get_partition_id(node_id);
        if (pid < 0 || pid >= (int)partitions.size()) return nullptr;
        return partitions[pid];
    }

   private:
    std::string base_path, base_vec_file, batch_id_path;
    int node_num, dim, max_degree, partition_count, nodes_per_partition;
    bool construct, use;
    std::vector<int> node_offsets;
    std::vector<DiskPartition*> partitions;
    std::thread checkpoint_thread;
    std::atomic<bool> stop_all;

    // prefetch threadpool
    std::mutex prefetch_vec_mutex, prefetch_slot_mutex;
    std::condition_variable prefetch_slot_cv, prefetch_vec_cv;
    std::vector<std::thread> prefetch_slot_threads, prefetch_vec_threads;
    std::vector<std::pair<std::vector<int>, std::function<void(bool, std::vector<HostSlot>)>>>
        prefetch_slot_queue;
    std::vector<std::pair<std::vector<int>, std::function<void(bool, std::vector<float>)>>>
        prefetch_vec_queue;

    void slot_prefetch_worker_loop();
    void vec_prefetch_worker_loop();
    void checkpoint_loop() {
        while (!stop_all) {
            std::this_thread::sleep_for(std::chrono::seconds(CHECKPOINT_INTERVAL_SECONDS));
            checkpoint_all();
        }
    }
};

};  // namespace efanna2e