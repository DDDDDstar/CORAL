#pragma once
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>

#include <array>
#include <cfloat>
#include <cmath>
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

#include "disk_graph.h"
#include "fileout.h"
#include "uni.h"

#define PREFETCH_THREADPOOL_SIZE 4

namespace efanna2e {

struct DiskSlot {
    int node_id, deg = 0;
    int nbrs[MAX_DEGREE];
    float dists[MAX_DEGREE];
};

struct DeviceSlot {
    int node_id, deg = 0;
    int nbrs[MAX_DEGREE];
    float dists[MAX_DEGREE];
};

class Node_Batch_Data {
   public:
    int *nbrs, *degs, n;
    float *dists, *nbrs_vec, *vecs;
    Node_Batch_Data(int n, int max_degree, int dim, cudaStream_t& stream);
    void self_copy(int batch, int single_n, int copy_n, int max_degree, int dim,
                   cudaStream_t& stream);
    ~Node_Batch_Data();
};
using NBD = Node_Batch_Data;

class Slot_State {
   private:
    std::atomic<unsigned int> wait_num, dirty;

   public:
    Slot_State() : wait_num(0), dirty(0) {};
    Slot_State(int wait_num) : wait_num(wait_num), dirty(0) {}
    void wait() { wait_num++; }
    inline void use() {
        if (wait_num > 0) wait_num--;
    }
    inline void free() {
        wait_num = 0;
        dirty = 0;
    }
    inline void mark_dirty() {
        dirty = 1;
        use();
    }
    inline void clean() { dirty = 0; }
    inline int is_dirty() { return dirty; }
    inline int get_wait_num() { return wait_num.load(); }
};

class Vec_Slot_State {
   private:
    std::atomic<unsigned int> wait_num;

   public:
    Vec_Slot_State() : wait_num(0) {};
    Vec_Slot_State(int wait_num) : wait_num(wait_num) {}
    void wait() { wait_num++; }
    inline void use() {
        if (wait_num > 0) wait_num--;
    }
    inline int get_wait_num() { return wait_num; }
    inline void free() { wait_num = 0; }
};

struct Slot_Access_Count {
    int access_count;
    int prev_slot;  // prev slot index in bucket
    int next_slot;  // next slot index in bucket
    Slot_Access_Count() : access_count(0), prev_slot(-1), next_slot(-1) {};
};

class Graph {
   public:
    Graph(const std::string& graph_file, float* h_base, int cache_capacity, int num_nodes,
          int max_degree, int dim, int k, bool construct);
    Graph(const std::string& base_file, const std::string& graph_file, int num_nodes,
          int cache_capacity, int max_degree, int dim, int k, bool construct);
    ~Graph();

    // process a batch of updates:
    // batch_node_ids: list of node ids in this batch
    // update_values: example payload per node (same length)
    const HostSlot* get_cpu_slot(int slot) { return cpu_slots + slot; }
    void load_data(const std::vector<int>& node_ids, std::vector<int>& cpu_slots_res,
                   std::vector<int>& gpu_slots_res, NBD* d_load_data, int N, cudaStream_t& stream);
    void update_graph_data(const std::vector<int>& gpu_slot_idxs, const NBD* upd_data, int N,
                           cudaStream_t& stream);
    void mark_used_graph_data(const std::vector<int>& cpu_slot_idxs,
                              const std::vector<int>& gpu_slot_idxs);
    // void process_batch(const std::vector<int>& node_ids, const int* nbrs, const int* degs,
    //                    const float* dists);

    // flush all dirty slots to dist (synchronous)
    void flush_all_to_disk();

    void get_innbrs(const std::vector<int>& node_ids, std::vector<int>& h_innbrs_out,
                    std::vector<int>& h_idxs_out);
    void delete_nodes(const std::vector<int>& del_ids);
    void insert_nodes(const float* h_ins_vecs, std::vector<int>& h_ins_ids_out, int ins_num);
    int get_top_M_access_nodes(std::vector<int>& res);

    void search_init();
    void load_beam_data_for_search(std::vector<int>& node_ids, std::vector<int>& cpu_slots_res,
                                   std::vector<int>& gpu_slots_res, NBD* d_load_data, int batch,
                                   int len, int start_flag, BP bp, cudaStream_t& stream);

   private:
    DiskStorage DS;
    std::shared_mutex slot_mutex;  // mutex for the change of the whole node in slot
    std::shared_mutex data_mutex;  // mutex for the change of data of a node
    // Host-side authoritative directory: node_id -> slot_index (if present)
    std::unordered_map<int, int> cpu_cache_dir, cpu_cache_vec_dir;
    std::unordered_map<int, int> gpu_cache_dir, gpu_cache_vec_dir;

    // Host-side free slots list
    std::stack<int> cpu_free_slots, cpu_free_vec_slots;  // 存在 CPU 中的空闲槽位
    std::stack<int> gpu_free_slots, gpu_free_vec_slots;

    std::unordered_map<int, std::vector<int>> h_innbrs;  // for update

    // Host graph store (pinned)
    HostSlot* cpu_slots;  // size: host_num_nodes
    std::vector<Slot_State> cpu_slot_states;
    // std::vector<Vec_Slot_State> cpu_vec_slot_states;
    // float* h_base_vecs;
    VecSlot* cpu_vec_slots;

    // device pointers
    DeviceSlot* gpu_slots;
    std::vector<Slot_State> gpu_slot_states;
    std::vector<Slot_Access_Count> gpu_slot_access_counts;
    // std::vector<Vec_Slot_State> gpu_vec_slot_states;
    VecSlot* gpu_vec_slots;

    std::vector<int> access_count_buckets;

    // std::vector<int> node_search_visited;  // for search

    int gpu_slot_size, cpu_slot_size, total_node_size, max_degree, dim, k;
    bool construct;

    int new_node_id;

    int graph_update_time = 0;
    const int flush_time = 50;

    std::atomic<int> total_degree = 0, noniso_num = 0;

    // helper functions
    void ensure_slots_for_nodes(const std::vector<int>& nodes, std::vector<int>& cpu_slots_res,
                                std::vector<int>& gpu_slots_res, cudaStream_t& stream);
    int prefetch_slot(const int node_id);
    VecSlot* prefetch_vec(const int node_id);
    void evict_random_slots(int need, cudaStream_t& stream);
    void evict_random_vec_slots(int need, cudaStream_t& stream);
    void evict_random_cpu_slots(int need);
    void evict_random_cpu_vec_slots(int need);
    void init_innbrs();  // find in-neighbors for each node
    void remove_from_bucket(int old_count, int slot);
    void insert_into_bucket(int count, int slot);
    void write_back_all_gpu_slots();
    void write_back_all_cpu_slots();
};

};  // namespace efanna2e