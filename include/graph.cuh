#pragma once
#include <cuda_runtime.h>
#include <omp.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>

#include <array>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <numeric>  // accumulate
#include <shared_mutex>
#include <sstream>
#include <stack>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "clock.cuh"
#include "disk_graph.h"
#include "fileout.h"
#include "uni.h"

#define PREFETCH_THREADPOOL_SIZE 4
#define FREE_SLOT_UPDATE_SIZE 100000

namespace efanna2e {

inline void setBit(uint32_t* bitset, int bit_index) {
    bitset[bit_index >> 5] |= 1u << (bit_index & 31);
}
class Free_Slots {
   private:
    std::vector<int> free_slots;
    std::atomic<int> head = 0;

   public:
    void init(int size) {
        free_slots.resize(size);
        std::iota(free_slots.begin(), free_slots.end(), 0);
    }
    // inline void set_size(int size) { free_slots.assign(size, 0); }
    inline int size() { return free_slots.size() - head; }
    inline bool empty() { return free_slots.size() == head; }
    inline void head_increase(int num = 1) {
        head += num;
        if (head >= FREE_SLOT_UPDATE_SIZE) {
            std::vector<int> tmp(free_slots.begin() + head, free_slots.end());
            free_slots.swap(tmp);
            head = 0;
        }
    }
    void batch_pop(int num, std::vector<int>& slots_res) {
        if (!num) return;
        // std::vector<int> slots(free_slots.end() - num, free_slots.end());
        // free_slots.resize(free_slots.size() - num);
        slots_res.assign(free_slots.begin() + head, free_slots.begin() + head + num);
        head_increase(num);
    }
    int pop() {
        int slot = free_slots[head];
        head_increase();
        return slot;
    }
    inline void batch_push(std::vector<int>& slots) {
        free_slots.insert(free_slots.end(), slots.begin(), slots.end());
    }
    inline void push(const int slot) { free_slots.push_back(slot); }
};

class Slot_State {
   private:
    uint8_t ref_ = 0, pin_ = 0, dirty_ = 0;

   public:
    Slot_State() = default;

    inline void accumulate() { ref_++; }
    inline void release() {
        if (!(--ref_) && !pin_) pin_ = 1;
    }
    inline void to_free() {}
    inline void load() {
        if (!ref_) ref_ = 1;
    }
    inline void used() {
        ref_ = 0;
        pin_ = 1;
    }
    inline void free() { ref_ = pin_ = dirty_ = 0; }
    inline void write() {
        ref_ = 0;
        pin_ = dirty_ = 1;
    }
    inline bool ref() { return ref_; }
    inline void clean() { dirty_ = 0; }
    inline bool is_dirty() { return dirty_; }
    inline bool can_evict() {
        if (ref_)
            return false;
        else if (pin_) {
            pin_ = 0;
            return false;
        }
        return true;
    }
};

class Vec_Slot_State {
   private:
    uint8_t ref_ = 0, pin_ = 0;

   public:
    Vec_Slot_State() = default;
    inline void accumulate() { ref_++; }
    inline void release() {
        if (!(--ref_) && !pin_) pin_ = 1;
    }
    inline void load() { ref_ = 1; }
    inline void used() {
        ref_ = 0;
        pin_ = 1;
    }
    // inline bool ref() { return ref_; }
    inline bool can_evict() {
        if (ref_)
            return false;
        else if (pin_) {
            pin_ = 0;
            return false;
        }
        return true;
    }
};

struct FPSWorkspaceCPU {
    int N = 0;
    int max_threads = 1;

    // BFS state
    std::vector<int> visit_tag;  // 当前点最近一次被哪个 bfs_round 访问
    std::vector<int> hops;       // 仅当 visit_tag[i] == bfs_round 时有效
    int bfs_round = 0;

    // FPS state
    std::vector<int> min_hops;    // 到已选点集合的最小 hop，-1 表示目前仍不可达
    std::vector<uint8_t> chosen;  // 是否已被选中过

    // frontier buffers
    std::vector<int> frontier;
    std::vector<int> next_frontier;

    // thread-local buffers
    std::vector<std::vector<int>> local_bufs;
    std::vector<int> local_sizes;
    std::vector<int> local_offsets;

    explicit FPSWorkspaceCPU(int n = 0) { reset(n); }

    void reset(int n) {
        N = n;
#ifdef _OPENMP
        max_threads = omp_get_max_threads();
#else
        max_threads = 1;
#endif

        visit_tag.assign(N, 0);
        hops.resize(N);
        min_hops.assign(N, INT_MAX);
        chosen.assign(N, 0);

        frontier.resize(N);
        next_frontier.resize(N);

        local_bufs.clear();
        local_bufs.resize(max_threads);
        local_sizes.assign(max_threads, 0);
        local_offsets.assign(max_threads, 0);

        bfs_round = 0;
    }

    void clear_fps_state() {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            min_hops[i] = INT_MAX;
            chosen[i] = 0;
            visit_tag[i] = 0;
        }
        bfs_round = 0;
    }

    // 开启一轮新的 BFS
    int next_round() {
        ++bfs_round;
        if (bfs_round == INT_MAX) {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < N; ++i) visit_tag[i] = 0;
            bfs_round = 1;
        }
        return bfs_round;
    }
};

class Graph {
   public:
    Graph(const std::string base_path, const fpath& graph_dir_path, const float* h_base,
          int num_nodes, int max_degree, int dim, int k, DIST_METRIC m, Mode mode,
          GraphType graph_type)
        : search_start_ids_path(graph_dir_path / "graph.search_start_ids"),
          DS(base_path, graph_dir_path, num_nodes, dim, max_degree, mode == Mode::CONS),
          cpu_slot_size(mode == Mode::UPD ? 2 * num_nodes : std::min(CPU_N, num_nodes)),
          total_node_size(num_nodes),
          gpu_slot_size(mode == Mode::UPD ? 2 * num_nodes : std::min(GPU_N, num_nodes)),
          max_degree(max_degree),
          dim(dim),
          k(k),
          metric(m),
          cpu_slots(mode == Mode::UPD ? 2 * num_nodes : std::min(CPU_N, num_nodes)),
          h_base(h_base),
          mode(mode),
          gtype(graph_type) {
        fo.print("Create graph: num_nodes-" + TOS(num_nodes) + ", gpu_slot_size-" +
                 TOS(gpu_slot_size));
        if (num_nodes <= GPU_N && graph_type == GraphType::GPU) {
            CUDA_CHECK(cudaMalloc(&gpu_slots, sizeof(Slot) * gpu_slot_size));
            CUDA_CHECK(cudaMalloc(&gpu_vec_slots, sizeof(float) * (size_t)dim * gpu_slot_size));

            load_search_start_nodes();
            if (mode == Mode::CONS) {
#pragma omp parallel for
                for (int i = 0; i < cpu_slot_size; ++i) {
                    Slot& s = cpu_slots[i];
                    s.node_id = i;
                    s.deg = 0;
                }
            } else {
                std::vector<int> node_ids(total_node_size);
                std::iota(node_ids.begin(), node_ids.end(), 0);
                DS.load_batch_slots(node_ids, cpu_slots);  // 从磁盘加载所有的 graph slots
            }

            CUDA_CHECK(cudaMemcpy(gpu_slots, &(cpu_slots[0]), num_nodes * sizeof(Slot),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(gpu_vec_slots, h_base, (size_t)num_nodes * sizeof(float) * dim,
                                  cudaMemcpyHostToDevice));

            if (mode == Mode::UPD) {
                init_innbrs();                  // 找到每个 base 向量的入邻居向量
                new_node_id = total_node_size;  // 新节点的 id 从 host_num_nodes 开始
            }
        } else if (!MEM_MODE && mode != Mode::UPD) {
            load_search_start_nodes();
            if (mode == Mode::CONS) {
#pragma omp parallel for
                for (int i = 0; i < cpu_slot_size; ++i) cpu_slots[i].reset(i);
            } else {
                std::vector<int> node_ids(total_node_size);
                std::iota(node_ids.begin(), node_ids.end(), 0);
                DS.load_batch_slots(node_ids, cpu_slots);  // 从磁盘加载所有的 graph slots
            }

            GpuClockCache::Config config(num_nodes, dim, max_degree);
            gpu_cache = new GpuClockCache(config, cpu_slots, h_base);
        } else if (MEM_MODE && mode != Mode::UPD) {
            load_search_start_nodes();
            search_locks = std::make_unique<std::atomic_flag[]>(num_nodes);
            search_vec_locks = std::make_unique<std::atomic_flag[]>(num_nodes);
#pragma omp parallel for
            for (int i = 0; i < num_nodes; i++) {
                search_locks[i].clear();
                search_vec_locks[i].clear();
            }
            visited.assign((num_nodes + 31) / 32, 0);
            if (graph_type == GraphType::DISK) {
                cpu_free_slots.init(cpu_slot_size);
                cpu_free_vec_slots.init(cpu_slot_size);
                slot_states.resize(cpu_slot_size);
                vec_slot_states.resize(cpu_slot_size);
                cpu_vec_slots.resize((size_t)cpu_slot_size * dim);
            } else {
                if (mode == Mode::CONS) {
#pragma omp parallel for
                    for (int i = 0; i < cpu_slot_size; ++i) cpu_slots[i].reset(i);
                } else {
                    std::vector<int> node_ids(total_node_size);
                    std::iota(node_ids.begin(), node_ids.end(), 0);
                    DS.load_batch_slots(node_ids, cpu_slots);  // 从磁盘加载所有的 graph slots
                    if (mode == Mode::CE) {
                        noniso_num = 0;
                        for (int i = 0; i < total_node_size; i++)
                            if (cpu_slots[i].deg > 0) {
                                total_degree += cpu_slots[i].deg;
                                setBit(visited.data(), i);
                                noniso_num++;
                            }
                        fo.print("graph only CE iso_num: " + TOS(total_node_size - noniso_num));
                    }
                }
            }
        } else
            fo.eprint("Parameter ERROR for graph build");

        fo.print("Create graph done.");
    }

    ~Graph() {
        cudaDeviceSynchronize();

        if (gpu_slots) cudaFree(gpu_slots);
        if (gpu_vec_slots) cudaFree(gpu_vec_slots);
        if (gpu_cache) delete gpu_cache;
        if (ws) delete ws;

        search_free();
        enhance_free();
        if (ev) CUDA_CHECK(cudaEventDestroy(ev));
    }

    void LoadRoarGraph(const char* filename);

    const Slot& load_data_in_cpu(int id);
    const float* load_vec_in_cpu(int id);

    void load_data(const int* ids, int N, Slot* slots_out, float* vecs_out, bool need_vec,
                   bool need_nbr = true) {
        auto s = now_time();
        if (gtype == GraphType::CPU)
            load_data_cpu(ids, N, slots_out, vecs_out, need_vec, need_nbr);
        else
            load_data_disk(ids, N, slots_out, vecs_out, need_vec, need_nbr);
        static float total_time = 0.0f;
        static int cnt = 0;
        // total_time += time_diff(s);
        // if (++cnt == 100) {
        //     fo.print("load_data avg time: " + TOS(total_time / cnt) + "s");
        //     total_time = 0.0f;
        //     cnt = 0;
        // }
    }

    void load_search_data(const int* beam_ids, const float* d_queries, const int* d_qids,
                          int batch, int capacity, Slot* slots_out, int* h_slot_idxs,
                          float* d_dists2query_out, cudaStream_t st) {
        auto s = now_time();
        if (gtype == GraphType::CPU)
            load_search_data_cpu(beam_ids, d_queries, d_qids, batch, capacity, slots_out,
                                 h_slot_idxs, d_dists2query_out, st);
        else
            load_search_data_disk(beam_ids, d_queries, d_qids, batch, capacity, slots_out,
                                  h_slot_idxs, d_dists2query_out, st);
        // static float total_time = 0.0f;
        // static int cnt = 0;
        // total_time += time_diff(s);
        // if (++cnt == 10) {
        //     fo.print("load_data avg time: " + TOS(total_time / cnt) + "s");
        //     total_time = 0.0f;
        //     cnt = 0;
        // }
    }
    void load_search_data(const int* beam_ids, const float* d_queries, const int* d_qids,
                          int batch, int capacity, Slot* slots_out, float* h_vecs_out,
                          float* d_dists2query_out, cudaStream_t st, bool need_vec = false) {
        auto s = now_time();
        if (gtype == GraphType::CPU)
            load_search_data_cpu(beam_ids, d_queries, d_qids, batch, capacity, slots_out,
                                 h_vecs_out, d_dists2query_out, st, need_vec);
        else
            load_search_data_disk(beam_ids, d_queries, d_qids, batch, capacity, slots_out,
                                  h_vecs_out, d_dists2query_out, st, need_vec);
        // static float total_time = 0.0f;
        // static int cnt = 0;
        // total_time += time_diff(s);
        // if (++cnt == 10) {
        //     fo.print("load_data avg time: " + TOS(total_time / cnt) + "s");
        //     total_time = 0.0f;
        //     cnt = 0;
        // }
    }
    inline void load_vecs(const int* node_ids, int N, float* vecs_out) {
        if (gtype == GraphType::CPU)
            load_vecs_cpu(node_ids, N, vecs_out);
        else
            load_vecs_disk(node_ids, N, vecs_out);
    }
    inline void load_calc_vecs_dists(const float* d_queries, const int* node_ids, int batch, int k,
                                     float* d_vecs_out, float* d_dists_out, cudaStream_t st) {
        auto s = now_time();

        if (gtype == GraphType::CPU)
            load_calc_vecs_dists_cpu(d_queries, node_ids, batch, k, d_vecs_out, d_dists_out, st);
        else
            load_calc_vecs_dists_disk(d_queries, node_ids, batch, k, d_vecs_out, d_dists_out, st);

        static float total_time = 0.0f;
        static int cnt = 0;
        total_time += time_diff(s);
        if (++cnt == 1) {
            fo.print("load_calc_vecs_dists avg time: " + TOS(total_time / cnt) + "s");
            total_time = 0.0f;
            cnt = 0;
        }
    }
    inline void write_data(const Slot* slots, int N) {
        static int cnt = 0;
        if (gtype == GraphType::CPU)
            write_data_cpu(slots, N);
        else
            write_data_disk(slots, N);
        if (++cnt == 50) {
            fo.print("avg_deg=" + TOS(total_degree * 1.0 / total_node_size) +
                     ", noniso_num=" + TOS(noniso_num));
            cnt = 0;
        }
    }

    int load_isolated_nodes(std::vector<int>& iso_ids, int*& d_iso_ids, float*& d_iso_vecs,
                            bool need_vec, cudaStream_t stream);

    // void update_graph_data(const std::vector<int>& gpu_slot_idxs, const NBD* upd_data, int N,
    //                        cudaStream_t& stream);
    void mark_used_graph_data(const std::vector<int>& cpu_slot_idxs,
                              const std::vector<int>& gpu_slot_idxs);
    // void process_batch(const std::vector<int>& node_ids, const int* nbrs, const int* degs,
    //                    const float* dists);

    void get_del_innbrs(const std::vector<int>& node_ids, std::vector<int>& innbrs_out,
                        std::vector<int>& offsets);
    // void get_innbrs(const std::vector<int>& node_ids, std::vector<int>& innbrs_out,
    //                 std::vector<int>& del_ids_for_innbr_out, std::vector<int>& offsets);
    void delete_nodes(const int* del_ids, const int* innbr_ids, const int* old_nbrs,
                      const int* new_nbrs, int del_num, int innbrs_num);
    void insert_nodes(const float* h_ins_vecs, std::vector<int>& h_ins_ids_out, int ins_num);
    void update_innbrs_with_upd_nbrs(const int* pivot_ids, const int* upd_nbrs, int pivot_num);
    void update_innbrs_with_new_nbrs(const int* pivot_ids, const int* new_nbrs, int pivot_num);

    std::vector<int> get_search_start_nodes_fps(int M, int reset);

    int connect_enhance(const int* d_iso_ids, const int* d_new_nbrs, const float* d_new_nbr_dists,
                        int N, cudaStream_t stream, BP bp);
    void search_prepare(int capacity) {
        cap = capacity;
        search_free();
        if (ev == nullptr) CUDA_CHECK(cudaEventCreate(&ev));
        const size_t BATCH = 2 * blocks * tpb / capacity / max_degree,
                     slot_size = BATCH * capacity;
        CUDA_CHECK(cudaMallocHost(&h_slots, slot_size * sizeof(Slot)));
        CUDA_CHECK(cudaMallocHost(&h_qidxs, slot_size * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&h_nbr_vecs, slot_size * max_degree * dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_nbr_vecs, slot_size * max_degree * dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_qidxs, slot_size * sizeof(int)));
    }
    void enhance_prepare(int N) {
        enhance_free();
        CUDA_CHECK(cudaMallocHost(&h_vecs, (size_t)N * dim * sizeof(float)));
        // CUDA_CHECK(cudaMalloc(&d_vecs, N * sizeof(float)));
    }
    void enhance_free() {
        if (h_vecs) CUDA_CHECK(cudaFreeHost(h_vecs));
        if (d_vecs) CUDA_CHECK(cudaFree(d_vecs));
    }
    /**
     * @brief 保存构建的函数
     * 该函数用于执行保存操作，包括保存搜索起始节点和将所有数据刷新到磁盘
     */
    inline void save_build() {
        save_search_start_nodes();  // 保存搜索起始节点到磁盘
        flush_all_to_disk();
    }
    inline GraphType Type() { return gtype; }
    inline float* get_gpu_vec_slots() { return gpu_vec_slots; }
    inline Slot* get_gpu_slots() { return gpu_slots; }
    inline float* get_cpu_vec_slots() { return cpu_vec_slots.data(); }
    inline CPU_Big_Slots<Slot>& get_cpu_slots() { return cpu_slots; }
    inline GpuClockCache* GPUCache() { return gpu_cache; }
    inline ull* get_total_degree() { return &total_degree; }
    inline fpath get_graph_dir() { return DS.get_graph_dir(); }
    inline uint32_t* get_visited() { return visited.data(); }
    std::vector<int>& get_start_ids() { return start_ids; }

    inline void slot_lock(int id) {
        while (search_locks[id].test_and_set(std::memory_order_acquire)) {
            std::this_thread::yield();  // 或者 _mm_pause()
        }
    }
    inline void vec_lock(int id) {
        while (search_vec_locks[id].test_and_set(std::memory_order_acquire)) {
            std::this_thread::yield();  // 或者 _mm_pause()
        }
    }
    inline void slot_unlock(int id) {
        auto it = cpu_cache_dir.find(id);
        slot_states[it->second].release();
        search_locks[id].clear(std::memory_order_release);
    }
    inline void vec_unlock(int id) {
        auto it = cpu_cache_vec_dir.find(id);
        vec_slot_states[it->second].release();
        search_vec_locks[id].clear(std::memory_order_release);
    }

   private:
    Mode mode;
    fpath search_start_ids_path;
    GraphType gtype;
    DiskStorage DS;
    GpuClockCache* gpu_cache = nullptr;
    std::shared_mutex slot_mutex;  // mutex for the change of the whole node in slot
    std::shared_mutex data_mutex;  // mutex for the change of data of a node
    // Host-side authoritative directory: node_id -> slot_index (if present)
    std::unordered_map<int, int> cpu_cache_dir, cpu_cache_vec_dir;
    // int clock_hand = 0;
    // std::unordered_map<int, uint32_t> gpu_cache_dir, gpu_cache_vec_dir;

    // Host-side free slots list
    // std::stack<int> cpu_free_slots, cpu_free_vec_slots;  // 存在 CPU 中的空闲槽位
    // std::stack<int> gpu_free_slots, gpu_free_vec_slots;
    Free_Slots cpu_free_slots, cpu_free_vec_slots;  // 存在 CPU 中的空闲槽位
    // Free_Slots gpu_free_slots, gpu_free_vec_slots;

    std::vector<std::unordered_set<int>> h_innbrs;  // for update

    // Host graph store (pinned)
    CPU_Big_Slots<Slot> cpu_slots;  // size: host_num_nodes
    // std::vector<uint8_t> slot_pins, slot_refs;
    std::vector<Slot_State> slot_states;
    std::vector<Vec_Slot_State> vec_slot_states;
    // std::vector<Vec_Slot_State> cpu_vec_slot_states;
    // float* h_base_vecs;
    std::vector<float> cpu_vec_slots;
    const float* h_base = nullptr;

    // device pointers
    Slot* gpu_slots;
    // std::vector<Slot_State> gpu_slot_states;
    // std::vector<Slot_Access_Count> gpu_slot_access_counts;
    // std::vector<Vec_Slot_State> gpu_vec_slot_states;
    float* gpu_vec_slots;

    std::vector<int> start_ids;

    // std::vector<int> node_search_visited;  // for search

    int gpu_slot_size, cpu_slot_size, total_node_size, max_degree, dim, k;
    DIST_METRIC metric;
    // bool construct;

    std::atomic<int> new_node_id;

    int graph_update_time = 0;
    const int flush_time = 50;

    ull total_degree = 0, noniso_num = 0;

    std::vector<uint8_t> del_flags;

    std::vector<uint32_t> visited;

    cudaEvent_t ev = nullptr;

    // search prepare:
    Slot* h_slots = nullptr;
    int *h_qidxs = nullptr, *d_qidxs = nullptr;
    float *h_nbr_vecs = nullptr, *d_nbr_vecs = nullptr;
    const int blocks = 2048, tpb = 512;
    int cap;

    // enhance prepare:
    float *h_vecs = nullptr, *d_vecs = nullptr;

    std::unique_ptr<std::atomic_flag[]> search_locks;
    std::unique_ptr<std::atomic_flag[]> search_vec_locks;

    FPSWorkspaceCPU* ws = nullptr;

    // helper functions

    void search_free() {
        if (h_slots) CUDA_CHECK(cudaFreeHost(h_slots));
        if (h_qidxs) CUDA_CHECK(cudaFreeHost(h_qidxs));
        if (h_nbr_vecs) CUDA_CHECK(cudaFreeHost(h_nbr_vecs));
        if (d_nbr_vecs) CUDA_CHECK(cudaFree(d_nbr_vecs));
        if (d_qidxs) CUDA_CHECK(cudaFree(d_qidxs));
    }

    void bfs(int start, int* d_hops, int* d_frontier, int* d_next_frontier, int* d_fsize,
             int* d_nsize, int* h_fsize);
    void FPS(std::vector<int>& selected, int M);
    void FPS_cpu(std::vector<int>& selected, int M);
    int argmax_min_hops_cpu();
    void update_min_hops_cpu();
    void bfs_cpu(int start);
    void ensure_slots_for_nodes(const std::vector<int>& nodes, std::vector<int>& cpu_slots_res,
                                std::vector<int>& gpu_slots_res, cudaStream_t& stream);
    int prefetch_slot(const int node_id, cudaStream_t stream);
    void prefetch_slots(const std::vector<int>& node_ids, std::vector<int>& slots_res,
                        cudaStream_t stream);
    void prefetch_vecs(const std::vector<int>& node_ids, std::vector<int>& slots_res);
    // void evict_random_slots(int need, cudaStream_t& stream);
    // void evict_random_vec_slots(int need);
    void evict_random_cpu_slots(int need);
    void evict_random_cpu_vec_slots(int need);
    void init_innbrs();  // find in-neighbors for each node
    // void change_bucket(int slot);
    // void remove_from_bucket(int old_count, int slot);
    // void insert_into_bucket(int count, int slot);
    void write_back_all_gpu_slots();
    void write_back_all_cpu_slots();
    void flush_all_to_disk();
    void save_search_start_nodes();
    void load_search_start_nodes();
    void update_start_ids(const int* del_ids, int del_num) {
        start_ids.erase(std::remove_if(start_ids.begin(), start_ids.end(),
                                       [&](int id) {
                                           return std::find(del_ids, del_ids + del_num, id) !=
                                                  del_ids + del_num;
                                       }),
                        start_ids.end());
    }

    void write_data_cpu(const Slot* slots, int N);
    void write_data_disk(const Slot* slots, int N);
    void load_calc_vecs_dists_cpu(const float* d_queries, const int* node_ids, int batch, int k,
                                  float* d_vecs_out, float* d_dists_out, cudaStream_t st);
    void load_calc_vecs_dists_cpu(const float* d_queries, const int* node_ids,
                                  const int* h_offsets, const int* d_offsets, int batch,
                                  float* d_vecs_out, float* d_dists_out, cudaStream_t st);
    void load_calc_vecs_dists_disk(const float* d_queries, const int* node_ids, int batch, int k,
                                   float* d_vecs_out, float* d_dists_out, cudaStream_t st);
    void load_vecs_cpu(const int* node_ids, int N, float* vecs_out);
    void load_vecs_disk(const int* node_ids, int N, float* vecs_out);
    void calc_query_dists(const float* d_queries, const int* d_qids, int N, Slot* d_slots_out,
                          float* d_dists2query_out, cudaStream_t st);
    void load_search_data_cpu(const int* beam_ids, const float* d_queries, const int* d_qids,
                              int batch, int capacity, Slot* slots_out, int* h_slot_idxs,
                              float* d_dists2query_out, cudaStream_t st);
    void load_search_data_disk(const int* beam_ids, const float* d_queries, const int* d_qids,
                               int beam_batch, int capacity, Slot* d_slots_out, int* h_slot_idxs,
                               float* d_dists2query_out, cudaStream_t st);
    void load_search_data_cpu(const int* beam_ids, const float* d_queries, const int* d_qids,
                              int batch, int capacity, Slot* slots_out, float* h_vecs_out,
                              float* d_dists2query_out, cudaStream_t st, bool need_vec);
    void load_search_data_disk(const int* beam_ids, const float* d_queries, const int* d_qids,
                               int batch, int capacity, Slot* slots_out, float* h_vecs_out,
                               float* d_dists2query_out, cudaStream_t st, bool need_vec);
    void load_data_cpu(const int* ids, int N, Slot* slots_out, float* vecs_out, bool need_vec,
                       bool need_nbr);
    void load_data_disk(const int* ids, int N, Slot* slots_out, float* vecs_out, bool need_vec,
                        bool need_nbr);
    void inline load_miss_slots(const std::unordered_set<int>& miss) {
        const int miss_n = miss.size(), need = miss_n - cpu_free_slots.size();
        std::vector<int> free_slots, miss_ids(miss.begin(), miss.end());
        if (need > 0) evict_random_cpu_slots(need);
        cpu_free_slots.batch_pop(miss_n, free_slots);
        DS.load_slots_sync(miss_ids.data(), free_slots.data(), miss_n, cpu_slots);
        for (int i = 0; i < miss_n; i++) {
            int slot_idx = free_slots[i];
            cpu_cache_dir[miss_ids[i]] = slot_idx;
            slot_states[slot_idx].load();
        }
    }
    void inline load_miss_vecs(const std::unordered_set<int>& miss) {
        const int miss_vec_n = miss.size(), vec_need = miss_vec_n - cpu_free_vec_slots.size();
        std::vector<int> free_slots, miss_ids(miss.begin(), miss.end());
        if (vec_need > 0) evict_random_cpu_vec_slots(vec_need);
        cpu_free_vec_slots.batch_pop(miss_vec_n, free_slots);
        DS.load_vecs_sync(miss_ids.data(), free_slots.data(), miss_vec_n, cpu_vec_slots.data());
        for (int i = 0; i < miss_vec_n; i++) {
            int slot_idx = free_slots[i];
            cpu_cache_vec_dir[miss_ids[i]] = slot_idx;
            vec_slot_states[slot_idx].load();
        }
    }
};

};  // namespace efanna2e