#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "disk_graph.h"
#include "fileout.h"
#include "uni.h"
#include "utils.cuh"

namespace efanna2e {
// device hash table markers
static constexpr int kEmptyKey = -1;
static constexpr int kTombKey = -2;

// ---------- Device hash table ----------
__device__ __forceinline__ uint32_t hash_u32(uint32_t x) {
    // cheap mix (xorshift/murmur-like)
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

struct DevHash {
    int* keys;  // [H]
    int* vals;  // [H]
    int H;      // power of 2 recommended
    int mask;   // H-1
    int max_probe;
};

struct EnsureResult {
    int uniq_n = 0;
    int uniq_vec_n = 0;
    int miss_n = 0;
    int miss_vec_n = 0;
};

__device__ __forceinline__ int hash_lookup(const DevHash& ht, int key) {
    // returns cache_idx or -1 if not found
    uint32_t h = hash_u32((uint32_t)key) & (uint32_t)ht.mask;
#pragma unroll 1
    for (int i = 0; i < ht.max_probe; ++i) {
        const int k = ht.keys[h];
        if (k == key) return ht.vals[h];
        if (k == kEmptyKey) return -1;  // stop on empty
        h = (h + 1) & ht.mask;
    }
    return -1;
}

__device__ __forceinline__ void hash_erase(const DevHash& ht, int key) {
    uint32_t h = hash_u32((uint32_t)key) & (uint32_t)ht.mask;
#pragma unroll 1
    for (int i = 0; i < ht.max_probe; ++i) {
        const int k = ht.keys[h];
        if (k == key) {  // mark tombstone; vals can be left as-is
            ht.keys[h] = kTombKey;
            return;
        }
        if (k == kEmptyKey) return;
        h = (h + 1) & ht.mask;
    }
}

struct DeviceView {
    Slot* d_slots;                // [N1]
    float* d_vecs;                // [N1*dim]
    uint8_t* d_dirty;             // [N1]
    uint32_t *d_pin, *d_vec_pin;  // [N1]
    uint8_t *d_ref, *d_vec_ref;   // [N1]
    int dim;
    DevHash ht, vec_ht;

    __device__ __forceinline__ int lookup(int node_id) const { return hash_lookup(ht, node_id); }
    __device__ __forceinline__ int vec_lookup(int node_id) const {
        return hash_lookup(vec_ht, node_id);
    }
};

class GpuClockCache {
   public:
    struct Config {
        int64_t N = 0;  // total nodes (CPU)
        int dim = 0, max_degree = MAX_DEGREE;
        int N1 = GPU_N;  // GPU cache capacity (#nodes)
        int hash_H = 0;  // hash table size (power of 2 >= ~1.4*N1 recommended)
        int max_probe = 64;
        int staging_cap = 1 << 16;   // max nodes to swap per batch (adjust)
        int cub_temp_bytes_cap = 0;  // 0 => auto
        // CLOCK tuning (adaptive still uses these as bounds)
        int oversub_init = 8, oversub_max = 128;
        int scan_blocks = 256, scan_threads = 256;  //  for clock_scan
        Config() = default;
        Config(int64_t N, int dim, int max_degree, int N1 = GPU_N, int hash_H = 0,
               int max_probe = 64, int staging_cap = 1 << 16)
            : N(N),
              dim(dim),
              max_degree(max_degree),
              N1(N1),
              hash_H(hash_H),
              max_probe(max_probe),
              staging_cap(staging_cap) {}
    };

    GpuClockCache(const Config& cfg, CPU_Big_Slots<Slot>& slots_cpu, const float* vecs_cpu)
        : cfg_(cfg), slots_cpu_(slots_cpu), vecs_cpu_(vecs_cpu) {
        if (cfg_.N1 <= 0 || cfg_.dim <= 0 || cfg_.N <= 0) fo.eprint("Bad config");
        if (cfg_.hash_H <= 0) {  // choose power-of-two >= 1.5*N1
            int H = 1;
            while (H < (int)(cfg_.N1 * 1.5)) H <<= 1;
            cfg_.hash_H = H;
        }
        if ((cfg_.hash_H & (cfg_.hash_H - 1)) != 0) fo.eprint("hash_H must be power of 2");
        if (cfg_.staging_cap <= 0) cfg_.staging_cap = 1 << 16;

        alloc_device();
        alloc_host_staging();
        init_device_structs();

        CUDA_CHECK(cudaStreamCreate(&stream_hash_));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_tmp_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_look_, cudaEventDisableTiming));
    }

    ~GpuClockCache() {
        cudaEventDestroy(ev_tmp_);
        cudaEventDestroy(ev_look_);
        cudaStreamDestroy(stream_hash_);
        free_host_staging();
        free_device();
    }

    DeviceView device_view() const {
        DeviceView v{};
        v.d_slots = d_slots_;
        v.d_vecs = d_vecs_;
        v.d_dirty = d_dirty_;
        v.d_pin = d_pin_;
        v.d_vec_pin = d_vec_pin_;
        v.dim = cfg_.dim;
        v.ht = ht_;
        v.vec_ht = vec_ht_;
        v.d_ref = d_ref_;
        v.d_vec_ref = d_vec_ref_;
        return v;
    }

    // void ensure_cached_from_device_ids(const int* d_node_ids_to_use, int n, cudaStream_t
    // stream);
    EnsureResult ensure_cached_from_device_ids(const int* d_node_ids_to_use, int n,
                                               uint32_t pin_id, bool need_vec,
                                               cudaStream_t stream);
    EnsureResult ensure_self_from_device_ids(const int* d_node_ids_to_use, int n, uint32_t pin_id,
                                             bool need_vec, cudaStream_t stream);
    EnsureResult ensure_vecs_from_device_ids(const int* d_node_ids_to_use, int n, uint32_t pin_id,
                                             cudaStream_t stream);
    void unpin_from_ids(const int* d_ids, int n, uint32_t pin_id, cudaStream_t stream);

    void lookup_all_data(const int* d_node_ids_to_use, int n, int* d_cache_idxs_out,
                         int* d_vec_cache_idxs_out, int* d_nbr_vec_cache_idxs_out,
                         cudaStream_t stream, bool need_vec);
    void lookup_slots_vecs(const int* d_node_ids_to_use, int n, int* d_cache_idxs_out,
                           int* d_vec_cache_idxs_out, cudaStream_t stream, bool need_vec);
    void lookup_vecs(const int* d_node_ids_to_use, int n, int* d_vec_cache_idxs_out,
                     cudaStream_t stream);
    void flush_all();

   private:
    struct SwapPlan {
        int n = 0;
        uint32_t pin_id = 0;
        int* victim_cache_idx;  // size n, cache_idx chosen
        int* victim_node_id;    // old node_id in that cache_idx (or -1)
        // uint8_t* victim_was_dirty;  // host snapshot of dirty
        int* in_node_id = nullptr;  // new node_id to load
        // staging buffers host pointers (pinned)
        Slot* h_slots_in = nullptr;   // [staging_cap]
        float* h_vecs_in = nullptr;   // [staging_cap*dim]
        Slot* h_slots_out = nullptr;  // [staging_cap]
        int* h_ids_out = nullptr;     // [staging_cap]
        // device staging buffers
        int* d_erase_keys = nullptr;      // [staging_cap]
        const int* d_ins_keys = nullptr;  // [staging_cap]
        int* d_ins_vals = nullptr;        // [staging_cap]
    };

    void alloc_device() {
        CUDA_CHECK(cudaMalloc(&d_slots_, sizeof(Slot) * cfg_.N1));
        CUDA_CHECK(cudaMalloc(&d_vecs_, sizeof(float) * (size_t)cfg_.N1 * (size_t)cfg_.dim));
        CUDA_CHECK(cudaMalloc(&d_vecs_id_, sizeof(int) * (size_t)cfg_.N1));
        CUDA_CHECK(cudaMalloc(&d_dirty_, cfg_.N1));
        CUDA_CHECK(cudaMalloc(&d_pin_, cfg_.N1 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_vec_pin_, cfg_.N1 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_ref_, cfg_.N1));
        CUDA_CHECK(cudaMalloc(&d_vec_ref_, cfg_.N1));
        // hash table
        CUDA_CHECK(cudaMalloc(&d_hash_keys_, sizeof(int) * cfg_.hash_H));
        CUDA_CHECK(cudaMalloc(&d_hash_vals_, sizeof(int) * cfg_.hash_H));
        ht_.keys = d_hash_keys_;
        ht_.vals = d_hash_vals_;
        CUDA_CHECK(cudaMalloc(&d_vec_hash_keys_, sizeof(int) * cfg_.hash_H));
        CUDA_CHECK(cudaMalloc(&d_vec_hash_vals_, sizeof(int) * cfg_.hash_H));
        vec_ht_.keys = d_vec_hash_keys_;
        vec_ht_.vals = d_vec_hash_vals_;
        ht_.H = vec_ht_.H = cfg_.hash_H;
        ht_.mask = vec_ht_.mask = cfg_.hash_H - 1;
        ht_.max_probe = vec_ht_.max_probe = cfg_.max_probe;

        // CUDA_CHECK(cudaMalloc(&plan_.d_erase_keys, sizeof(int) * cfg_.staging_cap));
        // CUDA_CHECK(cudaMalloc(&plan_.d_ins_keys, sizeof(int) * cfg_.staging_cap));
        CUDA_CHECK(cudaMalloc(&plan_.d_ins_vals, sizeof(int) * cfg_.staging_cap));
        // init dirty/pin to 0
        CUDA_CHECK(cudaMemset(d_dirty_, 0, cfg_.N1));
        CUDA_CHECK(cudaMemset(d_pin_, 0, cfg_.N1 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_vec_pin_, 0, cfg_.N1 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_ref_, 1, cfg_.N1));  // 初始都当作最近访问，避免冷启动大量淘汰抖动
        CUDA_CHECK(cudaMemset(d_vec_ref_, 1, cfg_.N1));
        // contig buffers for big memcpy
        CUDA_CHECK(cudaMalloc(&d_slots_gather_, sizeof(Slot) * cfg_.staging_cap));
        CUDA_CHECK(cudaMalloc(&d_slots_in_contig_, sizeof(Slot) * cfg_.staging_cap));
        CUDA_CHECK(cudaMalloc(&d_vecs_in_contig_,
                              sizeof(float) * (size_t)cfg_.staging_cap * (size_t)cfg_.dim));
        // CLOCK scan temp
        CUDA_CHECK(cudaMalloc(&d_clock_found_, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_clock_hand_, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_clock_hand_, 0, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_vec_clock_hand_, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_vec_clock_hand_, 0, sizeof(int)));
        // victim staging
        CUDA_CHECK(cudaMalloc(&d_victim_cache_idxs_, sizeof(int) * cfg_.staging_cap));
        CUDA_CHECK(cudaMalloc(&d_dirty_cache_idxs_, sizeof(int) * cfg_.staging_cap));
        CUDA_CHECK(cudaMalloc(&d_victim_node_ids_, sizeof(int) * cfg_.staging_cap));
        // CUDA_CHECK(cudaMalloc(&d_victim_dirty_, sizeof(uint8_t) * cfg_.staging_cap));
        CUDA_CHECK(cudaMalloc(&d_dirty_slot_idx_, sizeof(int)));
    }

    void free_device() {
        cudaFree(d_cub_temp_);
        cudaFree(d_ids_);
        cudaFree(d_ids_sorted_);
        cudaFree(d_ids_unique_);
        cudaFree(d_vec_ids_);
        cudaFree(d_vec_ids1_);
        cudaFree(d_unique_count_);
        cudaFree(d_cache_idxs_);
        cudaFree(d_miss_ids_tmp_);
        cudaFree(d_miss_flags_);
        cudaFree(d_miss_ids_);
        cudaFree(d_miss_count_);

        cudaFree(d_slots_);
        cudaFree(d_vecs_);
        cudaFree(d_vecs_id_);
        cudaFree(d_dirty_);
        cudaFree(d_pin_);
        cudaFree(d_vec_pin_);
        cudaFree(d_ref_);
        cudaFree(d_vec_ref_);

        cudaFree(d_hash_keys_);
        cudaFree(d_hash_vals_);
        cudaFree(d_vec_hash_keys_);
        cudaFree(d_vec_hash_vals_);

        // cudaFree(plan_.d_erase_keys);
        // cudaFree(plan_.d_ins_keys);
        cudaFree(plan_.d_ins_vals);

        cudaFree(d_slots_gather_);
        cudaFree(d_slots_in_contig_);
        cudaFree(d_vecs_in_contig_);

        cudaFree(d_clock_found_);
        cudaFree(d_clock_hand_);
        cudaFree(d_vec_clock_hand_);

        cudaFree(d_victim_cache_idxs_);
        cudaFree(d_dirty_cache_idxs_);
        cudaFree(d_victim_node_ids_);
        // cudaFree(d_victim_dirty_);
        cudaFree(d_dirty_slot_idx_);
    }

    void alloc_host_staging() {
        CUDA_CHECK(cudaMallocHost(&plan_.victim_cache_idx, sizeof(int) * cfg_.staging_cap));
        CUDA_CHECK(cudaMallocHost(&plan_.victim_node_id, sizeof(int) * cfg_.staging_cap));
        // CUDA_CHECK(cudaMallocHost(&plan_.victim_was_dirty, sizeof(uint8_t) * cfg_.staging_cap));
        CUDA_CHECK(cudaMallocHost(&plan_.in_node_id, sizeof(int) * cfg_.staging_cap));
        // pinned staging buffers
        CUDA_CHECK(cudaMallocHost(&plan_.h_slots_in, sizeof(Slot) * cfg_.staging_cap));
        CUDA_CHECK(cudaMallocHost(&plan_.h_vecs_in,
                                  sizeof(float) * (size_t)cfg_.staging_cap * (size_t)cfg_.dim));
        CUDA_CHECK(cudaMallocHost(&plan_.h_slots_out, sizeof(Slot) * cfg_.staging_cap));
        CUDA_CHECK(cudaMallocHost(&plan_.h_ids_out, sizeof(int) * cfg_.staging_cap));

        CUDA_CHECK(cudaMallocHost(&miss_n_host_, sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&uniq_n_host_, sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&found_host_, sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&dirty_slot_num_host_, sizeof(int)));

        host_node_id_of_cache_.assign(cfg_.N1, -1);
    }

    void free_host_staging() {
        cudaFreeHost(plan_.victim_cache_idx);
        cudaFreeHost(plan_.victim_node_id);
        // cudaFreeHost(plan_.victim_was_dirty);
        cudaFreeHost(plan_.in_node_id);
        cudaFreeHost(plan_.h_slots_in);
        cudaFreeHost(plan_.h_vecs_in);
        cudaFreeHost(plan_.h_slots_out);
        cudaFreeHost(plan_.h_ids_out);
        cudaFreeHost(miss_n_host_);
        cudaFreeHost(uniq_n_host_);
        cudaFreeHost(found_host_);
        cudaFreeHost(dirty_slot_num_host_);
    }

    void ensure_workspace(int n) {
        const int vec_n = n * (cfg_.max_degree + 1);
        if (workspace_cap_ >= n && workspace_cap1_ >= vec_n) return;
        // free old
        cudaFree(d_ids_);
        cudaFree(d_ids_sorted_);
        cudaFree(d_ids_unique_);
        cudaFree(d_vec_ids_);
        cudaFree(d_vec_ids1_);
        cudaFree(d_unique_count_);
        cudaFree(d_cache_idxs_);
        cudaFree(d_miss_ids_tmp_);
        cudaFree(d_miss_flags_);
        cudaFree(d_miss_ids_);
        cudaFree(d_miss_count_);

        workspace_cap_ = 1;
        while (workspace_cap_ < n) workspace_cap_ <<= 1;
        workspace_cap1_ = 1;
        while (workspace_cap1_ < vec_n) workspace_cap1_ <<= 1;

        CUDA_CHECK(cudaMalloc(&d_ids_, sizeof(int) * workspace_cap_));
        CUDA_CHECK(cudaMalloc(&d_ids_sorted_, sizeof(int) * workspace_cap_));
        CUDA_CHECK(cudaMalloc(&d_ids_unique_, sizeof(int) * workspace_cap_));
        CUDA_CHECK(cudaMalloc(&d_vec_ids_, sizeof(int) * workspace_cap1_));
        CUDA_CHECK(cudaMalloc(&d_vec_ids1_, sizeof(int) * workspace_cap1_));
        CUDA_CHECK(cudaMalloc(&d_unique_count_, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cache_idxs_, sizeof(int) * workspace_cap1_));
        CUDA_CHECK(cudaMalloc(&d_miss_ids_tmp_, sizeof(int) * workspace_cap1_));
        CUDA_CHECK(cudaMalloc(&d_miss_flags_, sizeof(uint8_t) * workspace_cap1_));
        CUDA_CHECK(cudaMalloc(&d_miss_ids_, sizeof(int) * workspace_cap1_));
        CUDA_CHECK(cudaMalloc(&d_miss_count_, sizeof(int)));
    }

    void alloc_cub_temp(size_t need_bytes) {
        if (need_bytes <= cub_temp_bytes_) return;
        cudaFree(d_cub_temp_);
        cub_temp_bytes_ = std::max<size_t>(need_bytes, (size_t)cfg_.cub_temp_bytes_cap);
        CUDA_CHECK(cudaMalloc(&d_cub_temp_, cub_temp_bytes_));
    }

    void init_device_structs();
    void execute_swap_in(cudaStream_t stream);
    void execute_vec_swap_in(cudaStream_t stream);
    void execute_hash_updates(DevHash& ht);
    void prepare_swap_plan_clock(const int* d_miss_ids, int miss_n, cudaStream_t stream_io);
    void prepare_vec_swap_plan_clock(const int* d_miss_ids, int miss_n, cudaStream_t stream_io);
    int get_miss_ids(const DevHash& ht, const int* d_ids, int n, uint8_t* d_ref, uint32_t* d_pin,
                     uint32_t pin_id, cudaStream_t stream);

   private:
    Config cfg_;
    CPU_Big_Slots<Slot>& slots_cpu_;
    const float* vecs_cpu_;

    // device cache arrays
    Slot* d_slots_ = nullptr;
    float* d_vecs_ = nullptr;
    int* d_vecs_id_ = nullptr;
    uint8_t* d_dirty_ = nullptr;
    uint32_t* d_pin_ = nullptr;
    uint32_t* d_vec_pin_ = nullptr;
    uint8_t* d_ref_ = nullptr;      // [N1]
    uint8_t* d_vec_ref_ = nullptr;  // [N1]

    // device hash table
    int* d_hash_keys_ = nullptr;
    int* d_hash_vals_ = nullptr;
    int* d_vec_hash_keys_ = nullptr;
    int* d_vec_hash_vals_ = nullptr;
    DevHash ht_{}, vec_ht_{};

    // CUB workspace
    void* d_cub_temp_ = nullptr;
    size_t cub_temp_bytes_ = 0;

    // per-batch workspace
    int workspace_cap_ = 0, workspace_cap1_ = 0;
    int* d_ids_ = nullptr;
    int* d_ids_sorted_ = nullptr;
    int* d_ids_unique_ = nullptr;
    int* d_vec_ids_ = nullptr;
    int* d_vec_ids1_ = nullptr;
    int* d_unique_count_ = nullptr;
    int* d_cache_idxs_ = nullptr;  // lookup results for uniq
    int* d_miss_ids_tmp_ = nullptr;
    uint8_t* d_miss_flags_ = nullptr;
    int* d_miss_ids_ = nullptr;
    int* d_miss_count_ = nullptr;

    // host pinned scalars
    int* uniq_n_host_ = nullptr;
    int* miss_n_host_ = nullptr;
    int* found_host_ = nullptr;

    SwapPlan plan_;  // swap plan

    std::vector<int> host_node_id_of_cache_;  // [N1] cache_idx -> node_id

    int* d_victim_cache_idxs_ = nullptr;  // [staging_cap]
    int* d_dirty_cache_idxs_ = nullptr;   // [staging_cap]
    int* d_victim_node_ids_ = nullptr;    // [staging_cap]
    // uint8_t* d_victim_dirty_ = nullptr;   // [staging_cap]
    int* d_dirty_slot_idx_ = nullptr;     // single int
    int* dirty_slot_num_host_ = nullptr;  // single int

    Slot* d_slots_gather_ = nullptr;     // [staging_cap]
    Slot* d_slots_in_contig_ = nullptr;  // [staging_cap]
    float* d_vecs_in_contig_ = nullptr;  // [staging_cap*dim]

    // CLOCK
    int* d_clock_found_ = nullptr;     // single int
    int* d_clock_hand_ = nullptr;      // single int
    int* d_vec_clock_hand_ = nullptr;  // single int

    cudaStream_t stream_hash_ = nullptr;
    cudaEvent_t ev_tmp_ = nullptr, ev_look_ = nullptr;
    // std::shared_mutex hash_mtx_;

    // std::vector<int> dirty_victim_cache_idxs_;
    // std::vector<int> dirty_victim_node_ids_;
};
};  // namespace efanna2e

// class HostLRU {
//    public:
//     HostLRU(int cap)
//         : cap_(cap), prev_(cap, -1), next_(cap, -1), in_(cap, 0), head_(-1), tail_(-1) {}

//     void insert(int idx) {  // insert at head
//         if (idx < 0) return;
//         if (in_[idx]) {
//             touch(idx);
//             return;
//         }
//         in_[idx] = 1;
//         prev_[idx] = -1;
//         next_[idx] = head_;
//         if (head_ >= 0) prev_[head_] = idx;
//         head_ = idx;
//         if (tail_ < 0) tail_ = idx;
//     }

//     void touch(int idx) {
//         if (idx < 0 || !in_[idx] || head_ == idx) return;
//         const int p = prev_[idx], n = next_[idx];
//         if (p >= 0) next_[p] = n;
//         if (n >= 0) prev_[n] = p;
//         if (tail_ == idx) tail_ = p;

//         prev_[idx] = -1;
//         next_[idx] = head_;
//         if (head_ >= 0) prev_[head_] = idx;
//         head_ = idx;
//         if (tail_ < 0) tail_ = idx;
//     }
//     // from tail; skip pinned ones by walking backward
//     int pop_victim(const std::vector<uint8_t>& pinned) {
//         int v = tail_;
//         while (v >= 0 && pinned[v]) v = prev_[v];
//         if (v < 0) return -1;
//         // remove v from list
//         int p = prev_[v], n = next_[v];
//         if (p >= 0) next_[p] = n;
//         if (n >= 0) prev_[n] = p;
//         if (head_ == v) head_ = n;
//         if (tail_ == v) tail_ = p;

//         prev_[v] = next_[v] = -1;
//         in_[v] = 0;
//         return v;
//     }

//    private:
//     int cap_;
//     // 双向链表存储 cache idx：
//     std::vector<int> prev_, next_;
//     std::vector<uint8_t> in_;  // in_ 为 1 表示该 cache idx 在 LRU 缓存中
//     int head_, tail_;
// };