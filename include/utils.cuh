#pragma once
#include <cuda_runtime.h>
#include <faiss/Clustering.h>
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/MetricType.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/index_io.h>
#include <float.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <roaring/roaring.hh>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "uni.h"

#ifndef UNROLL_FACTOR
#define UNROLL_FACTOR 4
#endif

namespace efanna2e {

#define UNROLL_FACTOR 4  // 根据 dim 调整，建议 4 或 8

#define BLOCK_BLOOM_BITS 2048        // 2048 bit = 256 B
#define BLOCK_BLOOM_U32 (2048 / 32)  // 64 uint32_t

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl;                   \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

#define NVML_CHECK(call)                                                                        \
    do {                                                                                        \
        nvmlReturn_t result = call;                                                             \
        if (result != NVML_SUCCESS) {                                                           \
            std::cerr << "NVML Error: " << nvmlErrorString(result) << " at " << __FILE__ << ":" \
                      << __LINE__ << std::endl;                                                 \
            exit(EXIT_FAILURE);                                                                 \
        }                                                                                       \
    } while (0)

typedef roaring::Roaring Roar;

struct Beam_Candidate_D {
    int id;
    float dist;
    __device__ __host__ Beam_Candidate_D() = default;
    __device__ __host__ Beam_Candidate_D(int id, float dist) : id(id), dist(dist) {}
    __device__ __host__ Beam_Candidate_D(const Beam_Candidate_D& other)
        : id(other.id), dist(other.dist) {}
    __device__ __host__ Beam_Candidate_D& operator=(const Beam_Candidate_D& other) {
        if (this != &other) {
            id = other.id;
            dist = other.dist;
        }
        return *this;
    }
    __device__ __host__ void swap(Beam_Candidate_D& other) {
        if (this != &other) {
            Beam_Candidate_D temp(other);
            other = *this;
            *this = temp;
        }
    }
};
using BCD = Beam_Candidate_D;

struct Beam_Data {
    int *d_beam_ids = nullptr, *d_ids = nullptr, *d_nums = nullptr;
    int *d_idxs = nullptr, *d_nbr_vec_idxs = nullptr;
    uint8_t* d_expanded = nullptr;
    int* d_beams_hash = nullptr;
    float *d_dists = nullptr, *d_farthests = nullptr;
    void resize(int batch, BP bp) {
        if (d_nums) cudaFree(d_nums);
        if (d_farthests) cudaFree(d_farthests);
        if (d_expanded) cudaFree(d_expanded);
        if (d_ids) cudaFree(d_ids);
        if (d_dists) cudaFree(d_dists);
        if (d_beam_ids) cudaFree(d_beam_ids);
        if (d_idxs) cudaFree(d_idxs);
        if (d_nbr_vec_idxs) cudaFree(d_nbr_vec_idxs);
        if (d_beams_hash) cudaFree(d_beams_hash);

        const size_t num1 = (size_t)batch * bp.beam_size, num2 = (size_t)batch * bp.beam_capacity;
        cudaMalloc(&d_nums, batch * sizeof(int));
        cudaMalloc(&d_farthests, batch * sizeof(float));
        cudaMalloc(&d_expanded, num2);
        cudaMalloc(&d_ids, num1 * sizeof(int));
        cudaMalloc(&d_dists, num1 * sizeof(float));
        cudaMalloc(&d_beam_ids, num2 * sizeof(int));
        cudaMalloc(&d_idxs, num2 * sizeof(int));
        cudaMalloc(&d_nbr_vec_idxs, num2 * bp.max_degree * sizeof(int));
        cudaMalloc(&d_beams_hash, (size_t)batch * bp.hash_n * sizeof(int));
    }
    ~Beam_Data() {
        if (d_nums) cudaFree(d_nums);
        if (d_farthests) cudaFree(d_farthests);
        if (d_expanded) cudaFree(d_expanded);
        if (d_ids) cudaFree(d_ids);
        if (d_dists) cudaFree(d_dists);
        if (d_beam_ids) cudaFree(d_beam_ids);
        if (d_idxs) cudaFree(d_idxs);
        if (d_nbr_vec_idxs) cudaFree(d_nbr_vec_idxs);
        if (d_beams_hash) cudaFree(d_beams_hash);
        // if (d_visited) cudaFree(d_visited);
    }
};

__device__ __forceinline__ void swap_if(float& a, float& b, bool cond) {
    if (cond) {
        float tmp = a;
        a = b;
        b = tmp;
    }
}

// 对 shared array s[0..n-1] 做 bitonic sort（n 必须是 2 的幂）
// m == L2：升序；false：降序
// __device__ __forceinline__ void bitonic_sort(BCD* s, int n, DIST_METRIC m) {
//     const int tid = threadIdx.x;
//     // k 控制 bitonic merge 的大段长度：2,4,8,...,n
//     for (int k = 2; k <= n; k <<= 1) {
//         // j 控制比较距离：k/2, k/4, ..., 1
//         for (int j = k >> 1; j > 0; j >>= 1) {
//             for (int i = tid; i < n; i += blockDim.x) {
//                 const int ixj = i ^ j;
//                 if (ixj > i) {
//                     // 这一段是升还是降，取决于 i 在当前 k 段内的位置
//                     bool dir = ((i & k) == 0);
//                     dir = m == DIST_METRIC::L2_ ? dir : !dir;
//                     const float ai = s[i], aj = s[ixj];
//                     // dir=true 表示升序：ai>aj 则交换
//                     // dir=false 表示降序：ai<aj 则交换
//                     if (dir ? (ai > aj) : (ai < aj)) {
//                         s[i] = aj;
//                         s[ixj] = ai;
//                     }
//                 }
//             }
//             __syncthreads();
//         }
//     }
// }

__device__ __forceinline__ uint32_t hash1(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ uint32_t hash2(uint32_t x) {
    x ^= x >> 17;
    x *= 0xed5ad4bb;
    x ^= x >> 11;
    x *= 0xac4c1b51;
    x ^= x >> 15;
    return x;
}

__device__ __forceinline__ bool beam_nocontains(const int* hash_beam, int key, uint32_t cap,
                                                uint32_t mask) {
    const uint32_t h = hash1((uint32_t)key) & mask;
    for (int i = 0; i < cap; ++i) {  // linear probing
        const int v = hash_beam[(h + i) & mask];
        if (v == -1) return true;  // 遇到空槽即可停
        if (v == key) return false;
    }
    return true;
}

// __global__ void compute_centroid_kernel(const float* __restrict__ data, float* center, int N,
//                                         int dim);
__global__ void normalize_centroid_kernel(float* center, int N, int dim);

__device__ int atomicTestAndSetBit(uint32_t* bitset, int bit_index);
// __device__ __forceinline__ void setBit(uint32_t* bitset, int bit_index) {
//     bitset[bit_index >> 5] |= (1u << (bit_index & 31));
// }
__device__ __forceinline__ int getBit(uint32_t* bitset, int bit_index) {
    return (bitset[bit_index >> 5] & (1u << (bit_index & 31))) != 0;
}
// bloom: uint32_t 数组
// bloom_bits: Bloom Filter 总 bit 数（必须是 32 的倍数）
// 返回值: 1 = 可能已访问过, 0 = 第一次访问
__device__ int atomicBloomTestAndSet(uint32_t* bloom, int bloom_bits, int key);
__device__ int atomicBloomTestAndSet3(uint32_t* bloom, uint32_t bloom_u32, int key);

__device__ float atomicMinFloat(float* address, float val);
__device__ float atomicMaxFloat(float* address, float val);
__device__ __forceinline__ float atomicClosestFloat(float* address, float val, BP bp) {
    return bp.metric == DIST_METRIC::IP_ ? atomicMaxFloat(address, val)
                                         : atomicMinFloat(address, val);
}

void prefix_exclusive_sum(int* d_array, int* d_prefixsum, int N, cudaStream_t& stream);
void prefix_exclusive_sum(int* d_array, int* d_prefixsum, int N);

void segmented_sort_pairs_inplace(BCD* targets, float* values, size_t* offsets,
                                  cudaStream_t& stream, int batch, int n);
void segmented_sort_pairs(BCD* targets, float* values, BCD* sort_targets, float* sort_values,
                          size_t* offsets, cudaStream_t stream, int batch, int n);
void segmented_sort_pairs(BCD* targets, float* values, BCD* sort_targets, float* sort_values,
                          size_t* starts, size_t* ends, cudaStream_t stream, int batch, int n);

void sort_pairs(float* d_keys, BCD* d_values, int n, cudaStream_t& stream);
void sort_pairs(float* d_keys, int* d_values, int n, cudaStream_t& stream);
void sort_pairs_desc(uint32_t* d_keys, int* d_values, int n);

std::pair<int*, float*> segmented_sort_pairs(int* targets, float* values, int* sort_targets,
                                             float* sort_values, size_t* offsets,
                                             cudaStream_t stream, int batch, int n);
void segmented_sort_pairs(CN* targets, float* values, CN* sort_targets, float* sort_values,
                          size_t* offsets, cudaStream_t stream, int batch, int n);
std::pair<CN*, float*> segmented_sort_pairs_nocpy(CN* targets, float* values, CN* sort_targets,
                                                  float* sort_values, size_t* offsets,
                                                  cudaStream_t stream, int batch, int n,
                                                  bool desc = false);
std::pair<CN*, float*> segmented_sort_pairs_nocpy_off(CN* targets, float* values, CN* sort_targets,
                                                      float* sort_values, const size_t* offsets,
                                                      cudaStream_t stream, int batch, int N);
// ======================= L2 distance =======================
__device__ __forceinline__ float l2_distance(const float* __restrict__ query,
                                             const float* __restrict__ base, int dim) {
    float sum = 0.0f;
    int i = 0;

    // 向量化处理（dim为4的倍数）
    for (; i <= dim - UNROLL_FACTOR; i += UNROLL_FACTOR) {
        float4 q_vec = reinterpret_cast<const float4*>(query)[i / 4];
        float4 b_vec = reinterpret_cast<const float4*>(base)[i / 4];

        sum += (q_vec.x - b_vec.x) * (q_vec.x - b_vec.x);
        sum += (q_vec.y - b_vec.y) * (q_vec.y - b_vec.y);
        sum += (q_vec.z - b_vec.z) * (q_vec.z - b_vec.z);
        sum += (q_vec.w - b_vec.w) * (q_vec.w - b_vec.w);
    }

    for (; i < dim; ++i) {
        float diff = query[i] - base[i];
        sum += diff * diff;
    }
    return sum;
}

__device__ __forceinline__ float l2_distance(const float* base, const int a_index,
                                             const int b_index, const int dim) {
    return l2_distance(base + (size_t)a_index * dim, base + (size_t)b_index * dim, dim);
}

__device__ __forceinline__ float l2_distance(const float* a, const float* b_base, int b_index,
                                             int dim) {
    return l2_distance(a, b_base + (size_t)b_index * dim, dim);
}

// ======================= Inner Product =======================
// __device__ __forceinline__ float ip_distance(const float* __restrict__ query,
//                                              const float* __restrict__ base, int dim) {
//     float sum = 0.0f;
//     int i = 0;

//     for (; i <= dim - 4; i += 4) {
//         float4 q_vec = reinterpret_cast<const float4*>(query)[i / 4];
//         float4 b_vec = reinterpret_cast<const float4*>(base)[i / 4];

//         sum += q_vec.x * b_vec.x;
//         sum += q_vec.y * b_vec.y;
//         sum += q_vec.z * b_vec.z;
//         sum += q_vec.w * b_vec.w;
//     }

//     for (; i < dim; ++i) sum += query[i] * base[i];

//     return sum;
// }
__device__ __forceinline__ float ip_distance(const float* __restrict__ q,
                                             const float* __restrict__ b, int dim) {
    float sum = 0.f;

#pragma unroll
    for (int i = 0; i < dim; i += 4) {
        float4 qv = *(const float4*)(q + i);
        float4 bv = *(const float4*)(b + i);

        sum = fmaf(qv.x, bv.x, sum);
        sum = fmaf(qv.y, bv.y, sum);
        sum = fmaf(qv.z, bv.z, sum);
        sum = fmaf(qv.w, bv.w, sum);
    }

    return sum;
}

__device__ __forceinline__ float ip_distance(const float* base, const int a_index,
                                             const int b_index, const int dim) {
    return ip_distance(base + (size_t)a_index * dim, base + (size_t)b_index * dim, dim);
}

__device__ __forceinline__ float ip_distance(const float* a, const float* b_base, int b_index,
                                             int dim) {
    return ip_distance(a, b_base + (size_t)b_index * dim, dim);
}

__device__ __forceinline__ float calc_distance(const float* a, const float* b_base, int b_index,
                                               BP bp) {
    return bp.metric == DIST_METRIC::L2_ ? l2_distance(a, b_base, b_index, bp.dim)
                                         : ip_distance(a, b_base, b_index, bp.dim);
}

__device__ __forceinline__ float calc_distance(const float* a, const float* b_base, int b_index,
                                               int dim, DIST_METRIC m) {
    return m == DIST_METRIC::L2_ ? l2_distance(a, b_base, b_index, dim)
                                 : ip_distance(a, b_base, b_index, dim);
}

__device__ __forceinline__ float calc_distance(const float* a, const float* b, BP bp) {
    return bp.metric == DIST_METRIC::L2_ ? l2_distance(a, b, bp.dim) : ip_distance(a, b, bp.dim);
}

__device__ __forceinline__ float calc_distance(const float* a, const float* b, int dim,
                                               DIST_METRIC m) {
    return m == DIST_METRIC::L2_ ? l2_distance(a, b, dim) : ip_distance(a, b, dim);
}

__device__ __forceinline__ float calc_distance(const float* base, const int a_index,
                                               const int b_index, BP bp) {
    return bp.metric == DIST_METRIC::L2_ ? l2_distance(base, a_index, b_index, bp.dim)
                                         : ip_distance(base, a_index, b_index, bp.dim);
}

__device__ __forceinline__ float calc_distance(const float* base, const int a_index,
                                               const int b_index, int dim, DIST_METRIC m) {
    return m == DIST_METRIC::L2_ ? l2_distance(base, a_index, b_index, dim)
                                 : ip_distance(base, a_index, b_index, dim);
}

__device__ __forceinline__ float farthest_dist_d(BP bp) {
    return bp.metric == DIST_METRIC::L2_ ? L2_FARTHEST : IP_FARTHEST;
}

__device__ __forceinline__ float closest_dist_d(BP bp) {
    return bp.metric == DIST_METRIC::L2_ ? L2_CLOSEST : IP_CLOSEST;
}

__device__ __forceinline__ float farthest_dist_d(DIST_METRIC m) {
    return m == DIST_METRIC::L2_ ? L2_FARTHEST : IP_FARTHEST;
}

__device__ __host__ __forceinline__ bool compare_dist(float d0, float d1, BP bp) {
    return bp.metric == DIST_METRIC::L2_ ? d0 > d1 : d0 < d1;  // false - d0 更近，true - d1 更近
}

__device__ __host__ __forceinline__ bool compare_dist_d(float d0, float d1, DIST_METRIC m) {
    return m == DIST_METRIC::L2_ ? d0 > d1 : d0 < d1;  // false - d0 更近，true - d1 更近
}

__device__ __forceinline__ bool is_bit_set(int x, int k) { return (x & (1 << k)) != 0; }

cudaEvent_t gpu_record_time_start(cudaStream_t stream);
float gpu_record_time_stop(cudaEvent_t event, cudaStream_t stream, std::string name = "");
cudaEvent_t gpu_record_time_reset(cudaEvent_t start, cudaStream_t stream, float& time,
                                  std::string name = "");

bool is_device_ptr(const void* p);

inline void cuda_check_last_error(const std::string& str = "") {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fo.print("Clearing previous CUDA error(" + str +
                 "): " + std::string(cudaGetErrorString(err)));
        // 可选的设备重置
        cudaDeviceReset();
    }
}

__device__ __forceinline__ void heap_sift_down(BCD* heap, int i, int num, BP bp) {
    while (true) {
        const int l = 2 * i + 1, r = l + 1;
        int worst = i;
        if (l < num && compare_dist(heap[l].dist, heap[worst].dist, bp)) worst = l;
        if (r < num && compare_dist(heap[r].dist, heap[worst].dist, bp)) worst = r;

        if (worst == i) break;
        heap[i].swap(heap[worst]);
        i = worst;
    }
}

__device__ __forceinline__ void heap_sift_down(int* ids, float* dists, int i, int num, BP bp) {
    while (true) {
        const int l = 2 * i + 1, r = l + 1;
        int worst = i;
        if (l < num && (compare_dist(dists[l], dists[worst], bp))) worst = l;
        if (r < num && (compare_dist(dists[r], dists[worst], bp))) worst = r;

        if (worst == i) break;
        const int tmp = ids[i];
        const float tmp_d = dists[i];
        ids[i] = ids[worst];
        dists[i] = dists[worst];
        ids[worst] = tmp;
        dists[worst] = tmp_d;

        i = worst;
    }
}

}  // namespace efanna2e