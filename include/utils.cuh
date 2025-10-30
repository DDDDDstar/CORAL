#pragma once
#include <cuda_runtime.h>
#include <float.h>

#include "gpufuncs.cuh"

#ifndef UNROLL_FACTOR
#define UNROLL_FACTOR 4
#endif

namespace efanna2e {
using BCD = efanna2e::BCD;

__global__ void compute_centroid_kernel(const float *__restrict__ data, float *center, int N,
                                        int dim);
__global__ void normalize_centroid_kernel(float *center, int N, int dim);

__device__ int atomicTestAndSetBit(int *bitset, int bit_index);

__device__ float atomicMinFloat(float *address, float val);
__device__ float atomicMaxFloat(float *address, float val);
__device__ __forceinline__ float atomicClosestFloat(float *address, float val, BP bp) {
    return bp.metric == DIST_METRIC::IP ? atomicMaxFloat(address, val)
                                        : atomicMinFloat(address, val);
}

void prefix_exclusive_sum(int *d_array, int *d_prefixsum, int N, cudaStream_t &stream);

void segmented_sort_pairs_inplace(BCD *targets, float *values, int *offsets, cudaStream_t &stream,
                                  int batch, int n);

std::pair<int *, float *> segmented_sort_pairs(int *targets, float *values, int *sort_targets,
                                               float *sort_values, int *offsets,
                                               cudaStream_t &stream, int batch, int n);

// ======================= L2 distance =======================
__device__ __forceinline__ float l2_distance(const float *__restrict__ query,
                                             const float *__restrict__ base, int dim) {
    float sum = 0.0f;
    int i = 0;

    // 向量化处理（dim为4的倍数）
    for (; i <= dim - UNROLL_FACTOR; i += UNROLL_FACTOR) {
        float4 q_vec = reinterpret_cast<const float4 *>(query)[i / 4];
        float4 b_vec = __ldg(reinterpret_cast<const float4 *>(base) + i / 4);

        sum += (q_vec.x - b_vec.x) * (q_vec.x - b_vec.x);
        sum += (q_vec.y - b_vec.y) * (q_vec.y - b_vec.y);
        sum += (q_vec.z - b_vec.z) * (q_vec.z - b_vec.z);
        sum += (q_vec.w - b_vec.w) * (q_vec.w - b_vec.w);
    }

    for (; i < dim; ++i) {
        float diff = query[i] - __ldg(&base[i]);
        sum += diff * diff;
    }
    return sum;
}

__device__ __forceinline__ float l2_distance(const float *base, const int a_index,
                                             const int b_index, const int dim) {
    return l2_distance(base + a_index * dim, base + b_index * dim, dim);
}

__device__ __forceinline__ float l2_distance(const float *a, const float *b_base, int b_index,
                                             int dim) {
    return l2_distance(a, b_base + b_index * dim, dim);
}

// ======================= Inner Product =======================
__device__ __forceinline__ float ip_distance(const float *__restrict__ query,
                                             const float *__restrict__ base, int dim) {
    float sum = 0.0f;
    int i = 0;

    for (; i <= dim - 4; i += 4) {
        float4 q_vec = reinterpret_cast<const float4 *>(query)[i / 4];
        float4 b_vec = __ldg(reinterpret_cast<const float4 *>(base) + i / 4);

        sum += q_vec.x * b_vec.x;
        sum += q_vec.y * b_vec.y;
        sum += q_vec.z * b_vec.z;
        sum += q_vec.w * b_vec.w;
    }

    for (; i < dim; ++i) sum += query[i] * __ldg(&base[i]);

    return sum;
}

__device__ __forceinline__ float ip_distance(const float *base, const int a_index,
                                             const int b_index, const int dim) {
    return ip_distance(base + a_index * dim, base + b_index * dim, dim);
}

__device__ __forceinline__ float ip_distance(const float *a, const float *b_base, int b_index,
                                             int dim) {
    return ip_distance(a, b_base + b_index * dim, dim);
}

__device__ __forceinline__ float calc_distance(const float *a, const float *b_base, int b_index,
                                               BP bp) {
    return bp.metric == DIST_METRIC::L2 ? l2_distance(a, b_base, b_index, bp.dim)
                                        : ip_distance(a, b_base, b_index, bp.dim);
}

__device__ __forceinline__ float calc_distance(const float *a, const float *b, BP bp) {
    return bp.metric == DIST_METRIC::L2 ? l2_distance(a, b, bp.dim) : ip_distance(a, b, bp.dim);
}

__device__ __forceinline__ float calc_distance(const float *base, const int a_index,
                                               const int b_index, BP bp) {
    return bp.metric == DIST_METRIC::L2 ? l2_distance(base, a_index, b_index, bp.dim)
                                        : ip_distance(base, a_index, b_index, bp.dim);
}

__device__ __forceinline__ float farthest_dist_d(BP bp) {
    return bp.metric == DIST_METRIC::L2 ? L2_FARTHEST : IP_FARTHEST;
}

__device__ __forceinline__ bool compare_dist(float d0, float d1, BP bp) {
    return bp.metric == DIST_METRIC::L2 ? d0 > d1 : d0 < d1;  // false - d0 更近，true - d1 更近
}

__device__ __forceinline__ bool is_bit_set(int x, int k) { return (x & (1 << k)) != 0; }

}  // namespace efanna2e