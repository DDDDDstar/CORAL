#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <cub/cub.cuh>

#include "gpufuncs.cuh"
#include "uni.h"
#include "utils.cuh"

namespace efanna2e {
// kernel: 每个线程处理一个向量，把它加到对应的中心累加数组
// __global__ void compute_centroid_kernel(const float* __restrict__ data, float* center, int N,
//                                         int dim) {
//     const int tid = threadIdx.x, bid = blockIdx.x, tpb = blockDim.x, global_tid = bid * tpb +
//     tid;

//     extern __shared__ float sh_center[];  // 每个 block 的共享内存，大小 = tpb * dim
//     for (int d = 0; d < dim; ++d) sh_center[tid * dim + d] = 0.0f;  // 初始化共享内存为 0
//     for (int i = global_tid; i < N; i += gridDim.x * tpb)           // 每个线程处理多个数据点
//         for (int d = 0; d < dim; ++d) sh_center[tid * dim + d] += data[i * dim + d];
//     __syncthreads();

//     for (int offset = tpb / 2; offset > 0;
//          offset >>= 1) {  // 使用归约将 block 内的向量累加到线程 0
//         if (tid < offset)
//             for (int d = 0; d < dim; ++d)
//                 sh_center[tid * dim + d] += sh_center[(tid + offset) * dim + d];
//         __syncthreads();
//     }

//     for (int d = tid; d < dim; d += tpb)  // block 内的线程将结果写回全局中心数组
//         atomicAdd(&center[d], sh_center[d]);
// }

// 单独 kernel 做归一化
__global__ void normalize_centroid_kernel(float* center, int N, int dim) {
    for (int d = threadIdx.x; d < dim; d += blockDim.x) center[d] /= N;
}

// bitset: 指向 int 数组；bit_index: 要访问的全局 bit 索引；返回值: 该 bit 的旧值 (0 or 1)
// __device__ int atomicTestAndSetBit(uint32_t* bitset, int bit_index) {
//     const int mask = 1 << (bit_index & 31);
//     uint32_t *addr = bitset + (bit_index >> 5), old_val = *addr, assumed;
//     while (true) {
//         assumed = old_val;
//         if (assumed & mask) return 1;  // 如果该位已经是 1，直接返回 1

//         old_val = atomicCAS(addr, assumed, assumed | mask);  // 尝试置位
//         if (old_val == assumed) return 0;  // CAS 成功，我们抢到了，把该位从 0 -> 1
//         // 否则 CAS 失败 ，别的线程改过了，因为不知道改的是不是当前位，所以继续循环判断
//     }
// }
// __device__ int atomicTestAndSetBit(uint32_t* bitset, int bit_index) {
//     assert(bit_index >= 0);
//     const uint32_t mask = 1u << (bit_index & 31);
//     uint32_t* addr = bitset + (bit_index >> 5);
//     const uint32_t old = atomicOr(addr, mask);
//     return (old & mask) != 0;
// }
// bloom: uint32_t 数组
// bloom_bits: Bloom Filter 总 bit 数（必须是 32 的倍数）
// 返回值: 1 = 可能已访问过, 0 = 第一次访问
__device__ int atomicBloomTestAndSet(uint32_t* bloom, int bloom_bits, int key) {
    const uint32_t h1 = hash1(key) % bloom_bits;
    const uint32_t h2 = hash2(key) % bloom_bits;

    const uint32_t mask1 = 1u << (h1 & 31);
    const uint32_t mask2 = 1u << (h2 & 31);

    uint32_t* addr1 = bloom + (h1 >> 5);
    uint32_t* addr2 = bloom + (h2 >> 5);

    // 先读（非原子，Bloom 允许竞态）
    const bool bit1 = (*addr1 & mask1);
    const bool bit2 = (*addr2 & mask2);

    // 原子置位（不关心旧值）
    atomicOr(addr1, mask1);
    atomicOr(addr2, mask2);

    // 如果两个 bit 之前都已经是 1，认为“访问过”
    return (bit1 & bit2);
}

__device__ __forceinline__ uint32_t hash_mix32(uint32_t x) {
    // MurmurHash3 finalizer
    x ^= x >> 16;
    x *= 0x85ebca6bu;
    x ^= x >> 13;
    x *= 0xc2b2ae35u;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ void bloom_index(uint32_t h, uint32_t bloom_u32, uint32_t& word_idx,
                                            uint32_t& bit_mask) {
    // 使用高位决定 word，低位决定 bit（彻底打散）
    uint32_t mixed = hash_mix32(h);

    word_idx = (mixed >> 5) % bloom_u32;  // word index
    uint32_t bit = mixed & 31;            // bit index in word
    bit_mask = 1u << bit;
}

__device__ int atomicBloomTestAndSet3(uint32_t* bloom, uint32_t bloom_u32, int key) {
    // 三个相互独立的 hash
    uint32_t h1 = hash_mix32((uint32_t)key * 0x9e3779b1u + 0x85ebca6bu);
    uint32_t h2 = hash_mix32((uint32_t)key * 0xc2b2ae35u + 0x27d4eb2fu);
    uint32_t h3 = hash_mix32(h1 ^ (h2 + 0x165667b1u));  // 非线性组合

    uint32_t w1, w2, w3;
    uint32_t m1, m2, m3;

    bloom_index(h1, bloom_u32, w1, m1);
    bloom_index(h2, bloom_u32, w2, m2);
    bloom_index(h3, bloom_u32, w3, m3);

    uint32_t* a1 = bloom + w1;
    uint32_t* a2 = bloom + w2;
    uint32_t* a3 = bloom + w3;

    int seen;

    // 合并 atomic（与你原来的逻辑一致）
    if (a1 == a2 && a2 == a3) {
        const uint32_t mask = m1 | m2 | m3;
        uint32_t old = atomicOr(a1, mask);
        seen = (old & mask) == mask;
    } else if (a1 == a2) {
        const uint32_t mask12 = m1 | m2;
        uint32_t old12 = atomicOr(a1, mask12);
        uint32_t old3 = atomicOr(a3, m3);
        seen = ((old12 & mask12) == mask12) && (old3 & m3);
    } else if (a1 == a3) {
        const uint32_t mask13 = m1 | m3;
        uint32_t old13 = atomicOr(a1, mask13);
        uint32_t old2 = atomicOr(a2, m2);
        seen = ((old13 & mask13) == mask13) && (old2 & m2);
    } else if (a2 == a3) {
        const uint32_t mask23 = m2 | m3;
        uint32_t old23 = atomicOr(a2, mask23);
        uint32_t old1 = atomicOr(a1, m1);
        seen = ((old23 & mask23) == mask23) && (old1 & m1);
    } else {
        uint32_t old1 = atomicOr(a1, m1);
        uint32_t old2 = atomicOr(a2, m2);
        uint32_t old3 = atomicOr(a3, m3);
        seen = (old1 & m1) && (old2 & m2) && (old3 & m3);
    }

    return seen;
}

__device__ float atomicMinFloat(float* address, float val) {
    int* addr_as_int = (int*)address;
    int old = *addr_as_int, assumed;

    do {
        assumed = old;
        if (__int_as_float(old) <= val) break;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(val));
    } while (assumed != old);

    return __int_as_float(old);
}

__device__ float atomicMaxFloat(float* address, float val) {
    int* addr_as_int = (int*)address;
    int old = *addr_as_int, assumed;

    do {
        assumed = old;
        if (__int_as_float(old) >= val) break;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(val));
    } while (assumed != old);

    return __int_as_float(old);
}

struct OffsetCalculator {
    int k;
    __host__ __device__ OffsetCalculator(int _k) : k(_k) {}
    __host__ __device__ int operator()(int i) const { return i * k; }
};

template <typename T1, typename T2>
std::pair<T1*, T2*> temp_segmented_sort_pairs_off(  // 模板函数
    T1* targets, T2* values, T1* sort_targets, T2* sort_values, const size_t* offsets,
    cudaStream_t& stream, int batch, int N) {
    void* temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    // 使用 CUB CachingDeviceAllocator 缓存临时内存分配:
    static cub::CachingDeviceAllocator g_allocator(true);

    cub::DoubleBuffer<T1> targets_buffer(targets, sort_targets);
    cub::DoubleBuffer<T2> values_buffer(values, sort_values);

    CUDA_CHECK(cub::DeviceSegmentedSort::SortPairs(temp_storage, temp_storage_bytes, values_buffer,
                                                   targets_buffer, N, batch, offsets, offsets + 1,
                                                   stream));

    CUDA_CHECK(g_allocator.DeviceAllocate(  // 分配临时空间内存
        &temp_storage, temp_storage_bytes, stream));

    CUDA_CHECK(cub::DeviceSegmentedSort::SortPairs(  // 第二次正式排序
        temp_storage, temp_storage_bytes, values_buffer, targets_buffer, N, batch, offsets,
        offsets + 1, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));         // 等待流中的排序操作完成
    CUDA_CHECK(g_allocator.DeviceFree(temp_storage));  // 安全释放临时存储

    return std::make_pair(targets_buffer.Current(), values_buffer.Current());
}

template <typename T1, typename T2>
std::pair<T1*, T2*> temp_segmented_sort_pairs(  // 模板函数
    T1* targets, T2* values, T1* sort_targets, T2* sort_values, size_t* offsets,
    cudaStream_t& stream, int batch, int n, bool desc) {
    // 使用 counting_iterator 快速生成 offsets，步长：n
    thrust::counting_iterator<int> count_iter(0);
    thrust::transform(thrust::cuda::par.on(stream), count_iter, count_iter + batch + 1,
                      thrust::device_pointer_cast(offsets), OffsetCalculator(n));

    void* temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    // 使用 CUB CachingDeviceAllocator 缓存临时内存分配:
    static cub::CachingDeviceAllocator g_allocator(true);

    cub::DoubleBuffer<T1> targets_buffer(targets, sort_targets);
    cub::DoubleBuffer<T2> values_buffer(values, sort_values);
    if (!desc) {
        CUDA_CHECK(cub::DeviceSegmentedSort::SortPairs(
            temp_storage, temp_storage_bytes, values_buffer, targets_buffer, (size_t)batch * n,
            batch, offsets, offsets + 1, stream));

        CUDA_CHECK(g_allocator.DeviceAllocate(  // 分配临时空间内存
            &temp_storage, temp_storage_bytes, stream));

        CUDA_CHECK(cub::DeviceSegmentedSort::SortPairs(  // 第二次正式排序
            temp_storage, temp_storage_bytes, values_buffer, targets_buffer, (size_t)batch * n,
            batch, offsets, offsets + 1, stream));
    } else {
        CUDA_CHECK(cub::DeviceSegmentedSort::SortPairsDescending(
            temp_storage, temp_storage_bytes, values_buffer, targets_buffer, (size_t)batch * n,
            batch, offsets, offsets + 1, stream));

        CUDA_CHECK(g_allocator.DeviceAllocate(  // 分配临时空间内存
            &temp_storage, temp_storage_bytes, stream));

        CUDA_CHECK(cub::DeviceSegmentedSort::SortPairsDescending(  // 第二次正式排序
            temp_storage, temp_storage_bytes, values_buffer, targets_buffer, (size_t)batch * n,
            batch, offsets, offsets + 1, stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));         // 等待流中的排序操作完成
    CUDA_CHECK(g_allocator.DeviceFree(temp_storage));  // 安全释放临时存储

    return std::make_pair(targets_buffer.Current(), values_buffer.Current());
}

template <typename T1, typename T2>
std::pair<T1*, T2*> temp_segmented_sort_pairs(  // 模板函数
    T1* targets, T2* values, T1* sort_targets, T2* sort_values, size_t* starts, size_t* ends,
    cudaStream_t& stream, int batch, int n) {
    void* temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    // 使用 CUB CachingDeviceAllocator 缓存临时内存分配:
    static cub::CachingDeviceAllocator g_allocator(true);

    cub::DoubleBuffer<T1> targets_buffer(targets, sort_targets);
    cub::DoubleBuffer<T2> values_buffer(values, sort_values);

    CUDA_CHECK(cub::DeviceSegmentedSort::SortPairs(temp_storage, temp_storage_bytes, values_buffer,
                                                   targets_buffer, (size_t)batch * n, batch,
                                                   starts, ends, stream));

    CUDA_CHECK(g_allocator.DeviceAllocate(  // 分配临时空间内存
        &temp_storage, temp_storage_bytes, stream));

    CUDA_CHECK(cub::DeviceSegmentedSort::SortPairs(  // 第二次正式排序
        temp_storage, temp_storage_bytes, values_buffer, targets_buffer, (size_t)batch * n, batch,
        starts, ends, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));         // 等待流中的排序操作完成
    CUDA_CHECK(g_allocator.DeviceFree(temp_storage));  // 安全释放临时存储

    return std::make_pair(targets_buffer.Current(), values_buffer.Current());
}

// void segmented_sort_pairs_inplace(BCD* targets, float* values, size_t* offsets,
//                                   cudaStream_t& stream, int batch, int n) {
//     BCD* targets_sort;
//     float* values_sort;
//     CUDA_CHECK(cudaMallocAsync(&targets_sort, (size_t)batch * n * sizeof(BCD), stream));
//     CUDA_CHECK(cudaMallocAsync(&values_sort, (size_t)batch * n * sizeof(float), stream));
//     auto res = temp_segmented_sort_pairs<BCD, float>(targets, values, targets_sort, values_sort,
//                                                      offsets, stream, batch, n);
//     CUDA_CHECK(
//         cudaMemcpyAsync(targets, res.first, n * sizeof(BCD), cudaMemcpyDeviceToDevice, stream));
//     CUDA_CHECK(cudaFreeAsync(targets_sort, stream));
//     CUDA_CHECK(cudaFreeAsync(values_sort, stream));
// }

void sort_pairs(float* d_keys, BCD* d_values, int n, cudaStream_t& stream) {
    thrust::device_ptr<float> keys_ptr(d_keys);
    thrust::device_ptr<BCD> values_ptr(d_values);
    thrust::sort_by_key(thrust::cuda::par.on(stream), keys_ptr, keys_ptr + n, values_ptr);
}

void sort_pairs(float* d_keys, int* d_values, int n, cudaStream_t& stream) {
    thrust::device_ptr<float> keys_ptr(d_keys);
    thrust::device_ptr<int> values_ptr(d_values);
    thrust::sort_by_key(thrust::cuda::par.on(stream), keys_ptr, keys_ptr + n, values_ptr);
}

void sort_pairs_desc(uint32_t* d_keys, int* d_values, int n) {
    thrust::device_ptr<uint32_t> keys_ptr(d_keys);
    thrust::device_ptr<int> values_ptr(d_values);
    thrust::sort_by_key(keys_ptr, keys_ptr + n, values_ptr, thrust::greater<uint32_t>());
}

// std::pair<int*, float*> segmented_sort_pairs(int* targets, float* values, int* sort_targets,
//                                              float* sort_values, size_t* offsets,
//                                              cudaStream_t stream, int batch, int n) {
//     return temp_segmented_sort_pairs<int, float>(targets, values, sort_targets, sort_values,
//                                                  offsets, stream, batch, n);
// }

// void segmented_sort_pairs(CN* targets, float* values, CN* sort_targets, float* sort_values,
//                           size_t* offsets, cudaStream_t stream, int batch, int n) {
//     auto res = temp_segmented_sort_pairs<CN, float>(targets, values, sort_targets, sort_values,
//                                                     offsets, stream, batch, n);
//     CUDA_CHECK(cudaMemcpyAsync(targets, res.first, (size_t)batch * n * sizeof(CN),
//                                cudaMemcpyDeviceToDevice, stream));
// }

std::pair<CN*, float*> segmented_sort_pairs_nocpy(CN* targets, float* values, CN* sort_targets,
                                                  float* sort_values, size_t* offsets,
                                                  cudaStream_t stream, int batch, int n,
                                                  bool desc) {
    return temp_segmented_sort_pairs<CN, float>(targets, values, sort_targets, sort_values,
                                                offsets, stream, batch, n, desc);
}

std::pair<CN*, float*> segmented_sort_pairs_nocpy_off(CN* targets, float* values, CN* sort_targets,
                                                      float* sort_values, const size_t* offsets,
                                                      cudaStream_t stream, int batch, int N) {
    return temp_segmented_sort_pairs_off<CN, float>(targets, values, sort_targets, sort_values,
                                                    offsets, stream, batch, N);
}

// void segmented_sort_pairs(BCD* targets, float* values, BCD* sort_targets, float* sort_values,
//                           size_t* offsets, cudaStream_t stream, int batch, int n) {
//     auto res = temp_segmented_sort_pairs<BCD, float>(targets, values, sort_targets, sort_values,
//                                                      offsets, stream, batch, n);
//     CUDA_CHECK(cudaMemcpyAsync(targets, res.first, (size_t)batch * n * sizeof(BCD),
//                                cudaMemcpyDeviceToDevice, stream));
//     CUDA_CHECK(cudaMemcpyAsync(values, res.second, (size_t)batch * n * sizeof(float),
//                                cudaMemcpyDeviceToDevice, stream));
// }

void segmented_sort_pairs(BCD* targets, float* values, BCD* sort_targets, float* sort_values,
                          size_t* starts, size_t* ends, cudaStream_t stream, int batch, int n) {
    auto res = temp_segmented_sort_pairs<BCD, float>(targets, values, sort_targets, sort_values,
                                                     starts, ends, stream, batch, n);
    CUDA_CHECK(cudaMemcpyAsync(targets, res.first, (size_t)batch * n * sizeof(BCD),
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(values, res.second, (size_t)batch * n * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream));
}

void prefix_exclusive_sum(int* d_array, int* d_prefixsum, int N, cudaStream_t& stream) {
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    CUDA_CHECK(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_array,
                                             d_prefixsum, N,
                                             stream));  // 第一次调用获取临时存储大小
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_array,
                                             d_prefixsum, N, stream));
}

void prefix_exclusive_sum(int* d_array, int* d_prefixsum, int N) {
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    CUDA_CHECK(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_array,
                                             d_prefixsum, N));  // 第一次调用获取临时存储大小
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_array,
                                             d_prefixsum, N));  // 第二次调用执行 prefix sum
}

cudaEvent_t gpu_record_time_start(cudaStream_t stream) {
    cudaEvent_t event;
    cudaEventCreate(&event);
    cudaEventRecord(event, stream);  // 在流中记录开始事件
    return event;
}

float gpu_record_time_stop(cudaEvent_t start, cudaStream_t stream, std::string name) {
    float milliseconds = 0;
    cudaEvent_t stop;
    cudaEventCreate(&stop);
    cudaEventRecord(stop, stream);
    auto err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) fo.eprint(std::string("CUDA Error: ") + cudaGetErrorString(err));
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (name != "") record_time(name, milliseconds / 1000.0);
    return milliseconds / 1000.0;
}

cudaEvent_t gpu_record_time_reset(cudaEvent_t start, cudaStream_t stream, float& time,
                                  std::string name) {
    float milliseconds = 0;
    cudaEvent_t event;
    cudaEventCreate(&event);
    cudaEventRecord(event, stream);
    auto err = cudaEventSynchronize(event);
    if (err != cudaSuccess) fo.eprint(std::string("CUDA Error: ") + cudaGetErrorString(err));
    cudaEventElapsedTime(&milliseconds, start, event);
    cudaEventDestroy(start);
    time = milliseconds / 1000.0;
    if (name != "") record_time(name, time);
    return event;
}

bool is_device_ptr(const void* p) {
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, p);
    if (err != cudaSuccess) {  // 不是 CUDA 管理的指针（99% 是普通 CPU 内存）
        cudaGetLastError();    // 清掉错误
        return false;
    }
    return attr.type == cudaMemoryTypeDevice;
}

}  // namespace efanna2e