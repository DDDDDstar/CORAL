#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <cub/cub.cuh>

#include "gpufuncs.cuh"
#include "utils.cuh"

namespace efanna2e {

// kernel: 每个线程处理一个向量，把它加到对应的中心累加数组
__global__ void compute_centroid_kernel(const float *__restrict__ data, float *center, int N,
                                        int dim) {
    const int tid = threadIdx.x, bid = blockIdx.x, tpb = blockDim.x, global_tid = bid * tpb + tid;

    extern __shared__ float sh_center[];  // 每个 block 的共享内存，大小 = tpb * dim
    for (int d = 0; d < dim; ++d) sh_center[tid * dim + d] = 0.0f;  // 初始化共享内存为 0
    for (int i = global_tid; i < N; i += gridDim.x * tpb)           // 每个线程处理多个数据点
        for (int d = 0; d < dim; ++d) sh_center[tid * dim + d] += data[i * dim + d];
    __syncthreads();

    for (int offset = tpb / 2; offset > 0;
         offset >>= 1) {  // 使用归约将 block 内的向量累加到线程 0
        if (tid < offset)
            for (int d = 0; d < dim; ++d)
                sh_center[tid * dim + d] += sh_center[(tid + offset) * dim + d];
        __syncthreads();
    }

    for (int d = tid; d < dim; d += tpb)  // block 内的线程将结果写回全局中心数组
        atomicAdd(&center[d], sh_center[d]);
}

// 单独 kernel 做归一化
__global__ void normalize_centroid_kernel(float *center, int N, int dim) {
    for (int d = threadIdx.x; d < dim; d += blockDim.x) center[d] /= N;
}

// bitset: 指向 int 数组；bit_index: 要访问的全局 bit 索引；返回值: 该 bit 的旧值 (0 or 1)
__device__ int atomicTestAndSetBit(int *bitset, int bit_index) {
    const int word_index = bit_index >> 5, bit_offset = bit_index & 31, mask = 1 << bit_offset;
    int *addr = bitset + word_index, old_val = *addr, assumed;
    while (true) {
        assumed = old_val;
        if (assumed & mask) return 1;  // 如果该位已经是 1，直接返回 1

        old_val = atomicCAS(addr, assumed, assumed | mask);  // 尝试置位
        if (old_val == assumed) return 0;  // CAS 成功，我们抢到了，把该位从 0 -> 1
        // 否则 CAS 失败 ，别的线程改过了，因为不知道改的是不是当前位，所以继续循环判断
    }
}

__device__ float atomicMinFloat(float *address, float val) {
    int *addr_as_int = (int *)address;
    int old = *addr_as_int, assumed;

    do {
        assumed = old;
        if (__int_as_float(old) <= val) break;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(val));
    } while (assumed != old);

    return __int_as_float(old);
}

__device__ float atomicMaxFloat(float *address, float val) {
    int *addr_as_int = (int *)address;
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
std::pair<T1 *, T2 *> temp_segmented_sort_pairs(  // 模板函数
    T1 *targets, T2 *values, T1 *sort_targets, T2 *sort_values, int *offsets, cudaStream_t &stream,
    int batch, int n) {
    // 使用 counting_iterator 快速生成 offsets，步长：n
    thrust::counting_iterator<int> count_iter(0);
    thrust::transform(thrust::cuda::par.on(stream), count_iter, count_iter + batch + 1,
                      thrust::device_pointer_cast(offsets), OffsetCalculator(n));

    void *temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    // 使用 CUB CachingDeviceAllocator 缓存临时内存分配:
    static cub::CachingDeviceAllocator g_allocator(true);

    cub::DoubleBuffer<T1> targets_buffer(targets, sort_targets);
    cub::DoubleBuffer<T2> values_buffer(values, sort_values);

    CUDA_CHECK(cub::DeviceSegmentedSort::SortPairs(temp_storage, temp_storage_bytes, values_buffer,
                                                   targets_buffer, batch * n, batch, offsets,
                                                   offsets + 1, stream));

    CUDA_CHECK(g_allocator.DeviceAllocate(  // 分配临时空间内存
        &temp_storage, temp_storage_bytes, stream));

    CUDA_CHECK(cub::DeviceSegmentedSort::SortPairs(  // 第二次正式排序
        temp_storage, temp_storage_bytes, values_buffer, targets_buffer, batch * n, batch, offsets,
        offsets + 1, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));         // 等待流中的排序操作完成
    CUDA_CHECK(g_allocator.DeviceFree(temp_storage));  // 安全释放临时存储

    return std::make_pair(targets_buffer.Current(), values_buffer.Current());
}

void segmented_sort_pairs_inplace(BCD *targets, float *values, int *offsets, cudaStream_t &stream,
                                  int batch, int n) {
    BCD *targets_sort;
    float *values_sort;
    CUDA_CHECK(cudaMallocAsync(&targets_sort, batch * n * sizeof(BCD), stream));
    CUDA_CHECK(cudaMallocAsync(&values_sort, batch * n * sizeof(float), stream));
    auto res = temp_segmented_sort_pairs<BCD, float>(targets, values, targets_sort, values_sort,
                                                     offsets, stream, batch, n);
    CUDA_CHECK(
        cudaMemcpyAsync(targets, res.first, n * sizeof(BCD), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaFreeAsync(targets_sort, stream));
    CUDA_CHECK(cudaFreeAsync(values_sort, stream));
}

std::pair<int *, float *> segmented_sort_pairs(int *targets, float *values, int *sort_targets,
                                               float *sort_values, int *offsets,
                                               cudaStream_t &stream, int batch, int n) {
    return temp_segmented_sort_pairs<int, float>(targets, values, sort_targets, sort_values,
                                                 offsets, stream, batch, n);
}

void prefix_exclusive_sum(int *d_array, int *d_prefixsum, int N, cudaStream_t &stream) {
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // 第一次调用获取临时存储大小
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_array,
                                             d_prefixsum, N, stream));
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // 第二次调用执行 prefix sum
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_array,
                                             d_prefixsum, N, stream));
}
}  // namespace efanna2e