#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <cfloat>
#include <utility>
#include <iostream>
#include <cmath>
#include <cub/cub.cuh>

#include "knn.cuh"
#include "fileout.h"

using namespace efanna2e;
using CN = Candidate_Neighbor;

bool cuda_check_last_error(const char *func_name)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Kernel error in " << func_name << ": " << cudaGetErrorString(err) << std::endl;
        return true;
    }
    return false;
}

// ----------- 欧式距离平方计算 -----------
__device__ float l2_distance(const float *a, const float *b, int dim)
{
    float dist = 0.0f;
    for (int i = 0; i < dim; ++i)
    {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

__device__ float l2_distance_runtime_load(
    const float *a, const float *b_base, uint32_t b_index, int dim)
{
    float dist = 0.0f;
    for (int i = 0; i < dim; ++i)
    {
        float diff = a[i] - b_base[b_index * dim + i];
        dist += diff * diff;
    }
    return dist;
}

__global__ void compute_knn_candidates(
    const float *__restrict__ d_q, // 查询向量
    const float *__restrict__ d_b, // 基础向量库
    float *dists,                  // 输出：所有距离（每个查询占 base_n 个）
    uint32_t *idxs,                // 输出：所有索引（与距离一一对应）
    int batch, int base_n, int dim)
{
    int qid = blockIdx.x % batch; // 当前处理的查询向量编号
    int bid = blockIdx.x / batch;
    int tid = threadIdx.x, threads = blockDim.x;
    int total_blocks = gridDim.x;
    // 使用 shared memory 缓存当前查询向量，提高访存效率
    extern __shared__ float query_vec[];
    if (bid == 0 && tid < dim)
        query_vec[tid] = d_q[qid * dim + tid];
    __syncthreads();
    // 每个线程负责计算部分 base_n 向量的距离
    for (int i = bid * threads + tid; i < base_n; i += threads * total_blocks / batch)
    {
        float dist = l2_distance(query_vec, &d_b[i * dim], dim);
        dists[qid * base_n + i] = dist;
        idxs[qid * base_n + i] = i;
    }
}

__global__ void knn_dist_compute_kernel(
    const float *__restrict__ base, int base_n,
    const uint32_t *__restrict__ knns, // 查询的 knn 结果 batch * k
    CN *cand_nbrs,                     // 存储结果：K 个向量两两之间距离（batch*K*K 矩阵）
    const int tile_k, const int K, const int dim)
{
    const int group_id = blockIdx.x; // 当前处理的一组 KNN 的编号
    const int tid = threadIdx.x, threads_per_block = blockDim.x;
    // 首先计算一组 KNN 中所有基础向量之间的距离
    extern __shared__ float tile_vecs[]; // 存储一个 tile 的基础向量，大小为 tile_k * dim

    for (int row_tile = 0; row_tile < K; row_tile += tile_k)
    {
        // 加载 tile_k 个 row 向量进共享内存
        for (int i = tid; i < tile_k * dim; i += threads_per_block)
        {
            const int vec_idx = i / dim; // tile 中第 vec_idx 个向量
            const int dim_idx = i % dim;
            const int global_idx = row_tile + vec_idx;
            if (global_idx < K)
            {
                const uint32_t base_idx = knns[group_id * K + global_idx];
                tile_vecs[i] = base[base_idx * dim + dim_idx];
            }
        }
        __syncthreads();

        for (int col_tile = 0; col_tile < K; col_tile += tile_k)
        {
            for (int i = tid; i < tile_k * tile_k; i += threads_per_block)
            {
                const int local_row = i / tile_k, local_col = i % tile_k;
                const int global_row = row_tile + local_row;
                const int global_col = col_tile + local_col;
                if (global_row < K && global_col < K)
                {
                    const uint32_t col_base_idx = knns[group_id * K + global_col];
                    const float *tile_row_vec = &tile_vecs[local_row * dim];
                    const int matrix_idx = group_id * K * K + global_row * K + global_col;
                    cand_nbrs[matrix_idx].dist = l2_distance_runtime_load(
                        tile_row_vec, base, col_base_idx, dim);
                    cand_nbrs[matrix_idx].idx = global_col;
                    cand_nbrs[matrix_idx].id = col_base_idx;
                }
            }
            __syncthreads(); // 等待所有线程完成当前 tile 的计算
        }
    }
}

__device__ CN &get_CN(
    CN *cand_nbrs, const uint8_t *sort_idxs,
    const uint8_t i, const int batch_id, const int pivot_idx, const int k)
{
    const uint8_t idx = sort_idxs[batch_id * k * k + pivot_idx * k + i];
    return cand_nbrs[batch_id * k * k + pivot_idx * k + idx];
}

// wait_to_ignore = wti
__global__ void candidate_ignore_kernel(
    CN *cand_nbrs,           // 每个批次的所有 KNN 中每个 pivot 的候选邻居向量到 pivot 的距离等信息，大小：batch * k * k
    uint8_t *sort_idxs,      // 每个批次的 KKN 中向量每个 pivot 的候选邻居向量根据到 pivot 的距离排序后的原索引（0~k-1），大小：batch * k * k
    uint8_t *nbr_num,        // 记录当前每个 pivot 的邻居数量，大小：batch * k
    const uint8_t new_nbr_i, // 1 <= new_nbr_i <= k - 1
    const int batch, const int k, const int max_degree)
{
    const int batch_id = blockIdx.x, tid = threadIdx.x, threads_per_block = blockDim.x;
    const int wti_num = k - new_nbr_i - 1;
    for (int i = tid; i < k * wti_num; i += threads_per_block)
    {
        const uint8_t pivot_idx = i / wti_num;
        if (nbr_num[batch_id * k + pivot_idx] >= max_degree)
            continue;
        CN &cand_nbr = get_CN(cand_nbrs, sort_idxs, new_nbr_i, batch_id, pivot_idx, k); // pivot 的第 new_nbr_i 个候选邻居（排序后）

        if (!cand_nbr.status) // 若新候选邻居状态为待定（未淘汰）
        {
            cand_nbr.status = 1;                 // 成为 pivot 的新邻居
            nbr_num[batch_id * k + pivot_idx]++; // pivot 的邻居数量加一

            CN &wti_cand_nbr = get_CN(cand_nbrs, sort_idxs, i % wti_num + new_nbr_i + 1, batch_id, pivot_idx, k); // 待淘汰候选邻居

            if (!wti_cand_nbr.status) // 若待淘汰邻居状态为待定（未淘汰）
            {
                const float ignore_pivot_dist = wti_cand_nbr.dist;                                                        // 待淘汰候选邻居到 pivot 的距离
                const float new_nbr_ignore_dist = cand_nbrs[batch_id * k * k + cand_nbr.idx * k + wti_cand_nbr.idx].dist; // new_nbr 和 wti 之间的距离
                if (ignore_pivot_dist >= new_nbr_ignore_dist)
                    wti_cand_nbr.status = 2; // 淘汰
            }
        }
    }
}

// 根据 candidate_ignore_kernel 的结果（d_cand_nbrs）提取每个批次的 KNN 中每个 pivot 的邻居 base_id
// 结果存在 new_nbrs 中，大小：batch * k * max_degree
__global__ void get_new_nbrs_kernel(
    CN *cand_nbrs, uint32_t *new_nbr_ids, const int k, const int max_degree)
{
    const int batch_id = blockIdx.x, tid = threadIdx.x, threads_per_block = blockDim.x;
    for (int i = tid; i < k; i += threads_per_block)
    {
        uint32_t nbr_num = 0;
        for (int j = 0; j < k && nbr_num < max_degree; j++)
        {
            CN &cand_nbr = cand_nbrs[batch_id * k * k + i * k + j];
            if (cand_nbr.status == 1)
                new_nbr_ids[batch_id * k * max_degree + i * max_degree + nbr_num++] = cand_nbr.id;
        }
    }
}

// CUDA kernel: 生成 CUB 段排序所需 offsets 数组（每 batch * k 个长度为 k 的段）
__global__ void generate_segment_offsets(uint32_t *offsets, const int batch, const int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= batch * k)
        offsets[idx] = idx * k;
}

uint32_t *GPUFuncs::knn_compute(const float *h_queries, const int batch, float &time_ms)
{
    const std::string name = "knn";
    cudaStream_t &stream = streams[name].stream;
    float *d_all_dist; // 存储所有查询的距离与索引结果，用于之后排序
    uint32_t *d_all_idx;
    int blocks = batch * blocknum_per_query;
    int threads = 1024; // must > K
    event_record_time_start(name);
    // 传输 batch 查询
    CUDA_CHECK(cudaMemcpyAsync(
        d_q, h_queries, batch * dim * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMallocAsync(&d_all_dist, batch * base_n * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_all_idx, batch * base_n * sizeof(uint32_t), stream));
    compute_knn_candidates<<<blocks, threads, dim * sizeof(float), stream>>>(
        d_q, d_b, d_all_dist, d_all_idx, batch, base_n, dim);
    time_ms = event_record_time_stop(name, "dist_compute");
    // 对每个查询结果进行排序，提取 top-K（使用 thrust 排序）
    event_record_time_start(name);
    cudaMemsetAsync(d_knn_res, -1, batch * k * sizeof(int), stream);
    for (int qid = 0; qid < batch; ++qid)
    {
        thrust::device_ptr<float> dist_ptr(d_all_dist + qid * base_n);
        thrust::device_ptr<uint32_t> idx_ptr(d_all_idx + qid * base_n);
        // 对当前查询向量对应的 base_n 距离和索引进行排序
        thrust::sort_by_key(
            thrust::cuda::par.on(stream), dist_ptr, dist_ptr + base_n, idx_ptr);
        // 仅拷贝前 k 个最小距离对应的索引作为输出
        CUDA_CHECK(cudaMemcpyAsync(
            d_knn_res + qid * k, d_all_idx + qid * base_n, k * sizeof(int),
            cudaMemcpyDeviceToDevice, stream));
    }
    time_ms += event_record_time_stop(name, "knn_sort_compute");
    CUDA_CHECK(cudaFreeAsync(d_all_dist, stream));
    CUDA_CHECK(cudaFreeAsync(d_all_idx, stream));
    return d_knn_res;
}

struct ExtractDist
{
    __host__ __device__ float operator()(const CN &nbr) const
    {
        return nbr.dist;
    }
};

struct ExtractIdx
{
    __host__ __device__ float operator()(const CN &nbr) const
    {
        return nbr.idx;
    }
};

uint32_t *GPUFuncs::handle_knn_updates(const uint32_t *d_knn_idxs, const int batch)
{
    const std::string name = "upd";
    cudaStream_t &stream = streams[name].stream;

    const size_t max_vec_num = shared_mem_per_block / sizeof(float) / dim; // 共享内存最大向量存储数量
    const size_t tile_k = k / ((k + max_vec_num - 1) / max_vec_num);       // 每个 block 根据共享内存大小限制分块处理对应的 KNN
    const size_t shared_memsize = tile_k * dim * sizeof(float);            // 每个 block 的共享内存大小

    // 1. 计算一个 batch 中每组 KNN 中基础向量两两间距
    event_record_time_start(name);
    CN *d_cand_nbrs; // 每个批次的所有 KNN 中每个 pivot 的候选邻居向量信息，大小：batch * k * k
    CUDA_CHECK(cudaMallocAsync(&d_cand_nbrs, batch * k * k * sizeof(CN), stream));
    knn_dist_compute_kernel<<<batch, 256, shared_memsize, stream>>>(
        d_b, base_n, d_knn_idxs, d_cand_nbrs, tile_k, k, dim);
    event_record_time_stop(name, "knn_dist_compute");

    // 2.1. 针对每组 KNN 中每个 pivot 节点，对候选向量集（其他 k-1 个向量）按到 pivot 的距离排序
    // 初始距离和排序后距离都存储在作为上一步结果的 d_knn_dists（batch*k*k 的矩阵）中
    event_record_time_start(name);
    float *d_knn_dists;
    uint8_t *d_sort_idxs;
    CUDA_CHECK(cudaMallocAsync(&d_knn_dists, batch * k * k * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_sort_idxs, batch * k * k * sizeof(uint8_t), stream));
    thrust::device_ptr<CN> nbrs_ptr(d_cand_nbrs);
    thrust::device_ptr<float> dists_ptr(d_knn_dists);
    thrust::device_ptr<uint8_t> idxs_ptr(d_sort_idxs);
    // 提取结构体数组 d_cand_nbrs 中的 dist 字段，拷贝到数组 d_knn_dists（即 batch * k * k 的距离）
    thrust::transform(
        thrust::cuda::par.on(stream),
        nbrs_ptr, nbrs_ptr + batch * k * k, dists_ptr,
        ExtractDist());
    thrust::transform(
        thrust::cuda::par.on(stream),
        nbrs_ptr, nbrs_ptr + batch * k * k, d_sort_idxs,
        ExtractIdx());
    // 使用 cub::DeviceSegmentedSort 对 d_cand_nbrs 进行排序
    float *d_knn_dists_sort;
    CUDA_CHECK(cudaMallocAsync(&d_knn_dists_sort, batch * k * k * sizeof(float), stream));
    uint32_t *d_offsets;
    CUDA_CHECK(cudaMallocAsync(&d_offsets, (batch * k + 1) * sizeof(uint32_t), stream));
    generate_segment_offsets<<<(batch * k + 255) / 256, 256, 0, stream>>>(d_offsets, batch, k);
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    // 使用双缓冲模式
    cub::DoubleBuffer<float> dists_buffer(d_knn_dists, d_knn_dists);
    cub::DoubleBuffer<uint8_t> idxs_buffer(d_sort_idxs, d_sort_idxs);
    cub::DeviceSegmentedSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        dists_buffer, idxs_buffer,
        batch * k * k, batch * k, d_offsets, d_offsets + 1, stream);
    CUDA_CHECK(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));
    cub::DeviceSegmentedSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        dists_buffer, idxs_buffer,
        batch * k * k, batch * k, d_offsets, d_offsets + 1, stream);
    CUDA_CHECK(cudaFreeAsync(d_offsets, stream));
    CUDA_CHECK(cudaFreeAsync(d_knn_dists, stream));
    CUDA_CHECK(cudaFreeAsync(d_knn_dists_sort, stream));
    CUDA_CHECK(cudaFreeAsync(d_temp_storage, stream));
    event_record_time_stop(name, "knn_dist_sort");

    // 2.2. 针对 batch 组 k 近邻数据，每次迭代线程并行进行 𝑏𝑎𝑡𝑐ℎ * 𝑘 * (𝑘 − 2 − 𝑖) 次淘汰
    event_record_time_start(name);
    uint8_t *d_nbr_num; // 每个 pivot 的邻居数量，大小：batch * k
    CUDA_CHECK(cudaMallocAsync(&d_nbr_num, batch * k * sizeof(uint8_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_nbr_num, 0, batch * k * sizeof(uint8_t), stream));
    // 根据每个 pivot 候选邻居到 pivot 的距离，从近到远成为邻居并并行淘汰后面的后续邻居
    for (uint8_t i = 1; i < k; ++i)
        candidate_ignore_kernel<<<batch, 256, 0, stream>>>(
            d_cand_nbrs, d_sort_idxs, d_nbr_num, i, batch, k, max_degree);

    CUDA_CHECK(cudaMemsetAsync(d_new_nbr_ids, -1, BATCH * k * max_degree * sizeof(uint32_t), stream));
    get_new_nbrs_kernel<<<batch, k, 0, stream>>>(d_cand_nbrs, d_new_nbr_ids, k, max_degree);
    CUDA_CHECK(cudaFreeAsync(d_nbr_num, stream));
    CUDA_CHECK(cudaFreeAsync(d_cand_nbrs, stream));
    event_record_time_stop(name, "candidate_ignore");

    return d_new_nbr_ids;
}

GPUFuncs::GPUFuncs(
    const float *h_base, int base_n, int dim, int k, int max_degree)
    : base_n(base_n), dim(dim), k(k), max_degree(max_degree)
{
    cudaDeviceProp prop;
    // 获取 GPU 设备
    cudaGetDeviceProperties(&prop, 0);
    shared_mem_per_block = prop.sharedMemPerBlock; // GPU 最大共享内存大小

    fo.print("Compute Capability: " + std::to_string(prop.major) + " " + std::to_string(prop.minor));
    fo.print("Max shared memory per block: " + std::to_string(shared_mem_per_block) + " bytes");
    fo.print("Max shared memory per multiprocessor: " + std::to_string(prop.sharedMemPerMultiprocessor) + " bytes");

    if (!prop.deviceOverlap)
        fo.eprint("Device does not support overlap, multi-stream may not improve performance.");
    else
        fo.print("Device supports overlap, multi-stream may improve performance.");

    for (auto name : stream_names)
    {
        streams.emplace(name, MyStream());
        CUDA_CHECK(cudaStreamCreate(&streams[name].stream));
        CUDA_CHECK(cudaEventCreate(&streams[name].start));
        CUDA_CHECK(cudaEventCreate(&streams[name].stop));
    }

    // 基础向量数据 GPU 内存分配和数据传输（host->device）
    CUDA_CHECK(cudaMallocAsync(&d_b, base_n * dim * sizeof(float), streams["knn"].stream));
    CUDA_CHECK(cudaMemcpyAsync(
        d_b, h_base, base_n * dim * sizeof(float), // 字节数
        cudaMemcpyHostToDevice, streams["knn"].stream));
    // 查询向量数据（一个 batch） GPU 内存分配
    CUDA_CHECK(cudaMallocAsync(&d_q, BATCH * dim * sizeof(float), streams["knn"].stream));

    CUDA_CHECK(cudaMallocAsync(&d_knn_res, BATCH * k * sizeof(uint32_t), streams["knn"].stream));

    CUDA_CHECK(cudaMallocAsync(&d_new_nbr_ids, BATCH * k * max_degree * sizeof(uint32_t), streams["upd"].stream));
}

GPUFuncs::~GPUFuncs()
{
    CUDA_CHECK(cudaFreeAsync(d_q, streams["knn"].stream));
    CUDA_CHECK(cudaFreeAsync(d_b, streams["knn"].stream));
    CUDA_CHECK(cudaFreeAsync(d_knn_res, streams["knn"].stream));

    for (auto &pair : streams)
    {
        CUDA_CHECK(cudaStreamDestroy(pair.second.stream));
        CUDA_CHECK(cudaEventDestroy(pair.second.start));
        CUDA_CHECK(cudaEventDestroy(pair.second.stop));
    }
}

void GPUFuncs::event_record_time_start(const std::string name)
{
    cudaEventRecord(streams[name].start, streams[name].stream); // 在流中记录开始事件
}

float GPUFuncs::event_record_time_stop(const std::string name, std::string msg)
{
    if (cuda_check_last_error(msg.c_str()))
        exit(EXIT_FAILURE);
    float milliseconds = 0;
    cudaEvent_t &start = streams[name].start, &stop = streams[name].stop;
    // 在流中记录结束事件
    cudaEventRecord(stop, streams[name].stream);
    // 阻塞主机直到stop事件完成
    cudaEventSynchronize(stop);
    // 记录开始到结束的时间
    cudaEventElapsedTime(&milliseconds, start, stop);

    fo.print(msg + "任务耗时: " + std::to_string(milliseconds / 1000) + " s");

    return milliseconds;
}
