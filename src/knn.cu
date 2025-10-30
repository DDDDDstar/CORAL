#include <curand_kernel.h>

#include <cfloat>
#include <iostream>
#include <string>
#include <utility>

#include "gpufuncs.cuh"
#include "uni.h"
#include "utils.cuh"

#define MAX_MSG_LEN 20

namespace efanna2e {
// using CN = Candidate_Neighbor;

bool cuda_check_last_error(std::string func_name) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fo.eprint("Kernel error in " + func_name + ": " + cudaGetErrorString(err));
        return true;
    }
    return false;
}

__global__ void compute_knn_dist_kernel(const float* __restrict__ d_q,
                                        const float* __restrict__ d_b, float* dists, int* idxs,
                                        int batch, BP bp) {
    const int qid = blockIdx.x % batch, bid = blockIdx.x / batch, tid = threadIdx.x;
    const int threads = blockDim.x, total_blocks = gridDim.x;

    extern __shared__ float query_vec[];  // 使用 shared memory 缓存当前查询向量，提高访存效率
    if (tid < bp.dim) query_vec[tid] = __ldg(&d_q[qid * bp.dim + tid]);
    __syncthreads();

    // 每个线程负责计算部分 base_n 向量的距离
    for (int i = bid * threads + tid; i < bp.base_n; i += threads * total_blocks / batch) {
        const float* base_vec = d_b + i * bp.dim;  // 当前基础向量
        dists[qid * bp.base_n + i] = bp.metric == DIST_METRIC::L2
                                         ? l2_distance(query_vec, base_vec, bp.dim)
                                         : ip_distance(query_vec, base_vec, bp.dim);
        idxs[qid * bp.base_n + i] = i;
    }
}

__device__ void insert_topk(float* heap_vals, int* heap_idxs, float val, int idx, int k) {
    // Find max in heap and replace if necessary
    int max_idx = 0;
    for (int i = 1; i < k; ++i)
        if (heap_vals[i] > heap_vals[max_idx]) max_idx = i;
    if (val < heap_vals[max_idx]) {
        heap_vals[max_idx] = val;
        heap_idxs[max_idx] = idx;
    }
}

__global__ void topk_copy_kernel(int* d_knn_res, const int* d_all_idx, int batch, BP bp) {
    // 每个 block 处理一个 KNN 结果拷贝
    for (int qidx = blockIdx.x; qidx < batch; qidx += gridDim.x) {
        const int src_offset = qidx * bp.base_n, dst_offset = qidx * bp.k;
        for (int i = threadIdx.x; i < bp.k; i += blockDim.x) {
            const int idx = bp.metric == DIST_METRIC::L2 ? i : bp.base_n - 1 - i;
            d_knn_res[dst_offset + i] = d_all_idx[src_offset + idx];
        }
    }
}

__global__ void topk_copy_kernel(int* d_knn_res, const int* d_all_idx, float* d_dist_res,
                                 const float* d_all_dist, int batch, BP bp) {
    // 每个 block 处理一个 KNN 结果拷贝
    for (int qidx = blockIdx.x; qidx < batch; qidx += gridDim.x) {
        const int src_offset = qidx * bp.base_n, dst_offset = qidx * bp.k;
        for (int i = threadIdx.x; i < bp.k; i += blockDim.x) {
            // ip 越大，距离越近
            const int idx = bp.metric == DIST_METRIC::L2 ? i : bp.base_n - 1 - i;
            d_knn_res[dst_offset + i] = d_all_idx[src_offset + idx];
            d_dist_res[dst_offset + i] = d_all_dist[src_offset + idx];
        }
    }
}

__global__ void base_query_update_kernel(const int* __restrict__ knns,
                                         const float* __restrict__ all_dists_sort,
                                         int* __restrict__ base_query_ids,
                                         float* __restrict__ base_query_dists, int start_qid,
                                         int batch, BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    for (int qidx = bid; qidx < batch; qidx += gridDim.x) {
        for (int i = tid; i < bp.k; i += tpb) {
            const int knn_idx = qidx * bp.k + i, base_id = knns[knn_idx];
            const float dist = all_dists_sort[qidx * bp.base_n + i];
            // 原子比较更新最近距离和索引:
            atomicClosestFloat(base_query_dists + base_id, dist, bp);
            __threadfence();  // 刷写全局/共享内存，使其他 block 可见
            if (base_query_dists[base_id] - dist < EPS) base_query_ids[base_id] = start_qid + qidx;
        }
    }
}

int* GPUFuncs::knn_compute(int batch, int start_id, float& time_ms) {
    const std::string name = "knn";
    cudaStream_t& stream = streams[name].stream;

    event_record_time_start(name);

    compute_knn_dist_kernel<<<batch * blocknum_per_query, knn_threads, bp.dim * sizeof(float),
                              stream>>>(d_q + start_id * bp.dim, d_b.data().get(), d_all_dist,
                                        d_all_idx, batch, bp);

    time_ms = event_record_time_stop(name, "knn_dist_compute");
    // 对每个查询结果进行排序，提取 top-K
    event_record_time_start(name);

    auto res = segmented_sort_pairs(d_all_idx, d_all_dist, d_all_idx_sort, d_all_dist_sort,
                                    d_topk_offsets, stream, batch, bp.base_n);
    topk_copy_kernel<<<batch, 64, 0, stream>>>(d_knn_res, res.first, batch, bp);
    query_knns.record(start_id, d_knn_res, batch, stream);

    {
        std::unique_lock<std::shared_mutex> lock(base_query_mtx);
        base_query_update_kernel<<<batch, 64, 0, stream>>>(d_knn_res, res.second, base_query_ids,
                                                           base_query_dists, start_id, batch, bp);
    }
    time_ms += event_record_time_stop(name, "knn_sort_compute");

    return d_knn_res;
}

int* GPUFuncs::knn_compute(const float* h_queries, int batch, float& time_ms) {
    const std::string name = "knn";
    cudaStream_t& stream = streams[name].stream;

    event_record_time_start(name);

    thrust::device_vector<float> queries(batch * bp.dim);
    CUDA_CHECK(cudaMemcpyAsync(queries.data().get(), h_queries, batch * bp.dim * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    // thrust::copy(thrust::cuda::par.on(stream), h_queries, h_queries + batch * bp.dim,
    //              queries.begin());

    compute_knn_dist_kernel<<<batch * blocknum_per_query, knn_threads, bp.dim * sizeof(float),
                              stream>>>(queries.data().get(), d_b.data().get(), d_all_dist,
                                        d_all_idx, batch, bp);

    time_ms = event_record_time_stop(name, "knn_dist_compute");
    // 对每个查询结果进行排序，提取 top-K
    event_record_time_start(name);

    auto res = segmented_sort_pairs(d_all_idx, d_all_dist, d_all_idx_sort, d_all_dist_sort,
                                    d_topk_offsets, stream, batch, bp.base_n);
    topk_copy_kernel<<<batch, 64, 0, stream>>>(d_knn_res, res.first, batch, bp);

    time_ms += event_record_time_stop(name, "knn_sort_compute");
    return d_knn_res;
}

// GPU 内核：初始化随机状态
__global__ void init_curand(curandState* states, unsigned long seed, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    curand_init(seed, idx, 0, &states[idx]);
}

// GPU 内核：生成 0~1 随机数
__global__ void gen_uniform(curandState* states, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    out[idx] = curand_uniform(&states[idx]);
}

// GPU 内核：计算每个点到最近中心的平方距离（用于 KMeans++）
__global__ void compute_min_dist(const float* __restrict__ data,
                                 const float* __restrict__ centroids, float* min_dists, BP bp,
                                 int num_centers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bp.query_n) return;

    float dist = INFINITY;
    for (int k = 0; k < num_centers; k++) {
        float d = calc_distance(data + idx * bp.dim, centroids + k * bp.dim, bp);
        if (d < dist) dist = d;
    }
    min_dists[idx] = dist;
}

void GPUFuncs::knn_prepare(const float* h_base, const float* h_queries) {
    cudaStream_t& stream = streams["knn"].stream;
    // 基础向量数据和查询向量数据  GPU 内存分配和数据传输（host->device）
    CUDA_CHECK(cudaMallocAsync(&d_q, bp.query_n * bp.dim * sizeof(float), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_b.data().get(), h_base, bp.base_n * bp.dim * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_q, h_queries, bp.query_n * bp.dim * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaMallocAsync(&d_knn_res, BATCH * bp.k * sizeof(int), stream));

    // knn_compute 预分配内存
    CUDA_CHECK(cudaMallocAsync(&d_all_dist, BATCH * bp.base_n * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_all_dist_sort, BATCH * bp.base_n * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_all_idx, BATCH * bp.base_n * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&d_all_idx_sort, BATCH * bp.base_n * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&d_topk_offsets, (BATCH + 1) * sizeof(int), stream));

    CUDA_CHECK(cudaMallocAsync(&base_query_ids, bp.base_n * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&base_query_dists, bp.base_n * sizeof(float), stream));
    thrust::device_ptr<int> base_query_ids_ptr(base_query_ids);
    thrust::device_ptr<float> base_query_dists_ptr(base_query_dists);
    thrust::fill(thrust::cuda::par.on(stream), base_query_ids_ptr, base_query_ids_ptr + bp.base_n,
                 -1);
    thrust::fill(thrust::cuda::par.on(stream), base_query_dists_ptr,
                 base_query_dists_ptr + bp.base_n, farthest_dist(bp));
}
void GPUFuncs::knn_free() {
    cudaStream_t& stream = streams["knn"].stream;
    CUDA_CHECK(cudaFreeAsync(d_q, stream));
    CUDA_CHECK(cudaFreeAsync(d_knn_res, stream));
    CUDA_CHECK(cudaFreeAsync(d_all_dist, stream));
    CUDA_CHECK(cudaFreeAsync(d_all_idx, stream));
    CUDA_CHECK(cudaFreeAsync(d_topk_offsets, stream));
    CUDA_CHECK(cudaFreeAsync(d_all_dist_sort, stream));
    CUDA_CHECK(cudaFreeAsync(d_all_idx_sort, stream));
    CUDA_CHECK(cudaFreeAsync(base_query_ids, stream));
    CUDA_CHECK(cudaFreeAsync(base_query_dists, stream));
}

GPUFuncs::GPUFuncs(const float* h_base, float* h_queries, int base_n, int query_n, int dim, int k,
                   int max_degree, DIST_METRIC m, int beam_capacity)
    : bp(dim, max_degree, base_n, query_n, k, beam_capacity, m),
      query_knns(100000, bp),
      graph(bp.base_n * bp.max_degree, -1),
      graph_deg(bp.base_n, 0),
      graph_dist(bp.base_n * bp.max_degree),
      d_b(bp.base_n * bp.dim) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);              // 获取 GPU 设备
    shared_mem_per_block = prop.sharedMemPerBlock;  // GPU 最大共享内存大小
    fo.print("Compute Capability: " + std::to_string(prop.major) + " " +
             std::to_string(prop.minor));
    fo.print("Max shared memory per block: " + std::to_string(shared_mem_per_block) + " bytes");
    fo.print("Max shared memory permultiprocessor: " +
             std::to_string(prop.sharedMemPerMultiprocessor) + " bytes");
    if (!prop.deviceOverlap)
        fo.eprint("Device does not support overlap, multi-stream may not improve performance.");
    else
        fo.print("Device supports overlap, multi-stream may improve performance.");

    for (auto name : stream_names) {
        streams.emplace(name, MyStream());
        CUDA_CHECK(cudaStreamCreate(&streams[name].stream));
        CUDA_CHECK(cudaEventCreate(&streams[name].start));
        CUDA_CHECK(cudaEventCreate(&streams[name].stop));
    }
    query_data_kmeans(h_queries);
    knn_prepare(h_base, h_queries);  // knn 预分配内存
    upd_prepare();                   // upd 预分配内存
    search_prepare();                // search 预分配内存
    fo.iprint(bp.str());
}

GPUFuncs::~GPUFuncs() {
    knn_free();
    upd_free();
    search_free();
    for (auto& pair : streams) {
        CUDA_CHECK(cudaStreamDestroy(pair.second.stream));
        CUDA_CHECK(cudaEventDestroy(pair.second.start));
        CUDA_CHECK(cudaEventDestroy(pair.second.stop));
    }
}

void GPUFuncs::event_record_time_start(const std::string name) {
    cudaEventRecord(streams[name].start, streams[name].stream);  // 在流中记录开始事件
}

float GPUFuncs::event_record_time_stop(const std::string stream_name, std::string func_name) {
    if (cuda_check_last_error(func_name)) exit(EXIT_FAILURE);

    float milliseconds = 0;
    cudaEvent_t& stop = streams[stream_name].stop;
    cudaEventRecord(stop, streams[stream_name].stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, streams[stream_name].start, stop);

    record_time(func_name, milliseconds / 1000);

    return milliseconds;
}

void GPUFuncs::record_time(const std::string func_name, float time_s) {
    if (!avg_times.contains(func_name)) avg_times[func_name] = Avg_Time();
    avg_times[func_name].add(time_s);
}

void GPUFuncs::print_times() {
    std::vector<std::pair<std::string, Avg_Time>> avg_times_vec(avg_times.begin(),
                                                                avg_times.end());
    std::sort(avg_times_vec.begin(), avg_times_vec.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    std::string s;
    for (auto& pair : avg_times_vec) {
        const std::string& func = pair.first;
        s += (func.size() >= MAX_MSG_LEN ? func
                                         : func + std::string(MAX_MSG_LEN - func.size(), ' ')) +
             " 任务平均耗时: " + std::to_string(pair.second.get()) +
             " s; 总耗时: " + std::to_string(pair.second.get_total()) + " s\n";
    }
    fo.iprint(s);
}

int GPUFuncs::get_free_bytes() {
    size_t free_bytes, total_bytes;                                  // 总显存
    cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);  // 获取当前设备的显存信息

    std::string func = "get_free_bytes";
    if (status != cudaSuccess) fo.eprint(func + " Error: " + cudaGetErrorString(status));

    // std::cout << "Total GPU memory: " << (total_bytes / (1024.0 * 1024.0)) << " MB" <<
    // std::endl; std::cout << "Free GPU memory:  " << (free_bytes / (1024.0 * 1024.0)) << " MB" <<

    return free_bytes;
}

Query_KNNs::Query_KNNs(int size, BP bp) : size(size), knns(size * bp.k), bp(bp) {}

void Query_KNNs::record(int qid, int* new_knns, int batch, cudaStream_t& stream) {
    if (qid >= size) {
        size *= 2;
        knns.resize(size);
    }
    CUDA_CHECK(cudaMemcpyAsync(knns.data().get() + qid * bp.k, new_knns,
                               batch * bp.k * sizeof(int), cudaMemcpyDeviceToDevice, stream));
}
}  // namespace efanna2e