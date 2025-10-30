#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "gpufuncs.cuh"
#include "uni.h"
#include "utils.cuh"

#define MAX_MSG_LEN 20

namespace efanna2e {

// 在 [0, N-1] 中随机选取 K 个不重复的整数
std::vector<int> random_unique_numbers(int N, int K) {
    std::vector<int> nums(N);
    std::random_device rd;  // 随机数引擎
    std::mt19937 gen(rd());

    for (int i = 0; i < N; ++i) nums[i] = i;
    for (int i = 0; i < K; ++i) {  // Fisher-Yates 洗牌
        std::uniform_int_distribution<> dis(i, N - 1);
        int j = dis(gen);
        std::swap(nums[i], nums[j]);
    }

    return std::vector<int>(nums.begin(), nums.begin() + K);  // 返回前 K 个数
}

__global__ void cluster_assign_kernel(const float *__restrict__ queries,
                                      const int *__restrict__ centroid_ids,
                                      int *__restrict__ labels,  // 输出: 每个样本的簇 ID
                                      BP bp, int K, int n) {
    const int block_num = gridDim.x, tpb = blockDim.x, bid = blockIdx.x, tid = threadIdx.x;
    for (int id = bid * tpb + tid; id < n; id += block_num * tpb) {
        const float *query = queries + id * bp.dim;
        float min_dist = farthest_dist_d(bp);
        int best_cid = -1;
        for (int i = 0; i < K; ++i) {
            const float dist = calc_distance(query, queries + centroid_ids[i] * bp.dim, bp);
            if (compare_dist(min_dist, dist, bp)) {
                min_dist = dist;
                best_cid = i;
            }
        }
        labels[id] = best_cid;
    }
}

// kernel: 每个线程处理一个向量，把它加到对应的中心累加数组
__global__ void compute_centroid_kernel(const float *__restrict__ queries,
                                        const int *__restrict__ labels,
                                        float *__restrict__ centers,
                                        int *__restrict__ centroid_ids,
                                        int *__restrict__ not_converge,
                                        int *__restrict__ cluster_sizes, BP bp, int K, int sub_n) {
    const int block_num = gridDim.x, tpb = blockDim.x, bid = blockIdx.x, tid = threadIdx.x;
    extern __shared__ float sh_mem[];
    float *sh_min_dtcs = sh_mem;
    int *sh_min_ids = (int *)(sh_min_dtcs + tpb), *sh_cluster_size = sh_min_ids + tpb;

    if (bid == 0 && tid == 0) *not_converge = 0;

    for (int cluster_idx = bid; cluster_idx < K; cluster_idx += block_num) {
        float *centers_start = centers + cluster_idx * tpb * bp.dim;
        sh_min_dtcs[tid] = farthest_dist_d(bp);
        if (tid == 0) *sh_cluster_size = 0;
        __syncthreads();

        for (int d = 0; d < bp.dim; ++d) centers_start[tid * bp.dim + d] = 0.0f;  // 初始化
        for (int i = tid; i < sub_n; i += tpb) {
            if (labels[i] != cluster_idx) continue;
            atomicAdd(sh_cluster_size, 1);
            for (int d = 0; d < bp.dim; ++d)
                centers_start[tid * bp.dim + d] += queries[i * bp.dim + d];
        }
        __syncthreads();

        // 使用归约将 block 内的向量累加到线程 0
        for (int offset = tpb / 2; offset > 0; offset >>= 1) {
            if (tid < offset)
                for (int d = 0; d < bp.dim; ++d)
                    centers_start[tid * bp.dim + d] += centers_start[(tid + offset) * bp.dim + d];
            __syncthreads();
        }

        for (int d = tid; d < bp.dim; d += tpb)  // 计算中心点平均值
            centers_start[d] /= *sh_cluster_size;
        __syncthreads();

        for (int i = tid; i < sub_n; i += tpb) {  // 计算 cluster 中每个向量和中心点的距离
            if (labels[i] != cluster_idx) continue;
            const float dist = calc_distance(queries + i * bp.dim, centers_start, bp);
            if (compare_dist(sh_min_dtcs[tid], dist, bp)) {
                sh_min_dtcs[tid] = dist;
                sh_min_ids[tid] = i;
            }
        }
        __syncthreads();
        // 归约找到 centroid 向量：每次一半线程参与
        for (int stride = tpb / 2; stride > 0; stride >>= 1) {
            if (tid < stride)
                if (compare_dist(sh_min_dtcs[tid], sh_min_dtcs[tid + stride], bp)) {
                    sh_min_dtcs[tid] = sh_min_dtcs[tid + stride];
                    sh_min_ids[tid] = sh_min_ids[tid + stride];
                }
            __syncthreads();
        }
        if (tid == 0 && centroid_ids[cluster_idx] != sh_min_ids[0]) {
            centroid_ids[cluster_idx] = sh_min_ids[0];
            *not_converge = 1;
            cluster_sizes[cluster_idx] = *sh_cluster_size;
        }
    }
}

__global__ void gather_sub_queries_kernel(const float *__restrict__ queries,
                                          const int *__restrict__ idxs,
                                          float *__restrict__ sub_queries, BP bp, int sub_n) {
    const int block_num = gridDim.x, tpb = blockDim.x, bid = blockIdx.x, tid = threadIdx.x;
    for (int i = bid * block_num + tid; i < sub_n; i += block_num * tpb)
        for (int d = 0; d < bp.dim; ++d)
            sub_queries[i * bp.dim + d] = queries[idxs[i] * bp.dim + d];
}

__global__ void cluster_size_statistics_kernel(const int *__restrict__ labels,
                                               int *__restrict__ cluster_sizes, int K, int n) {
    const int block_num = gridDim.x, tpb = blockDim.x, bid = blockIdx.x, tid = threadIdx.x;
    extern __shared__ int sh_cluster_sizes[];
    for (int cluster_idx = tid; cluster_idx < K; cluster_idx += tpb)
        sh_cluster_sizes[cluster_idx] = 0;
    __syncthreads();

    for (int i = bid * tpb + tid; i < n; i += block_num * tpb)
        atomicAdd(sh_cluster_sizes + labels[i], 1);
    __syncthreads();

    for (int cluster_idx = tid; cluster_idx < K; cluster_idx += tpb)
        atomicAdd(cluster_sizes + cluster_idx, sh_cluster_sizes[cluster_idx]);
}

__global__ void query_cluster_bucket_kernel(const float *__restrict__ queries,
                                            const int *__restrict__ labels,
                                            const int *__restrict__ cluster_offsets,
                                            float *__restrict__ bucket_queries, int K, BP bp) {
    const int block_num = gridDim.x, tpb = blockDim.x, bid = blockIdx.x, tid = threadIdx.x;
    extern __shared__ int sh_cluster_member_idx[];
    for (int cluster_idx = bid; cluster_idx < K; cluster_idx += block_num) {
        if (tid == 0) *sh_cluster_member_idx = 0;
        __syncthreads();

        const int offset = cluster_idx == 0 ? 0 : cluster_offsets[cluster_idx - 1];
#ifdef GPU_TEST
        assert(offset >= 0 && offset < bp.query_n);
#endif
        float *queries_start = bucket_queries + offset * bp.dim;
        for (int i = tid; i < bp.query_n; i += tpb) {
            if (labels[i] != cluster_idx) continue;
            const int idx = atomicAdd(sh_cluster_member_idx, 1);
            for (int d = 0; d < bp.dim; ++d)
                queries_start[idx * bp.dim + d] = queries[i * bp.dim + d];
        }
#ifdef GPU_TEST
        __syncthreads();
        if (tid == 0 && cluster_idx > 0)
            assert(*sh_cluster_member_idx ==
                   cluster_offsets[cluster_idx] - cluster_offsets[cluster_idx - 1]);
#endif
    }
}

__global__ void query_round_robin_kernel(const float *__restrict__ queries,
                                         const int *__restrict__ cluster_sizes,
                                         const int *__restrict__ cluster_offsets,
                                         float *__restrict__ round_queries, int K, BP bp) {
    const int block_num = gridDim.x, tpb = blockDim.x, bid = blockIdx.x, tid = threadIdx.x;
    for (int idx = bid * tpb + tid; idx < bp.query_n; idx += block_num * tpb) {
        int cnt = 0;
        for (int r = 0;; r++) {
            for (int cluster_idx = 0; cluster_idx < K; cluster_idx++) {
                if (r < cluster_sizes[cluster_idx]) {
                    if (cnt++ == idx) {
                        round_queries[tid] = queries[cluster_offsets[cluster_idx] + r];
                        return;
                    }
                }
            }
        }
    }
}

void GPUFuncs::query_data_kmeans(float *h_queries, int K) {
    auto start_time = std::chrono::high_resolution_clock::now();

    std::string stream_name = "knn";
    cudaStream_t &stream = streams[stream_name].stream;

    const size_t query_data_size = bp.query_n * bp.dim * sizeof(float);
    thrust::device_vector<float> d_queries(bp.query_n * bp.dim);
    int *d_not_converge, not_converge = 1;
    const int tpb = 256;

    event_record_time_start(stream_name);

    CUDA_CHECK(cudaMallocAsync(&d_not_converge, sizeof(int), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_queries.data().get(), h_queries, query_data_size,
                               cudaMemcpyHostToDevice, stream));

    // 随机采样子集
    const int sub_n = KMEANS_SIZE_RATIO * bp.query_n;
    thrust::device_vector<int> d_idxs(bp.query_n), d_centroid_ids(K);
    thrust::sequence(thrust::cuda::par.on(stream), d_idxs.begin(),
                     d_idxs.end());  // 索引 [0..N-1]
    thrust::shuffle(thrust::cuda::par.on(stream), d_idxs.begin(), d_idxs.end(),
                    thrust::default_random_engine(1234));
    d_idxs.resize(sub_n);  // 保留前 sub_n 个

    thrust::device_vector<float> d_sub_queries((size_t)sub_n * bp.dim);
    // gather 子集数据
    gather_sub_queries_kernel<<<256, tpb, 0, stream>>>(d_queries.data().get(), d_idxs.data().get(),
                                                       d_sub_queries.data().get(), bp, sub_n);

    // 从子集随机选择 K 个点作为初始中心
    thrust::sequence(thrust::cuda::par.on(stream), d_idxs.begin(), d_idxs.begin() + sub_n);
    thrust::shuffle(thrust::cuda::par.on(stream), d_idxs.begin(), d_idxs.begin() + sub_n,
                    thrust::default_random_engine(5678));
    thrust::copy_n(thrust::cuda::par.on(stream), d_idxs.begin(), K, d_centroid_ids.begin());

    thrust::device_vector<int> d_labels(bp.query_n);
    thrust::device_vector<float> d_centers((size_t)K * tpb * bp.dim);
    thrust::device_vector<int> d_cluster_sizes(K), d_cluster_offsets(K, 0);

    float time = event_record_time_stop(stream_name, "kmeans_prepare") / 1000.0;

    // 迭代 K-means
    std::vector<int> centroid_ids(K), last_centroid_ids(K);
    for (int iter = 0; iter < KMEANS_MAX_ITERS && not_converge; iter++) {
        event_record_time_start(stream_name);
        // 1) 分配簇标签
        cluster_assign_kernel<<<K, tpb, 0, stream>>>(d_sub_queries.data().get(),
                                                     d_centroid_ids.data().get(),
                                                     d_labels.data().get(), bp, K, sub_n);

        // 2) 聚合更新质心
        const size_t shared_size = tpb * sizeof(float) + (tpb + 1) * (sizeof(int));
        assert(shared_size < shared_mem_per_block);
        thrust::fill(thrust::cuda::par.on(stream), d_cluster_sizes.begin(), d_cluster_sizes.end(),
                     0);
        compute_centroid_kernel<<<512, 256, shared_size, stream>>>(
            d_sub_queries.data().get(), d_labels.data().get(), d_centers.data().get(),
            d_centroid_ids.data().get(), d_not_converge, d_cluster_sizes.data().get(), bp, K,
            sub_n);

        CUDA_CHECK(cudaMemcpyAsync(&not_converge, d_not_converge, sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));

        time = event_record_time_stop(stream_name, "kmeans_loop") / 1000.0;
        cudaMemcpy(centroid_ids.data(), d_centroid_ids.data().get(), K * sizeof(int),
                   cudaMemcpyDeviceToHost);
        std::string idstr;
        if (!iter)
            for (int i = 0; i < K; i++) idstr += std::to_string(centroid_ids[i]) + " ";
        else
            for (int i = 0; i < K; i++)
                idstr += (centroid_ids[i] == last_centroid_ids[i]
                              ? std::string(numDigits(centroid_ids[i]), '_')
                              : std::to_string(centroid_ids[i])) +
                         " ";

        last_centroid_ids = centroid_ids;  // deepcopy

        fo.print("Kmeans Step " + TOS(iter) + " time: " + TOS(time) + "s\n" + idstr);
    }

    event_record_time_start(stream_name);

    CUDA_CHECK(cudaFreeAsync(d_not_converge, stream));
    fo.print(not_converge ? "Not Converged but finished!" : "Converged!");

    // 对所有查询分配簇标签
    cluster_assign_kernel<<<K, tpb, 0, stream>>>(d_queries.data().get(),
                                                 d_centroid_ids.data().get(),
                                                 d_labels.data().get(), bp, K, bp.query_n);
    // 统计每个聚类的查询数量，然后求前缀和
    thrust::fill(thrust::cuda::par.on(stream), d_cluster_sizes.begin(), d_cluster_sizes.end(), 0);
    cluster_size_statistics_kernel<<<256, tpb, K * sizeof(int), stream>>>(
        d_labels.data().get(), d_cluster_sizes.data().get(), K, bp.query_n);
    prefix_exclusive_sum(d_cluster_sizes.data().get(), d_cluster_offsets.data().get(), K,
                         stream);  // 求前缀和
    // std::vector<int> cluster_sizes(K), cluster_offsets(K);
    // std::string size_str, offset_str;
    // cudaMemcpy(cluster_sizes.data(), d_cluster_sizes.data().get(), K * sizeof(int),
    //            cudaMemcpyDeviceToHost);
    // cudaMemcpy(cluster_offsets.data(), d_cluster_offsets.data().get(), K * sizeof(int),
    //            cudaMemcpyDeviceToHost);
    // for (int i = 0; i < K; i++) {
    //     size_str += std::to_string(cluster_sizes[i]) + " ";
    //     offset_str += std::to_string(cluster_offsets[i]) + " ";
    // }
    // fo.print("Cluster Sizes: " + size_str + "\nCluster Offsets: " + offset_str);

    thrust::device_vector<float> d_queries_temp(bp.query_n * bp.dim);
    query_cluster_bucket_kernel<<<K, tpb, sizeof(int), stream>>>(
        d_queries.data().get(), d_labels.data().get(), d_cluster_offsets.data().get(),
        d_queries_temp.data().get(), K, bp);
    query_round_robin_kernel<<<256, tpb, 0, stream>>>(
        d_queries_temp.data().get(), d_cluster_sizes.data().get(), d_cluster_offsets.data().get(),
        d_queries.data().get(), K, bp);
    CUDA_CHECK(cudaMemcpyAsync(h_queries, d_queries.data().get(), query_data_size,
                               cudaMemcpyDeviceToHost, stream));

    event_record_time_stop(stream_name, "kmeans") / 1000.0;

    auto duration = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::high_resolution_clock::now() - start_time);
    fo.iprint("Total kmeans time cost: " + TOS(duration.count()) + "s");
}

}  // namespace efanna2e