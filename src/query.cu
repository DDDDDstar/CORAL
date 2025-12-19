#include <cfloat>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>

#include "gpufuncs.cuh"
#include "uni.h"
#include "utils.cuh"

namespace efanna2e {
__global__ void save_centroids_kernel(const float* __restrict__ queries,
                                      const int* __restrict__ centroid_ids,
                                      float* __restrict__ centroids_out, int K, int dim) {
    const int block_num = gridDim.x, tpb = blockDim.x, bid = blockIdx.x, tid = threadIdx.x;
    extern __shared__ int centroid_id[];
    for (int i = bid; i < K; i += block_num) {
        if (tid == 0) *centroid_id = centroid_ids[i];
        __syncthreads();

        float* centroid_start = centroids_out + i * dim;
        const float* queries_start = queries + *centroid_id * dim;
        for (int d = tid; d < dim; d += tpb) centroid_start[d] = queries_start[d];
    }
}
__global__ void query_cluster_bucket_kernel(const float* __restrict__ queries,
                                            const int* __restrict__ labels,
                                            const int* __restrict__ cluster_offsets,
                                            float* __restrict__ bucket_queries,
                                            int* __restrict__ query_ids, int K, BP bp, int N) {
    const int block_num = gridDim.x, tpb = blockDim.x, bid = blockIdx.x, tid = threadIdx.x;
    extern __shared__ int sh_cluster_num[];
    for (int cluster_idx = bid; cluster_idx < K; cluster_idx += block_num) {
        if (tid == 0) *sh_cluster_num = 0;
        __syncthreads();

        const int offset = cluster_idx == 0 ? 0 : cluster_offsets[cluster_idx - 1];
        float* queries_start = bucket_queries + offset * bp.dim;
        int* ids_start = query_ids + offset;
        for (int i = tid; i < N; i += tpb) {
            if (labels[i] != cluster_idx) continue;
            const int idx = atomicAdd(sh_cluster_num, 1);
            ids_start[idx] = i;
            for (int d = 0; d < bp.dim; ++d)
                queries_start[idx * bp.dim + d] = queries[i * bp.dim + d];
        }
    }
}

__global__ void cluster_search_kernel(const float* __restrict__ queries,
                                      const float* __restrict__ centroids,
                                      const float* __restrict__ bucket_vecs,
                                      const int* __restrict__ bucket_vec_ids,
                                      const int* __restrict__ cluster_offsets,
                                      int* __restrict__ closest_ids_out, int query_num, int K,
                                      BP bp) {
    const int block_num = gridDim.x, tpb = blockDim.x, bid = blockIdx.x, tid = threadIdx.x;
    extern __shared__ float s_query[];
    float* s_dists = s_query + bp.dim;
    int *s_ids = (int*)(s_dists + K), *s_start = s_ids + K, *s_end = s_start + 1;
    for (int qid = bid; qid < query_num; qid += block_num) {
        const float* query = queries + qid * bp.dim;
        for (int d = tid; d < bp.dim; d += tpb) s_query[d] = query[d];
        __syncthreads();

        for (int cid = tid; cid < K; cid += tpb) {
            s_dists[cid] = calc_distance(s_query, centroids, cid, bp);
            s_ids[cid] = cid;
        }
        __syncthreads();

        for (int offset = K / 2; offset > 0; offset /= 2)
            for (int i = tid; i < offset; i += tpb)
                if (compare_dist(s_dists[i], s_dists[i + offset], bp)) {
                    s_dists[i] = s_dists[i + offset];
                    s_ids[i] = s_ids[i + offset];
                }
        __syncthreads();

        if (tid == 0) {
            *s_start = s_ids[0] == 0 ? 0 : cluster_offsets[s_ids[0] - 1];
            *s_end = cluster_offsets[s_ids[0]];
        }
        __syncthreads();
        // assert(tpb <= K);
        s_dists[tid] = farthest_dist_d(bp);
        for (int i = *s_start + tid; i < *s_end; i += tpb) {
            const float dist = calc_distance(s_query, bucket_vecs, i, bp);
            if (compare_dist(s_dists[tid], dist, bp)) {
                s_dists[tid] = dist;
                s_ids[tid] = bucket_vec_ids[i];
            }
        }
        __syncthreads();

        for (int offset = tpb / 2; offset > 0; offset /= 2)
            if (compare_dist(s_dists[tid], s_dists[tid + offset], bp)) {
                s_dists[tid] = s_dists[tid + offset];
                s_ids[tid] = s_ids[tid + offset];
            }
        __syncthreads();

        if (tid == 0) closest_ids_out[qid] = s_ids[0];
    }
}

void QueryIndex::Top1_Search(const float* h_queries, int* d_closest_ids_out, int num,
                             cudaStream_t& stream) {
    thrust::device_vector<float> d_queries(bp.dim * num);

    CUDA_CHECK(cudaMemcpyAsync(d_queries.data().get(), h_queries, bp.dim * num * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    cluster_search_kernel<<<128, 256, (bp.dim + K) * sizeof(float) + (K + 2) * sizeof(int),
                            stream>>>(d_queries.data().get(), d_centroids.data().get(),
                                      d_bucket_vecs.data().get(), d_bucket_vec_ids.data().get(),
                                      d_cluster_offsets.data().get(), d_closest_ids_out, num, K,
                                      bp);
}

QueryIndex::QueryIndex(const std::string& filename, int K, int k, int dim, DIST_METRIC metric)
    : d_cluster_offsets(K, 0), d_centroids(K * dim), bp(dim, metric, k), K(K) {
    std::ifstream ifs(filename, std::ios::binary);

    ifs.read(reinterpret_cast<char*>(&query_index_size), sizeof(int));
    const int total_size = query_index_size * bp.dim * sizeof(float);
    float* h_queries;
    CUDA_CHECK(cudaMallocHost(&h_queries, total_size));
    ifs.read(reinterpret_cast<char*>(h_queries), total_size);

    std::vector<int> knns(query_index_size * bp.k);
    ifs.read(reinterpret_cast<char*>(knns.data()), query_index_size * bp.k * sizeof(int));
    query_knns = new Query_KNNs(knns, query_index_size, bp);

    thrust::device_vector<float> d_queries(query_index_size * bp.dim);
    CUDA_CHECK(cudaMemcpy(d_queries.data().get(), h_queries, total_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaFreeHost(h_queries));

    int *d_not_converge, not_converge = 1;
    CUDA_CHECK(cudaMalloc(&d_not_converge, sizeof(int)));

    // 随机采样子集
    const int sub_n = KMEANS_SIZE_RATIO * query_index_size;
    thrust::device_vector<int> d_idxs(query_index_size);
    thrust::sequence(d_idxs.begin(), d_idxs.end());  // 索引 [0..N-1]
    thrust::shuffle(d_idxs.begin(), d_idxs.end(), thrust::default_random_engine(1234));
    d_idxs.resize(sub_n);  // 保留前 sub_n 个

    thrust::device_vector<float> d_sub_queries((size_t)sub_n * bp.dim);
    // gather 子集数据
    const int tpb = 256;
    gather_sub_queries_kernel<<<(sub_n + tpb - 1) / tpb, tpb, 0>>>(
        d_queries.data().get(), d_idxs.data().get(), d_sub_queries.data().get(), sub_n, bp.dim);

    // 从子集随机选择 K 个点作为初始中心
    thrust::sequence(d_idxs.begin(), d_idxs.begin() + sub_n);
    thrust::shuffle(d_idxs.begin(), d_idxs.begin() + sub_n, thrust::default_random_engine(5678));
    // d_idxs.resize(K);
    // int* d_centroid_ids = d_idxs.data().get();
    // centroid_init_kernel<<<K, 128>>>(d_sub_queries.data().get(), d_idxs.data().get(),
    //                                  d_centroids.data().get(), bp, K);
    gather_sub_queries_kernel<<<256, tpb>>>(d_sub_queries.data().get(), d_idxs.data().get(),
                                            d_centroids.data().get(), K, bp.dim);
    thrust::device_vector<int> d_labels(query_index_size);
    thrust::device_vector<float> d_vec_sums(K * tpb * bp.dim);

    for (int iter = 0; iter < KMEANS_MAX_ITERS; iter++) {  // 迭代 K-means
        CUDA_CHECK(cudaMemset(d_not_converge, 0, sizeof(int)));
        // 1) 分配簇标签
        cluster_assign_kernel<<<K, tpb, (bp.dim + K) * sizeof(float) + K * sizeof(int)>>>(
            d_sub_queries.data().get(), d_centroids.data().get(), d_labels.data().get(),
            d_not_converge, bp, K, sub_n);
        CUDA_CHECK(cudaMemcpy(&not_converge, d_not_converge, sizeof(int), cudaMemcpyDeviceToHost));
        if (iter && !not_converge) break;

        // 2) 聚合更新质心
        const size_t shared_size = tpb * sizeof(float) + (tpb + 1) * (sizeof(int));
        compute_centroid_kernel<<<K, tpb, shared_size>>>(
            d_sub_queries.data().get(), d_labels.data().get(), d_vec_sums.data().get(),
            d_centroids.data().get(), bp, K, sub_n);
    }
    CUDA_CHECK(cudaFree(d_not_converge));
    fo.print(not_converge ? "QueryIndex Not Converged but finished!" : "QueryIndex Converged!");

    // save_centroids_kernel<<<K, 128, sizeof(int)>>>(d_queries.data().get(), d_centroid_ids,
    //                                                d_centroids.data().get(), K, dim);

    // 对所有查询分配簇标签
    cluster_assign_kernel<<<K, tpb, (bp.dim + K) * sizeof(float) + K * sizeof(int)>>>(
        d_queries.data().get(), d_centroids.data().get(), d_labels.data().get(), nullptr, bp, K,
        query_index_size);

    // 统计每个聚类的查询数量，然后求前缀和
    thrust::device_vector<int> d_cluster_sizes(K);
    thrust::fill(d_cluster_sizes.begin(), d_cluster_sizes.end(), 0);
    cluster_size_statistics_kernel<<<K, tpb, K * sizeof(int)>>>(
        d_labels.data().get(), d_cluster_sizes.data().get(), K, query_index_size);
    prefix_exclusive_sum(d_cluster_sizes.data().get(), d_cluster_offsets.data().get(),
                         K);  // 求前缀和

    d_bucket_vecs.resize(query_index_size * bp.dim);
    d_bucket_vec_ids.resize(query_index_size);
    query_cluster_bucket_kernel<<<K, tpb, sizeof(int)>>>(
        d_queries.data().get(), d_labels.data().get(), d_cluster_offsets.data().get(),
        d_bucket_vecs.data().get(), d_bucket_vec_ids.data().get(), K, bp, query_index_size);
}

};  // namespace efanna2e