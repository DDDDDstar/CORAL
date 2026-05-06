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
    int* s_ids = (int*)(s_dists + K);
    int *s_start = (int*)(s_ids + K), *s_end = s_start + 1;
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

__global__ void calc_closest_query_kernel(const float* __restrict__ ins_vecs,
                                          const float* __restrict__ queries,
                                          const int* __restrict__ qids,
                                          int* __restrict__ closest_qids, int batch, int k,
                                          BP bp) {
    const int tid = threadIdx.x, tpb = blockDim.x;
    extern __shared__ int s_qids[];
    float* s_dists = (float*)(s_qids + k);
    const float farthest = farthest_dist_d(bp);
    for (int i = blockIdx.x; i < batch; i += gridDim.x) {
        const float* ins_vec = ins_vecs + i * bp.dim;
        for (int j = tid; j < k; j += tpb) {
            const int qid = s_qids[j] = qids[i * k + j];
            s_dists[j] = qid < 0 ? farthest : calc_distance(ins_vec, queries, qid, bp);
        }
        __syncthreads();

        for (int offset = k / 2; offset > 0; offset /= 2) {
            if (tid < offset && compare_dist(s_dists[tid], s_dists[tid + offset], bp)) {
                s_dists[tid] = s_dists[tid + offset];
                s_qids[tid] = s_qids[tid + offset];
            }
            __syncthreads();
        }

        if (tid == 0) closest_qids[i] = s_qids[0] < 0 ? 0 : s_qids[0];

        __syncthreads();
    }
}

void QueryIndex::get_query_of_base(const float* d_ins_vecs, int* ids, int* h_closest_qids,
                                   int batch, int k, BP bp) {
    query_knns->get_query_of_base(ids, batch * k);
    thrust::device_vector<int> d_qids(batch * k), d_closest_qids(batch);
    cudaMemcpy(d_qids.data().get(), ids, batch * k * sizeof(int), cudaMemcpyHostToDevice);
    calc_closest_query_kernel<<<batch, 128, k * (sizeof(int) + sizeof(float))>>>(
        d_ins_vecs, d_queries, d_qids.data().get(), d_closest_qids.data().get(), batch, k, bp);
    cudaGetLastError();
    cudaMemcpy(h_closest_qids, d_closest_qids.data().get(), batch * sizeof(int),
               cudaMemcpyDeviceToHost);
    query_knns->add_query_of_base(ids, h_closest_qids, batch);
}

void QueryIndex::Top1_Search_faiss(const float* h_queries, std::vector<int>& closest_ids_out,
                                   int num) {
    if (res_faiss.size() < num) {
        res_faiss.resize(num);
        dists_res.resize(num);
    }
    gpu_flat_index->search(num, h_queries, 1, dists_res.data(), res_faiss.data());
    std::transform(res_faiss.begin(), res_faiss.end(), closest_ids_out.begin(),
                   [](faiss::idx_t x) { return static_cast<int>(x); });
}

void QueryIndex::knn_Search_faiss(const float* d_queries, int* h_knns_out, int num,
                                  int query_num) {
    if (res_faiss.size() < num * query_num) {
        res_faiss.resize(num * query_num);
        dists_res.resize(num * query_num);
    }
    gpu_flat_index->search(num, d_queries, query_num, dists_res.data(), res_faiss.data());
    query_knns->get(res_faiss.data(), h_knns_out, num * query_num);
    // query_knns->get(res_faiss.data(), h_knns_out, num * query_num);
    // CUDA_CHECK(cudaMemcpy(d_knns_out, h_knns, num * bp.k * sizeof(int),
    // cudaMemcpyHostToDevice));
}

void QueryIndex::knn_Search_faiss(const float* d_queries, const int* ids,
                                  std::vector<int>& knns_out, size_t* offsets, int num) {
    if (res_faiss.size() < num) {
        res_faiss.resize(num);
        dists_res.resize(num);
    }
    gpu_flat_index->search(num, d_queries, 1, dists_res.data(), res_faiss.data());
    query_knns->get_and_add(res_faiss, ids, knns_out, offsets, num);
    // CUDA_CHECK(cudaMemcpy(d_knns_out, h_knns, num * bp.k * sizeof(int),
    // cudaMemcpyHostToDevice));
}

void QueryIndex::knn_Search_faiss(const float* d_queries, const int* ids, int* knns_out, int num,
                                  int query_num) {
    assert(query_num == 1);
    if (res_faiss.size() < num * query_num) {
        res_faiss.resize(num * query_num);
        dists_res.resize(num * query_num);
    }
    gpu_flat_index->search(num, d_queries, query_num, dists_res.data(), res_faiss.data());
    query_knns->get_and_add(res_faiss, ids, knns_out, num);
    // CUDA_CHECK(cudaMemcpy(d_knns_out, h_knns, num * bp.k * sizeof(int),
    // cudaMemcpyHostToDevice));
}

void QueryIndex::Topk_Search_faiss(const float* h_queries, std::vector<int>& closest_ids_out,
                                   int num, int k) {
    if (res_faiss.size() < num * k) {
        res_faiss.resize(num * k);
        dists_res.resize(num * k);
    }
    gpu_flat_index->search(num, h_queries, k, dists_res.data(), res_faiss.data());
    std::transform(res_faiss.begin(), res_faiss.end(), closest_ids_out.begin(),
                   [](faiss::idx_t x) { return static_cast<int>(x); });
}

// void QueryIndex::build() {
//     const int total_size = query_index_size * bp.dim * sizeof(float);
//     thrust::device_vector<float> d_queries(query_index_size * bp.dim);
//     CUDA_CHECK(cudaMemcpy(d_queries.data().get(), h_queries, total_size,
//     cudaMemcpyHostToDevice)); int *d_not_converge, not_converge = 1;
//     CUDA_CHECK(cudaMalloc(&d_not_converge, sizeof(int)));

//     // 随机采样子集
//     const int sub_n = KMEANS_SIZE_RATIO * query_index_size;
//     thrust::device_vector<int> d_idxs(query_index_size);
//     thrust::sequence(d_idxs.begin(), d_idxs.end());  // 索引 [0..N-1]
//     thrust::shuffle(d_idxs.begin(), d_idxs.end(), thrust::default_random_engine(1234));
//     d_idxs.resize(sub_n);  // 保留前 sub_n 个

//     thrust::device_vector<float> d_sub_queries((size_t)sub_n * bp.dim);
//     // gather 子集数据
//     const int tpb = 256;
//     gather_sub_queries_kernel<<<(sub_n + tpb - 1) / tpb, tpb, 0>>>(
//         d_queries.data().get(), d_idxs.data().get(), d_sub_queries.data().get(), sub_n, bp.dim);

//     // 从子集随机选择 K 个点作为初始中心
//     thrust::sequence(d_idxs.begin(), d_idxs.begin() + sub_n);
//     thrust::shuffle(d_idxs.begin(), d_idxs.begin() + sub_n,
//     thrust::default_random_engine(5678));
//     // d_idxs.resize(K);
//     // int* d_centroid_ids = d_idxs.data().get();
//     // centroid_init_kernel<<<K, 128>>>(d_sub_queries.data().get(), d_idxs.data().get(),
//     //                                  d_centroids.data().get(), bp, K);
//     gather_sub_queries_kernel<<<256, tpb>>>(d_sub_queries.data().get(), d_idxs.data().get(),
//                                             d_centroids.data().get(), K, bp.dim);
//     thrust::device_vector<int> d_labels(query_index_size);
//     thrust::device_vector<float> d_vec_sums(K * tpb * bp.dim);

//     for (int iter = 0; iter < KMEANS_MAX_ITERS; iter++) {  // 迭代 K-means
//         CUDA_CHECK(cudaMemset(d_not_converge, 0, sizeof(int)));
//         // 1) 分配簇标签
//         cluster_assign_kernel<<<K, tpb, (bp.dim + K) * sizeof(float) + K * sizeof(int)>>>(
//             d_sub_queries.data().get(), d_centroids.data().get(), d_labels.data().get(),
//             d_not_converge, bp, K, sub_n);
//         CUDA_CHECK(cudaMemcpy(&not_converge, d_not_converge, sizeof(int),
//         cudaMemcpyDeviceToHost)); if (iter && !not_converge) break;

//         // 2) 聚合更新质心
//         const size_t shared_size = tpb * sizeof(float) + (tpb + 1) * (sizeof(int));
//         compute_centroid_kernel<<<K, tpb, shared_size>>>(
//             d_sub_queries.data().get(), d_labels.data().get(), d_vec_sums.data().get(),
//             d_centroids.data().get(), bp, K, sub_n);
//     }
//     CUDA_CHECK(cudaFree(d_not_converge));
//     fo.print(not_converge ? "QueryIndex Not Converged but finished!" : "QueryIndex Converged!");

//     // save_centroids_kernel<<<K, 128, sizeof(int)>>>(d_queries.data().get(), d_centroid_ids,
//     //                                                d_centroids.data().get(), K, dim);

//     // 对所有查询分配簇标签
//     cluster_assign_kernel<<<K, tpb, (bp.dim + K) * sizeof(float) + K * sizeof(int)>>>(
//         d_queries.data().get(), d_centroids.data().get(), d_labels.data().get(), nullptr, bp, K,
//         query_index_size);

//     // 统计每个聚类的查询数量，然后求前缀和
//     thrust::device_vector<int> d_cluster_sizes(K);
//     thrust::fill(d_cluster_sizes.begin(), d_cluster_sizes.end(), 0);
//     cluster_size_statistics_kernel<<<K, tpb, K * sizeof(int)>>>(
//         d_labels.data().get(), d_cluster_sizes.data().get(), K, query_index_size);
//     prefix_exclusive_sum(d_cluster_sizes.data().get(), d_cluster_offsets.data().get(),
//                          K);  // 求前缀和

//     d_bucket_vecs.resize(query_index_size * bp.dim);
//     d_bucket_vec_ids.resize(query_index_size);
//     query_cluster_bucket_kernel<<<K, tpb, sizeof(int)>>>(
//         d_queries.data().get(), d_labels.data().get(), d_cluster_offsets.data().get(),
//         d_bucket_vecs.data().get(), d_bucket_vec_ids.data().get(), K, bp, query_index_size);
// }

void QueryIndex::build_faiss() {
    auto s = now_time();

    gpu_flat_index = new GPUFlatL2(&gr, bp.dim);
    gpu_flat_index->add(query_index_size, h_queries);

    fo.iprint("Build faiss index(nq=" + TOS(query_index_size) +
              ") for query search: " + TOS(time_diff(s)) + " s");
}

QueryIndex::QueryIndex(const std::string& filename, int base_n, int k, int dim, DIST_METRIC metric)
    : bp(base_n, dim, metric, k)
// d_cluster_offsets(K, 0), d_centroids(K * dim),K(K)
{
    fo.print("Load query data from " + filename);
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open()) fo.eprint("ERROR load query data.");

    ifs.read(reinterpret_cast<char*>(&query_index_size), sizeof(int));
    fo.iprint("QueryIndex size: " + TOS(query_index_size));
    const int total_size = query_index_size * bp.dim * sizeof(float);

    std::vector<int> knns(query_index_size * bp.k);
    if (!ifs.read(reinterpret_cast<char*>(knns.data()), query_index_size * bp.k * sizeof(int)))
        fo.eprint("Error reading knns in query index file!");

    CUDA_CHECK(cudaMallocHost(&h_queries, total_size));
    query_own = true;
    if (!ifs.read(reinterpret_cast<char*>(h_queries), total_size))
        fo.eprint("Error reading queries in query index file!");

    cudaMalloc(&d_queries, total_size);
    cudaMemcpy(d_queries, h_queries, total_size, cudaMemcpyHostToDevice);

    query_knns = new Query_KNNs(knns, query_index_size, bp);

    // build();

    // CUDA_CHECK(cudaFreeHost(h_queries));
}

QueryIndex::QueryIndex(Cache& knn_cache, float* h_queries, int nq, int base_n, int k, int dim,
                       DIST_METRIC metric)
    : query_index_size(nq), h_queries(h_queries), bp(base_n, dim, metric, k) {
    // const auto& results = knn_cache.results;
    assert(nq <= knn_cache.get_query_num());
    // std::vector<int> knns;
    // knns.reserve(query_size * k);
    // for (const auto& res : results) {
    //     assert(res.knn.size() == k * res.batch);
    //     knns.insert(knns.end(), res.knn.begin(), res.knn.end());
    // }
    // assert(knns.size() == query_size * k);

    query_knns = new Query_KNNs(knn_cache, nq, bp);
}

// QueryIndex::QueryIndex(const std::string& filename, GPUResources* gpu_res, int device,
//                        cudaStream_t stream, int base_n, int k, int dim, DIST_METRIC metric)
//     : QueryIndex(filename, base_n, k, dim, metric) {
//     gr = gpu_res;
//     gr->setDefaultStream(device, stream);
// }

void QueryIndex::set_gpu(cudaStream_t stream) {
    gr.setDefaultStream(0, stream);
    build_faiss();
    // if (need_knn) CUDA_CHECK(cudaMallocHost(&h_knns, PC.enhance_batch * bp.k * sizeof(int)));
}

};  // namespace efanna2e