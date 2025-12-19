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

__device__ __forceinline__ void insert_local_best(int* ids, float* dists, int len, int id,
                                                  float dist, BP bp) {
    // 保持 arr[0..len-1] 为当前的 top-k
    if (ids[len - 1] < 0) {
        ids[len - 1] = id;
        dists[len - 1] = dist;
        return;
    }
    if (ids[0] >= 0 && compare_dist(dist, dists[0], bp)) return;

    dists[0] = dist;  // 放到 arr[0]
    ids[0] = id;

    for (int i = 1; i < len; ++i) {  // 简单上浮到正确位置，因为 k 很小，直接插入排序
        if (ids[i] >= 0 && compare_dist(dists[i - 1], dists[i], bp)) break;

        const float tmp_dist = dists[i];
        const int tmp_id = ids[i];
        dists[i] = dists[i - 1];
        dists[i - 1] = tmp_dist;
        ids[i] = ids[i - 1];
        ids[i - 1] = tmp_id;
    }
}

__global__ void compute_knn_dist_kernel(const float* __restrict__ queries,
                                        const float* __restrict__ bases, float* dists, int* idxs,
                                        int batch, BP bp, int n) {
    const int qid = blockIdx.x % batch, bid = blockIdx.x / batch, tid = threadIdx.x,
              tpb = blockDim.x, blocks_per_query = gridDim.x / batch;
    extern __shared__ float query_vec[];  // 使用 shared memory 缓存当前查询向量，提高访存效率
    for (int d = tid; d < bp.dim; d += tpb) query_vec[tid] = __ldg(&queries[qid * bp.dim + d]);
    __syncthreads();

    // 每个线程负责计算部分 n 向量的距离
    float* dists_out = dists + qid * n;
    int* idxs_out = idxs + qid * n;
    for (int i = bid * tpb + tid; i < n; i += tpb * blocks_per_query) {
        const float* base_vec = bases + i * bp.dim;  // 当前基础向量
        dists_out[i] = bp.metric == DIST_METRIC::L2 ? l2_distance(query_vec, base_vec, bp.dim)
                                                    : -ip_distance(query_vec, base_vec, bp.dim);
        idxs_out[i] = i;
    }
}

__global__ void compute_knn_dist_kernel(const float* __restrict__ query,
                                        const float* __restrict__ bases, float* dists, int* idxs,
                                        BP bp, int base_n) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    extern __shared__ float s_query[];
    for (int d = tid; d < bp.dim; d += tpb) s_query[d] = query[d];
    __syncthreads();

    for (int i = bid * tpb + tid; i < base_n; i += gridDim.x * tpb) {
        dists[i] = bp.metric == DIST_METRIC::L2 ? l2_distance(s_query, bases, i, bp.dim)
                                                : -ip_distance(s_query, bases, i, bp.dim);
        idxs[i] = i;
    }
}

// __global__ void compute_knn_dist_kernel(const float* __restrict__ query,
//                                         const float* __restrict__ bases, float* dists, int*
//                                         idxs, BP bp, int base_n, int start_base_id) {
//     const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
//     extern __shared__ float s_query[];
//     for (int d = tid; d < bp.dim; d += tpb) s_query[d] = query[d];
//     __syncthreads();

//     for (int i = bid * tpb + tid; i < base_n; i += gridDim.x * tpb) {
//         dists[i] = calc_distance(s_query, bases, i, bp);
//         idxs[i] = start_base_id + i;
//     }
// }

__global__ void topk_copy_kernel(int* d_knn_res, const int* d_all_idx, int batch, BP bp, int n) {
    // 每个 block 处理一个 KNN 结果拷贝
    for (int qidx = blockIdx.x; qidx < batch; qidx += gridDim.x) {
        const int src_offset = qidx * n, dst_offset = qidx * bp.k;
        for (int i = threadIdx.x; i < bp.k; i += blockDim.x) {
            const int idx = bp.metric == DIST_METRIC::L2 ? i : n - 1 - i;
            d_knn_res[dst_offset + i] = d_all_idx[src_offset + idx];
        }
    }
}

__global__ void topk_copy_kernel(int* d_knn_res, const int* d_all_idxs, float* d_dist_res,
                                 const float* d_all_dists, int batch, BP bp, int n) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    extern __shared__ float s_dists[];
    int* s_idxs = (int*)(s_dists + bp.k);
    for (int batch_i = bid; batch_i < batch; batch_i += gridDim.x) {
        const int* idxs = d_all_idxs + bid * n;
        const float* dists = d_all_dists + bid * n;
        int* old_idxs = d_knn_res + bid * bp.k;
        float* old_dists = d_dist_res + bid * bp.k;
        if (old_idxs[0] == -1) {
            for (int i = tid; i < bp.k; i += tpb) {
                old_idxs[i] = idxs[i];
                old_dists[i] = dists[i];
            }
            continue;
        }
        for (int i = tid; i < bp.k; i += tpb) {
            s_idxs[i] = old_idxs[i];
            s_dists[i] = old_dists[i];
        }
        __syncthreads();

        if (tid == 0) {
            int ia = 0, ib = 0;
            for (int i = 0; i < bp.k; i++) {
                if (ib == bp.k || (ia < bp.k && dists[ia] <= s_dists[ib])) {
                    old_idxs[i] = idxs[ia];
                    old_dists[i] = dists[ia++];
                } else {
                    old_idxs[i] = s_idxs[ib];
                    old_dists[i] = s_dists[ib++];
                }
            }
        }
        __syncthreads();
    }
}

__global__ void local_topk_kernel(const float* __restrict__ queries,
                                  const float* __restrict__ bases,
                                  float* __restrict__ partial_dists_out,
                                  int* __restrict__ partial_ids_out, int N, int elems_per_block,
                                  int blocks_per_batch, int batch, BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x, M = tpb * T_LOCAL,
              batch_i = bid / blocks_per_batch;
    if (batch_i >= batch) return;

    const int block_i = bid % blocks_per_batch, start = block_i * elems_per_block;
    if (start >= N) return;
    const int end = min(start + elems_per_block, N);

    extern __shared__ unsigned char smem[];
    float *s_query = (float*)smem, *s_candidates_dist = s_query + bp.dim;
    int* s_candidates = (int*)(s_candidates_dist + M);

    const float *query = queries + batch_i * bp.dim, farthest = farthest_dist_d(bp);
    for (int d = tid; d < bp.dim; d += tpb) s_query[d] = query[d];
    __syncthreads();

    float local_dists[T_LOCAL];  // Per-thread local candidate array (in registers)
    int local_ids[T_LOCAL];
#pragma unroll
    for (int i = 0; i < T_LOCAL; ++i) {
        local_dists[i] = farthest;
        local_ids[i] = -1;
    }
    for (int idx = start + tid; idx < end; idx += tpb)  // compute block's element range
        insert_local_best(local_ids, local_dists, T_LOCAL, idx,
                          calc_distance(s_query, bases, idx, bp), bp);
    // Now each thread writes its T_LOCAL candidates into shared memory candidates buffer
    const int write_pos = tid * T_LOCAL;
#pragma unroll
    for (int i = 0; i < T_LOCAL; ++i) {
        s_candidates[write_pos + i] = local_ids[i];
        s_candidates_dist[write_pos + i] = local_dists[i];
    }
    __syncthreads();

    if (tid == 0) {  // Single thread (or few threads) reduces all candidates into block_topk
        const int total_candidates = tpb * T_LOCAL;
        int block_top[100];
        float block_top_dist[100];
        for (int i = 0; i < bp.k; ++i) {
            block_top_dist[i] = farthest;
            block_top[i] = -1;
        }
        for (int i = 0; i < total_candidates; ++i)
            insert_local_best(block_top, block_top_dist, bp.k, s_candidates[i],
                              s_candidates_dist[i], bp);

        float* dist_out = partial_dists_out + bid * bp.k;
        int* id_out = partial_ids_out + bid * bp.k;
        for (int i = 0; i < bp.k; ++i) {
            id_out[i] = block_top[bp.k - 1 - i];
            dist_out[i] = block_top_dist[bp.k - 1 - i];
        }
    }
}

__global__ void pairwise_merge_kernel(const int* __restrict__ partial_ids_in,
                                      const float* __restrict__ partial_dists_in,
                                      int* __restrict__ partial_ids_out,
                                      float* __restrict__ partial_dists_out, int M, BP bp) {
    // each thread block handles one merged output (one pair)
    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;  // allow many threads
    if (out_idx >= (M / 2)) return;

    const int a_idx = out_idx * 2, b_idx = a_idx + 1;
    const float *A_dist = partial_dists_in + a_idx * bp.k,
                *B_dist = partial_dists_in + b_idx * bp.k;
    const int *A = partial_ids_in + a_idx * bp.k, *B = partial_ids_in + b_idx * bp.k;
    float* O_dist = partial_dists_out + out_idx * bp.k;
    int* O = partial_ids_out + out_idx * bp.k;

    int ia = 0, ib = 0, io = 0;
    while (io < bp.k) {
        float va = (ia < bp.k) ? A_dist[ia] : farthest_dist_d(bp);
        float vb = (ib < bp.k) ? B_dist[ib] : farthest_dist_d(bp);
        if (compare_dist(vb, va, bp)) {
            O_dist[io] = va;
            O[io++] = A[ia++];
        } else {
            O_dist[io] = vb;
            O[io++] = B[ib++];
        }
        if (ia >= bp.k && ib >= bp.k) break;
    }
    // if fewer than K placed (should not happen), fill -inf
    for (; io < bp.k; ++io) O_dist[io] = farthest_dist_d(bp);
}

int* GPUFuncs::knn_compute(const float* h_queries, int start_id, int batch, float& time_s) {
    cudaStream_t stream = streams["knn"].stream;

    const float* queries_ptr;
    thrust::device_vector<float> queries;
    // thrust::device_vector<float> d_knn_dists(batch * bp.k);
    // thrust::device_vector<int> d_knns(batch * bp.k);
    if (start_id == -1) {
        queries.resize(batch * bp.dim);
        CUDA_CHECK(cudaMemcpyAsync(queries.data().get(), h_queries, batch * bp.dim * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        queries_ptr = queries.data().get();
    } else
        queries_ptr = d_q + start_id * bp.dim;

    thrust::fill(thrust::cuda::par.on(stream), d_knn_res.begin(), d_knn_res.end(), -1);
    float comp_time = 0, sort_time = 0;
    for (int start_bid = 0; start_bid < bp.base_n; start_bid += bp.gpu_n) {
        const int n = std::min(bp.base_n - start_bid, bp.gpu_n);
        float time;
        // fo.print("start_bid: " + TOS(start_bid) + ", n: " + TOS(n));
        for (int qidx = 0; qidx < batch; qidx++) {
            auto e = gpu_record_time_start(stream);
            if (round_copy)
                CUDA_CHECK(cudaMemcpyAsync(d_b.data().get(), h_base + start_bid * bp.dim,
                                           n * bp.dim * sizeof(float), cudaMemcpyHostToDevice,
                                           stream));

            const int threads = 512, blocks = 1024;
            compute_knn_dist_kernel<<<blocks, threads, bp.dim * sizeof(float), stream>>>(
                queries_ptr + qidx + bp.dim, d_b.data().get(), d_all_dist, d_all_idx, bp, n);
            e = gpu_record_time_reset(e, stream, time);
            comp_time += time;
            sort_pairs(d_all_dist, d_all_idx, n, stream);
            // auto res = segmented_sort_pairs(d_all_idx, d_all_dist, d_all_idx_sort,
            // d_all_dist_sort,
            //                                 d_topk_offsets, stream, batch, n);
            // CUDA_CHECK(cudaMemcpyAsync(d_knn_dists.data().get() + qidx * bp.k, d_all_dist,
            //                            bp.k * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            // CUDA_CHECK(cudaMemcpyAsync(d_knns.data().get() + qidx * bp.k, d_all_idx,
            //                            bp.k * sizeof(int), cudaMemcpyDeviceToDevice, stream));

            topk_copy_kernel<<<128, 64, bp.k * (sizeof(int) + sizeof(float)), stream>>>(
                d_knn_res.data().get(), d_all_idx, d_knn_dists_res.data().get(), d_all_dist, batch,
                bp, n);

            sort_time += gpu_record_time_stop(e, stream);
        }
    }
    time_s += comp_time + sort_time;

#ifdef INFO_PRINT
    fo.print("knn_compute(" + TOS(comp_time) + " s, " + TOS(sort_time) + "): batch_id-" +
             TOS(start_id / BATCH));
#endif

    return d_knn_res.data().get();
}

// int* GPUFuncs::knn_compute_nosort(const float* h_queries, int start_id, int batch, float&
// time_s) {
//     cudaStream_t& stream = streams["knn"].stream;
//     auto e = gpu_record_time_start(stream);

//     const float* queries_ptr;
//     thrust::device_vector<float> queries;
//     if (start_id == -1) {
//         queries.resize(batch * bp.dim);
//         CUDA_CHECK(cudaMemcpyAsync(queries.data().get(), h_queries, batch * bp.dim *
//         sizeof(float),
//                                    cudaMemcpyHostToDevice, stream));
//         queries_ptr = queries.data().get();
//     } else
//         queries_ptr = d_q + start_id * bp.dim;

//     for (int start_bid = 0; start_bid < bp.base_n; start_bid += bp.gpu_n) {
//         const int n = std::min(bp.base_n - start_bid, bp.gpu_n);
//         // fo.print("start_bid: " + TOS(start_bid) + ", n: " + TOS(n));
//         if (round_copy)
//             CUDA_CHECK(cudaMemcpyAsync(d_b.data().get(), h_base + start_bid * bp.dim,
//                                        n * bp.dim * sizeof(float), cudaMemcpyHostToDevice,
//                                        stream));

//         // for (int qidx = 0; qidx < batch; qidx++) {
//         const int threads = 512, blocks = 1024;
//         // compute_knn_dist_kernel<<<blocks, threads, bp.dim * sizeof(float), stream>>>(
//         //     queries_ptr + qidx * bp.dim, d_b.data().get(), d_all_dist, d_all_idx, bp, n,
//         //     start_bid);

//         const int shared_bytes =
//             knn_tpb * T_LOCAL * (sizeof(int) + sizeof(float)) + sizeof(float) * bp.dim;
//         local_topk_kernel<<<blocks_per_batch * 10, knn_tpb, shared_bytes, stream>>>(
//             queries_ptr, d_b.data().get(), partial_dists_a, partial_ids_a, n,
//             knn_tpb * items_per_thread, bp);

//         int M = knn_block_num;
//         if (start_bid > 0) {
//             M++;
//             CUDA_CHECK(cudaMemcpyAsync(partial_ids_a + knn_block_num * bp.k,
//                                        last_topk_ids + qidx * bp.k, bp.k * sizeof(int),
//                                        cudaMemcpyDeviceToDevice, stream));
//             CUDA_CHECK(cudaMemcpyAsync(partial_dists_a + knn_block_num * bp.k,
//                                        last_topk_dists + qidx * bp.k, bp.k * sizeof(float),
//                                        cudaMemcpyDeviceToDevice, stream));
//         }
//         float *cur_dists_in = partial_dists_a, *cur_dists_out = partial_dists_b;
//         int *cur_ids_in = partial_ids_a, *cur_ids_out = partial_ids_b;
//         while (M > 1) {
//             const int out_pairs = M / 2, tpb = 32, block_num = (out_pairs + tpb - 1) / tpb;
//             pairwise_merge_kernel<<<block_num, tpb, 0, stream>>>(
//                 cur_ids_in, cur_dists_in, cur_ids_out, cur_dists_out, M, bp);
//             if (M % 2 == 1) {
//                 CUDA_CHECK(cudaMemcpyAsync(cur_ids_out + out_pairs * bp.k,
//                                            cur_ids_in + (M - 1) * bp.k, bp.k * sizeof(int),
//                                            cudaMemcpyDeviceToDevice, stream));
//                 CUDA_CHECK(cudaMemcpyAsync(cur_dists_out + out_pairs * bp.k,
//                                            cur_dists_in + (M - 1) * bp.k, bp.k * sizeof(float),
//                                            cudaMemcpyDeviceToDevice, stream));
//                 M = out_pairs + 1;
//             } else
//                 M = out_pairs;

//             float* tmp_dists = cur_dists_in;
//             int* tmp_ids = cur_ids_in;
//             cur_dists_in = cur_dists_out;
//             cur_ids_in = cur_ids_out;
//             cur_dists_out = tmp_dists;
//             cur_ids_out = tmp_ids;
//         }
//         CUDA_CHECK(cudaMemcpyAsync(last_topk_ids + qidx * bp.k, cur_ids_in, bp.k * sizeof(int),
//                                    cudaMemcpyDeviceToDevice, stream));
//         CUDA_CHECK(cudaMemcpyAsync(last_topk_dists + qidx * bp.k, cur_dists_in,
//                                    bp.k * sizeof(float), cudaMemcpyDeviceToDevice, stream));
//         // }
//     }
//     const float time = gpu_record_time_stop(e, stream);
//     time_s += time;
// #ifdef INFO_PRINT
//     fo.print("knn_compute_nosort(" + TOS(time) + " s): batch_id-" + TOS(start_id / BATCH));
// #endif

//     // {
//     //     std::unique_lock<std::shared_mutex> lock(base_query_mtx);
//     //     base_query_update_kernel<<<batch, 64, 0, stream>>>(
//     //         last_topk_ids, last_topk_dists, base_query_ids, base_query_dists, start_id,
//     //         batch, bp);
//     // }

//     return last_topk_ids;
// }

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

void GPUFuncs::knn_prepare(const float* h_queries) {
    cudaStream_t stream = streams["knn"].stream;
    // 基础向量数据和查询向量数据  GPU 内存分配和数据传输（host->device）
    for (size_t i = 0; i < bp.base_n; i += bp.gpu_n)
        knn_index.add(std::min<size_t>(bp.gpu_n, bp.base_n - i), h_base + i * bp.dim);

    if (bp.base_n <= bp.gpu_n) {
        round_copy = false;
        CUDA_CHECK(cudaMemcpyAsync(d_b.data().get(), h_base, bp.base_n * bp.dim * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaMallocAsync(&d_q, bp.query_n * bp.dim * sizeof(float), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_q, h_queries, bp.query_n * bp.dim * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

    // CUDA_CHECK(cudaMallocAsync(&d_knn_res, BATCH * bp.k * sizeof(int), stream));
    d_knn_res.resize(BATCH * bp.k);
    d_knn_dists_res.resize(BATCH * bp.k);
    // CUDA_CHECK(cudaMallocAsync(&d_knn_dists_res, BATCH * bp.k * sizeof(float), stream));
    // if (bp.gpu_n < bp.base_n) {
    //     const int round = bp.base_n / bp.gpu_n + 1;
    //     CUDA_CHECK(cudaMallocAsync(&d_knn_1b, BATCH * bp.k * round * sizeof(int), stream));
    //     CUDA_CHECK(cudaMallocAsync(&d_knn_dists_1b, BATCH * bp.k * round * sizeof(float),
    //     stream));
    // }
    // knn_compute 预分配内存
    CUDA_CHECK(cudaMallocAsync(&d_all_dist, bp.gpu_n * sizeof(float), stream));
    // CUDA_CHECK(cudaMallocAsync(&d_all_dist_sort, BATCH * bp.gpu_n * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_all_idx, bp.gpu_n * sizeof(int), stream));
    // CUDA_CHECK(cudaMallocAsync(&d_all_idx_sort, BATCH * bp.gpu_n * sizeof(int), stream));
    // CUDA_CHECK(cudaMallocAsync(&d_topk_offsets, (BATCH + 1) * sizeof(int), stream));

    // CUDA_CHECK(cudaMallocAsync(&base_query_ids, bp.base_n * sizeof(int), stream));
    // CUDA_CHECK(cudaMallocAsync(&base_query_dists, bp.base_n * sizeof(float), stream));
    // thrust::device_ptr<int> base_query_ids_ptr(base_query_ids);
    // thrust::device_ptr<float> base_query_dists_ptr(base_query_dists);
    // thrust::fill(thrust::cuda::par.on(stream), base_query_ids_ptr, base_query_ids_ptr +
    // bp.base_n,
    //              -1);
    // thrust::fill(thrust::cuda::par.on(stream), base_query_dists_ptr,
    //              base_query_dists_ptr + bp.base_n, farthest_dist(bp));

    blocks_per_batch = (bp.gpu_n + knn_tpb * items_per_thread - 1) / (knn_tpb * items_per_thread);
    const int partial_size = (blocks_per_batch + 1) * bp.k;
    const size_t total_memsize =
        2 * BATCH * (partial_size * sizeof(int) + partial_size * sizeof(float));
    fo.iprint("knn mem size: " + TOS(total_memsize / 1024.0 / 1024 / 1024) +
              " G, blocks_per_batch: " + TOS(blocks_per_batch));
    CUDA_CHECK(cudaMallocAsync(&partial_ids_a, BATCH * partial_size * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&partial_ids_b, BATCH * partial_size * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&last_topk_ids, BATCH * bp.k * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&partial_dists_a, BATCH * partial_size * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&partial_dists_b, BATCH * partial_size * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&last_topk_dists, BATCH * bp.k * sizeof(float), stream));
}
void GPUFuncs::knn_free() {
    cudaStream_t& stream = streams["knn"].stream;
    CUDA_CHECK(cudaFreeAsync(d_q, stream));
    // CUDA_CHECK(cudaFreeAsync(d_knn_res, stream));
    // CUDA_CHECK(cudaFreeAsync(d_knn_dists_res, stream));
    CUDA_CHECK(cudaFreeAsync(d_all_dist, stream));
    CUDA_CHECK(cudaFreeAsync(d_all_idx, stream));
    // CUDA_CHECK(cudaFreeAsync(d_topk_offsets, stream));
    // CUDA_CHECK(cudaFreeAsync(d_all_dist_sort, stream));
    // CUDA_CHECK(cudaFreeAsync(d_all_idx_sort, stream));
    // CUDA_CHECK(cudaFreeAsync(base_query_ids, stream));
    // CUDA_CHECK(cudaFreeAsync(base_query_dists, stream));
    CUDA_CHECK(cudaFreeAsync(partial_ids_a, stream));
    CUDA_CHECK(cudaFreeAsync(partial_ids_b, stream));
    CUDA_CHECK(cudaFreeAsync(last_topk_ids, stream));
    CUDA_CHECK(cudaFreeAsync(partial_dists_a, stream));
    CUDA_CHECK(cudaFreeAsync(partial_dists_b, stream));
    CUDA_CHECK(cudaFreeAsync(last_topk_dists, stream));
}

GPUFuncs::GPUFuncs(Graph* graph, const float* h_base, float* h_queries, int base_n, int query_n,
                   int dim, int k, int max_degree, DIST_METRIC m, int beam_capacity)
    : construct(true),
      graph(graph),
      bp(dim, max_degree, base_n, query_n, k, beam_capacity, m),
      d_b(GPU_N * bp.dim),
      h_base(h_base),
      knn_index(&gpu_res, dim) {
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
    // query_data_kmeans(h_queries);
    knn_prepare(h_queries);  // knn 预分配内存
    upd_prepare(construct);  // upd 预分配内存
    search_prepare();        // search 预分配内存
    fo.iprint(bp.str());
}

GPUFuncs::GPUFuncs(Graph* graph, QueryIndex* qindex, int base_n, int dim, int k, int max_degree,
                   DIST_METRIC m, int beam_capacity)
    : construct(false),
      graph(graph),
      query_index(qindex),
      bp(dim, max_degree, base_n, 0, k, beam_capacity, m),
      d_b(bp.gpu_n * bp.dim),
      knn_index(&gpu_res, dim) {
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
        // CUDA_CHECK(cudaEventCreate(&streams[name].start));
        // CUDA_CHECK(cudaEventCreate(&streams[name].stop));
    }
    // knn_prepare(h_base, h_queries);  // knn 预分配内存
    upd_prepare(construct);  // upd 预分配内存
    search_prepare();        // search 预分配内存
    update_prepare();
    fo.iprint(bp.str());
}

GPUFuncs::~GPUFuncs() {
    if (construct) {
        knn_free();
        upd_free();
    }
    search_free();
    update_free();

    graph->flush_all_to_disk();
    delete graph;

    for (auto& pair : streams) {
        CUDA_CHECK(cudaStreamDestroy(pair.second.stream));
        CUDA_CHECK(cudaEventDestroy(pair.second.start));
        CUDA_CHECK(cudaEventDestroy(pair.second.stop));
    }
}

// void GPUFuncs::event_record_time_start(const std::string name) {
//     cudaEventRecord(streams[name].start, streams[name].stream);  // 在流中记录开始事件
// }

// float GPUFuncs::event_record_time_stop(const std::string stream_name, std::string func_name) {
//     if (cuda_check_last_error(func_name)) exit(EXIT_FAILURE);

//     float milliseconds = 0;
//     cudaEvent_t& stop = streams[stream_name].stop;
//     cudaEventRecord(stop, streams[stream_name].stream);
//     cudaEventSynchronize(stop);
//     cudaEventElapsedTime(&milliseconds, streams[stream_name].start, stop);

//     record_time(func_name, milliseconds / 1000);

//     return milliseconds;
// }

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

Query_KNNs::Query_KNNs(const std::vector<int>& knns_in, int query_size, BP bp)
    : size(query_size), knns(query_size * bp.k), bp(bp) {
    CUDA_CHECK(cudaMemcpy(knns.data().get(), knns_in.data(), query_size * bp.k * sizeof(int),
                          cudaMemcpyHostToDevice));
}

// void Query_KNNs::record(int qid, int* new_knns, int batch, cudaStream_t& stream) {
//     if (qid >= size) {
//         size *= 2;
//         knns.resize(size);
//     }
//     CUDA_CHECK(cudaMemcpyAsync(knns.data().get() + qid * bp.k, new_knns,
//                                batch * bp.k * sizeof(int), cudaMemcpyDeviceToDevice,
//                                stream));
// }
}  // namespace efanna2e

// int* GPUFuncs::knn_compute(const float* h_queries, int batch, float& time_ms) {
//     const std::string name = "knn";
//     cudaStream_t& stream = streams[name].stream;

//     event_record_time_start(name);

//     thrust::device_vector<float> queries(batch * bp.dim);
//     CUDA_CHECK(cudaMemcpyAsync(queries.data().get(), h_queries, batch * bp.dim *
//     sizeof(float),
//                                cudaMemcpyHostToDevice, stream));
//     // thrust::copy(thrust::cuda::par.on(stream), h_queries, h_queries + batch * bp.dim,
//     //              queries.begin());

//     compute_knn_dist_kernel<<<batch * blocknum_per_query, knn_threads, bp.dim *
//     sizeof(float),
//                               stream>>>(queries.data().get(), d_b.data().get(), d_all_dist,
//                                         d_all_idx, batch, bp);

//     time_ms = event_record_time_stop(name, "knn_dist_compute");
//     // 对每个查询结果进行排序，提取 top-K
//     event_record_time_start(name);

//     auto res = segmented_sort_pairs(d_all_idx, d_all_dist, d_all_idx_sort, d_all_dist_sort,
//                                     d_topk_offsets, stream, batch, bp.base_n);
//     topk_copy_kernel<<<batch, 64, 0, stream>>>(d_knn_res, res.first, batch, bp, bp.base_n);

//     time_ms += event_record_time_stop(name, "knn_sort_compute");
//     return d_knn_res;
// }

// __global__ void base_query_update_kernel(const int* __restrict__ knns,
//                                          const float* __restrict__ all_dists_sort,
//                                          int* __restrict__ base_query_ids,
//                                          float* __restrict__ base_query_dists, int
//                                          start_qid, int batch, BP bp) {
//     const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
//     for (int qidx = bid; qidx < batch; qidx += gridDim.x) {
//         for (int i = tid; i < bp.k; i += tpb) {
//             const int knn_idx = qidx * bp.k + i, base_id = knns[knn_idx];
//             const float dist = all_dists_sort[qidx * bp.base_n + i];
//             // 原子比较更新最近距离和索引:
//             atomicClosestFloat(base_query_dists + base_id, dist, bp);
//             __threadfence();  // 刷写全局/共享内存，使其他 block 可见
//             if (base_query_dists[base_id] - dist < EPS) base_query_ids[base_id] = start_qid
//             + qidx;
//         }
//     }
// }

// __global__ void base_query_update_kernel(const int* __restrict__ knns,
//                                          const float* __restrict__ knn_dists,
//                                          int* __restrict__ base_query_ids,
//                                          float* __restrict__ base_query_dists, int start_qid,
//                                          int batch, BP bp) {
//     const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
//     for (int qidx = bid; qidx < batch; qidx += gridDim.x) {
//         for (int i = tid; i < bp.k; i += tpb) {
//             const int knn_idx = qidx * bp.k + i, base_id = knns[knn_idx];
//             const float dist = knn_dists[qidx * bp.k + i];
//             // 原子比较更新最近距离和索引:
//             atomicClosestFloat(base_query_dists + base_id, dist, bp);
//             __threadfence();  // 刷写全局/共享内存，使其他 block 可见
//             if (base_query_dists[base_id] - dist < EPS) base_query_ids[base_id] = start_qid +
//             qidx;
//         }
//     }
// }
