#include <cfloat>
#include <cmath>
#include <cub/cub.cuh>
#include <iostream>
#include <string>
#include <utility>

#include "fileout.h"
#include "gpufuncs.cuh"
#include "utils.cuh"

namespace efanna2e {
__global__ void candidate_ignore_kernel(const float *__restrict__ base, CN *cand_nbrs,
                                        int *sort_idxs, int *nbr_num, const int new_nbr_i,
                                        const int batch, int cand_size, int vec_num, BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    for (int group_id = bid; group_id < batch; group_id += gridDim.x) {
        for (int pivot_idx = tid; pivot_idx < vec_num; pivot_idx += tpb) {
            CN *new_nbr =
                get_CN(cand_nbrs, sort_idxs, new_nbr_i, group_id, pivot_idx, vec_num, cand_size);
            // 无效节点或邻居已满，直接淘汰
            if (new_nbr->id == -1 || nbr_num[group_id * vec_num + pivot_idx] >= bp.max_degree)
                new_nbr->status = Discarded;
            else if (new_nbr->status == Initial)  // 待定节点，保留且用于淘汰后续节点
            {
                new_nbr->status = Retained;
                nbr_num[group_id * vec_num + pivot_idx]++;
                const int wti_num = cand_size - new_nbr_i - 1;
                for (int i = 0; i < wti_num; i++) {
                    CN *wti_cand_nbr = get_CN(cand_nbrs, sort_idxs, i + new_nbr_i + 1, group_id,
                                              pivot_idx, vec_num, cand_size);  // 待淘汰候选邻居
                    if (wti_cand_nbr->status == Initial) {
                        // 如果新邻居和待淘汰候选邻居是同一个节点，则直接淘汰
                        if (new_nbr->id == wti_cand_nbr->id) wti_cand_nbr->status = Repeated;
                        // 否则，比较 pivot 到待淘汰候选邻居的距离和新邻居到待淘汰候选邻居距离(ip
                        // 取负)
                        else if (compare_dist(
                                     wti_cand_nbr->dist,
                                     calc_distance(base, wti_cand_nbr->id, new_nbr->id, bp), bp))
                            wti_cand_nbr->status = Discarded;  // 淘汰
                    }
                }
            } else if (new_nbr->status == AllRetained)  // 候选邻居数量小于 max_degree，只需要判重
            {
                const int wti_num = cand_size - new_nbr_i - 1;
                for (int i = 0; i < wti_num; i++) {
                    CN *wti_cand_nbr = get_CN(cand_nbrs, sort_idxs, i + new_nbr_i + 1, group_id,
                                              pivot_idx, vec_num, cand_size);
                    if (wti_cand_nbr->id == new_nbr->id) wti_cand_nbr->status = Repeated;  // 淘汰
                }
            }
        }
    }
}

__global__ void get_new_nbrs_kernel(CN *cand_nbrs, const int *sort_idxs, int *new_nbr_ids,
                                    float *new_nbr_dists, int cand_size, BP bp) {
    const int group_id = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    for (int i = tid; i < bp.k; i += tpb) {
        int nbr_num = 0;
        for (int j = 0; j < cand_size && nbr_num < bp.max_degree; j++) {
            const CN *cand_nbr = get_CN(cand_nbrs, sort_idxs, j, group_id, i, bp.k, cand_size);
            if (cand_nbr->status == Retained || cand_nbr->status == AllRetained) {
                const int idx = group_id * bp.k * bp.max_degree + i * bp.max_degree + nbr_num++;
                // assert(cand_nbr.id != -1);
                new_nbr_ids[idx] = cand_nbr->id;
                new_nbr_dists[idx] = cand_nbr->dist;
            }
        }

        for (int j = 0; j < cand_size && nbr_num < bp.max_degree; j++) {
            const CN *cand_nbr = get_CN(cand_nbrs, sort_idxs, j, group_id, i, bp.k, cand_size);
            if (cand_nbr->status == AllRetained) break;

            if (cand_nbr->id >= 0 && cand_nbr->status == Discarded) {
                const int idx = group_id * bp.k * bp.max_degree + i * bp.max_degree + nbr_num++;
                new_nbr_ids[idx] = cand_nbr->id;
                new_nbr_dists[idx] = cand_nbr->dist;
            }
        }
        // assert(nbr_num > 0);
    }
}
#ifdef algo0
__global__ void add_reverse_kernel(CN *cand_nbrs, const int *d_old_nbrs,
                                   const float *d_old_nbr_dists, const int *d_old_nbr_nums,
                                   BP bp) {
    const int batch_id = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    const CN *pivot_nbr_start = cand_nbrs + batch_id * bp.k * bp.cand_size;

    for (int idx = tid + 1; idx < bp.k; idx += tpb) {
        const int old_nbr_num = d_old_nbr_nums[batch_id * bp.k + idx];
        STATUS status = old_nbr_num < bp.max_degree ? AllRetained : Initial;
        for (int i = 0; i < old_nbr_num; i++) {
            const int cand_nbr_idx = batch_id * bp.k * bp.cand_size + idx * bp.cand_size + i;
            const int old_nbr_idx = batch_id * bp.k * bp.max_degree + idx * bp.max_degree + i;
            // assert(d_old_nbrs[old_nbr_idx] >= 0);
            cand_nbrs[cand_nbr_idx].id = d_old_nbrs[old_nbr_idx];
            cand_nbrs[cand_nbr_idx].idx = i;
            cand_nbrs[cand_nbr_idx].dist = d_old_nbr_dists[old_nbr_idx];
            cand_nbrs[cand_nbr_idx].status = status;
        }
        // 将 pivot 加入其他 k-1 个节点的候选邻居，即加反边：
        const int cand_nbr_idx = batch_id * bp.k * bp.cand_size + idx * bp.cand_size + old_nbr_num;
        cand_nbrs[cand_nbr_idx].id = pivot_nbr_start[0].id;
        cand_nbrs[cand_nbr_idx].idx = old_nbr_num;
        cand_nbrs[cand_nbr_idx].dist = pivot_nbr_start[idx].dist;
        cand_nbrs[cand_nbr_idx].status = status;

        for (int i = old_nbr_num + 1; i < bp.cand_size; i++) {
            const int cand_nbr_idx = batch_id * bp.k * bp.cand_size + idx * bp.cand_size + i;
            cand_nbrs[cand_nbr_idx].id = -1;
            cand_nbrs[cand_nbr_idx].idx = i;
            cand_nbrs[cand_nbr_idx].dist =
                bp.metric == DIST_METRIC::L2 ? L2_FARTHEST : IP_FARTHEST;
            cand_nbrs[cand_nbr_idx].status = Discarded;
        }
    }
}

__global__ void pivot_to_others_dist_compute_kernel(const float *__restrict__ base,
                                                    const int *__restrict__ knns, CN *cand_nbrs,
                                                    BP bp) {
    const int group_id = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    const int pivot_idx = knns[group_id * bp.k];  // pivot 向量在 base 中的开始索引
    extern __shared__ float pivot_vec[];
    // 加载 pivot 向量到共享内存：
    for (int i = tid; i < bp.dim; i += tpb) pivot_vec[i] = base[pivot_idx * bp.dim + i];
    __syncthreads();

    // 计算 pivot 向量和其他 k - 1 个向量间的距离：
    for (int i = tid; i < bp.cand_size; i += tpb) {
        const int cand_idx = group_id * bp.k * bp.cand_size + i;
        const int id = i < bp.k ? knns[group_id * bp.k + i] : -1;
        cand_nbrs[cand_idx].dist =
            i == 0 ? (bp.metric == DIST_METRIC::L2 ? L2_CLOSEST : IP_CLOSEST)
                   : (i < bp.k ? (bp.metric == DIST_METRIC::L2
                                      ? l2_distance(pivot_vec, base, id, bp.dim)
                                      : ip_distance(pivot_vec, base, id, bp.dim))
                               : (bp.metric == DIST_METRIC::L2 ? L2_FARTHEST : IP_FARTHEST));
        cand_nbrs[cand_idx].idx = i;
        cand_nbrs[cand_idx].id = id;
        cand_nbrs[cand_idx].status =
            i == 0 ? Repeated : ((i > 0 && i < bp.k) ? Initial : Discarded);
    }
}
#endif

#ifndef GIG
__global__ void knn_dist_compute_kernel(const float *__restrict__ base,
                                        const int *__restrict__ knns, const int *d_old_nbrs,
                                        const float *d_old_nbr_dists, const int *d_old_nbr_nums,
                                        CN *cand_nbrs, int tile_k, BP bp) {
    const int batch_id = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    extern __shared__ unsigned char shared_mem[];
    uintptr_t smem_ptr = reinterpret_cast<uintptr_t>(shared_mem);  // 基地址
    // 存储一个 tile 的基础向量，大小为 tile_k * dim:
    float *tile_row_vecs = reinterpret_cast<float *>(smem_ptr);
    smem_ptr += tile_k * bp.dim * sizeof(float);  // 偏移字节数
    int *old_nbr_nums = reinterpret_cast<int *>(smem_ptr);

    for (int i = tid; i < bp.k; i += tpb) old_nbr_nums[i] = d_old_nbr_nums[batch_id * bp.k + i];
    __syncthreads();

    for (int row_tile = 0; row_tile < bp.k; row_tile += tile_k) {
        for (int i = tid; i < tile_k * bp.dim; i += tpb) {  // 加载 tile_k 个 row 向量进共享内存
            // 线程处理的tile 中第 i / dim 个向量在其所在 kNNs 中的索引：
            const int global_idx = row_tile + i / bp.dim;
            if (global_idx >= bp.k) break;

            tile_row_vecs[i] = base[knns[batch_id * bp.k + global_idx] * bp.dim + i % bp.dim];
        }
        __syncthreads();

        // 计算加载进共享内存的 tile_k 个 row 向量和 K 个列向量的距离
        // 每个线程负责 tile_k * K 距离矩阵中一个点的计算
        for (int i = tid; i < tile_k * bp.cand_size; i += tpb) {
            const int local_row = i / bp.cand_size;
            const int global_row = row_tile + local_row, global_col = i % bp.cand_size;
            if (global_row >= bp.k) break;

            const int matrix_idx =
                batch_id * bp.k * bp.cand_size + global_row * bp.cand_size + global_col;
            if (global_col >= bp.k + old_nbr_nums[global_row]) {
                cand_nbrs[matrix_idx].id = -1;
                cand_nbrs[matrix_idx].status = Discarded;
            } else if (global_col >= bp.k) {
                // assert(old_nbr_nums[global_row] > 0);
                const int old_nbr_idx = batch_id * bp.k * bp.max_degree +
                                        global_row * bp.max_degree + global_col - bp.k;
                cand_nbrs[matrix_idx].id = d_old_nbrs[old_nbr_idx];
                cand_nbrs[matrix_idx].idx = global_col;
                cand_nbrs[matrix_idx].dist = d_old_nbr_dists[old_nbr_idx];
                cand_nbrs[matrix_idx].status = Initial;
            } else if (global_row < global_col) {
                const int sym_matrix_idx =
                    batch_id * bp.k * bp.cand_size + global_col * bp.cand_size + global_row;
#ifdef algo1
                const int col_base_id = knns[batch_id * bp.k + global_col];
                // 候选邻居到 pivot 的距离:
                cand_nbrs[matrix_idx].dist = cand_nbrs[sym_matrix_idx].dist =
                    bp.metric == DIST_METRIC::L2 ? l2_distance(tile_row_vecs + local_row * bp.dim,
                                                               base, col_base_id, bp.dim)
                                                 : ip_distance(tile_row_vecs + local_row * bp.dim,
                                                               base, col_base_id, bp.dim);

                cand_nbrs[matrix_idx].idx = global_col;  // 候选邻居在 knns 中的索引（0~k-1）
                cand_nbrs[matrix_idx].id = col_base_id;  // 候选邻居在 base 中的索引
                cand_nbrs[matrix_idx].status = cand_nbrs[sym_matrix_idx].status = Initial;

                // 矩阵中另一个轴对称的点：
                cand_nbrs[sym_matrix_idx].idx = global_row;
                cand_nbrs[sym_matrix_idx].id = knns[batch_id * bp.k + global_row];
#endif
#ifdef algo2
                if (old_nbr_nums[global_row] > 0 && old_nbr_nums[global_col] > 0) {
                    cand_nbrs[matrix_idx].id = cand_nbrs[sym_matrix_idx].id = -1;
                    cand_nbrs[matrix_idx].status = cand_nbrs[sym_matrix_idx].status = Discarded;
                } else {
                    const int col_base_id = knns[batch_id * bp.k + global_col];
                    const float *row_vec = tile_row_vecs + local_row * bp.dim;
                    // 候选邻居到 pivot 的距离:
                    cand_nbrs[matrix_idx].dist = cand_nbrs[sym_matrix_idx].dist =
                        bp.metric == DIST_METRIC::L2
                            ? l2_distance(row_vec, base, col_base_id, bp.dim)
                            : ip_distance(row_vec, base, col_base_id, bp.dim);
                    cand_nbrs[matrix_idx].status = cand_nbrs[sym_matrix_idx].status = Initial;
                    cand_nbrs[matrix_idx].idx = global_col;  // 候选邻居在 knns 中的索引（0~k-1）
                    cand_nbrs[matrix_idx].id = col_base_id;  // 候选邻居在 base 中的索引
                    // 矩阵中另一个轴对称的点：
                    cand_nbrs[sym_matrix_idx].idx = global_row;
                    cand_nbrs[sym_matrix_idx].id = knns[batch_id * bp.k + global_row];
                }
#endif
            } else if (global_row == global_col)
                cand_nbrs[matrix_idx].status = Repeated;
        }
        __syncthreads();  // 等待所有线程完成当前 tile 行的计算
    }
}
#endif

void idx_dist_extract(CN *nbrs, int *idxs, float *dists, cudaStream_t &stream, int num, BP bp) {
    thrust::device_ptr<CN> nbrs_ptr(nbrs);
    thrust::device_ptr<float> dists_ptr(dists);
    thrust::device_ptr<int> idxs_ptr(idxs);

    // 使用 zip_iterator 组合输出
    auto output_begin = thrust::make_zip_iterator(thrust::make_tuple(dists_ptr, idxs_ptr));

    // 提取结构体数组 d_cand_nbrs 中的 dist 和 idx 字段，对于 IP 距离，取负实现从大到小排序
    if (bp.metric == DIST_METRIC::L2)
        thrust::transform(thrust::cuda::par.on(stream), nbrs_ptr, nbrs_ptr + num, output_begin,
                          [] __host__ __device__(const CN &nbr) {
                              return thrust::make_tuple(nbr.dist, nbr.idx);
                          });
    else
        thrust::transform(thrust::cuda::par.on(stream), nbrs_ptr, nbrs_ptr + num, output_begin,
                          [] __host__ __device__(const CN &nbr) {
                              return thrust::make_tuple(-nbr.dist, nbr.idx);
                          });
}

#ifndef GIG
std::pair<int *, float *> GPUFuncs::handle_knn_updates(const int *d_knn_idxs,
                                                       const int *d_old_nbrs,
                                                       const float *d_old_nbr_dists,
                                                       const int *d_old_nbr_nums,
                                                       const int batch) {
    const std::string name = "upd";
    cudaStream_t &stream = streams[name].stream;

    event_record_time_start(name);

// 1. 计算每组 kNN 中 pivot 节点（第一个节点）和其他 k-1 个节点的距离
#ifdef algo0
    pivot_to_others_dist_compute_kernel<<<batch, upd_threads, bp.dim * sizeof(float), stream>>>(
        d_b, d_knn_idxs, d_cand_nbrs, bp);

    add_reverse_kernel<<<batch, 32, 0, stream>>>(d_cand_nbrs, d_old_nbrs, d_old_nbr_dists,
                                                 d_old_nbr_nums, bp);
#else
    // 共享内存最大向量存储数量:
    const static size_t max_vec_num =
        (shared_mem_per_block - bp.k * sizeof(int)) / (bp.dim * sizeof(float));
    // 每个 block 根据共享内存大小限制分块处理对应的 KNN:
    const static size_t tile_k = std::min<size_t>(bp.k, max_vec_num);
    // 每个 block 的实际使用的共享内存大小:
    const static size_t shared_size = tile_k * bp.dim * sizeof(float) + bp.k * sizeof(int);

    knn_dist_compute_kernel<<<batch, upd_threads, shared_size, stream>>>(
        d_b, d_knn_idxs, d_old_nbrs, d_old_nbr_dists, d_old_nbr_nums, d_cand_nbrs, tile_k, bp);
#endif

    event_record_time_stop(name, "upd_dist_compute");

    // 2.1. 针对每组 KNN 中每个 pivot 节点，
    // 将其候选邻居集（其他 k-1 个节点）和之前的旧邻居 d_old_nbrs 合并得到总的候选邻居集，
    // 对候选向量集按到 pivot 的距离排序
    event_record_time_start(name);

    idx_dist_extract(d_cand_nbrs, d_idxs, d_knn_dists, stream, batch * bp.k * bp.cand_size, bp);

    auto res = segmented_sort_pairs(d_idxs, d_knn_dists, d_idxs_sort, d_knn_dists_sort, d_offsets,
                                    stream, batch * bp.k, bp.cand_size);

    event_record_time_stop(name, "upd_dist_sort");

    // 2.2. 针对 batch 组 cand_size 近邻数据，每次迭代线程并行进行 𝑏𝑎𝑡𝑐ℎ * 𝑘 * (𝑘 − 2 − 𝑖) 次淘汰
    event_record_time_start(name);

    CUDA_CHECK(cudaMemsetAsync(d_nbr_num, 0, batch * bp.k * sizeof(int), stream));

    // 根据每个 pivot 候选邻居到 pivot
    // 的距离，从第一个开始，从近到远成为邻居并并行淘汰后面的后续邻居
    for (int i = 0; i < bp.cand_size; ++i)
        candidate_ignore_kernel<<<batch, 128, 0, stream>>>(d_b, d_cand_nbrs, res.first, d_nbr_num,
                                                           i, batch, bp);

    CUDA_CHECK(
        cudaMemsetAsync(d_new_nbr_ids, -1, BATCH * bp.k * bp.max_degree * sizeof(int), stream));

    get_new_nbrs_kernel<<<batch, 64, 0, stream>>>(d_cand_nbrs, res.first, d_new_nbr_ids,
                                                  d_new_nbr_dists, bp);

    event_record_time_stop(name, "upd_candidate_ignore");

    return std::make_pair(d_new_nbr_ids, d_new_nbr_dists);
}
#endif
}  // namespace efanna2e