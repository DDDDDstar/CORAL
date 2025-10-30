#include <cfloat>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>

#include "fileout.h"
#include "gpufuncs.cuh"
#include "utils.cuh"

namespace efanna2e {
#ifdef GIG
__global__ void pair_dist_compute_kernel(const float *__restrict__ base,
                                         const int *__restrict__ vec_ids, const int *graph,
                                         const float *graph_dist, const int *graph_deg,
                                         CN *cand_nbrs, int tile_k, int vec_num, int batch,
                                         BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x,
              cand_size = vec_num + bp.max_degree;
    extern __shared__ unsigned char smem[];
    // 存储一个 tile 的基础向量，大小为 tile_k * dim:
    float *tile_row_vecs = (float *)(smem);
    int *old_nbr_nums = (int *)(tile_row_vecs + tile_k * bp.dim), *ids = old_nbr_nums + vec_num;
    for (int batch_id = bid; batch_id < batch; batch_id += gridDim.x) {
        for (int i = tid; i < vec_num; i += tpb) {
            ids[i] = vec_ids[batch_id * vec_num + i];
            old_nbr_nums[i] = graph_deg[ids[i]];
        }
        __syncthreads();

        for (int row_tile = 0; row_tile < vec_num; row_tile += tile_k) {
            for (int i = tid; i < tile_k * bp.dim;
                 i += tpb) {  // 加载 tile_k 个 row 向量进共享内存
                // 线程处理的tile 中第 i / dim 个向量在其所在 kNNs 中的索引：
                const int global_idx = row_tile + i / bp.dim;
                if (global_idx >= vec_num) break;
                tile_row_vecs[i] = base[ids[global_idx] * bp.dim + i % bp.dim];
            }
            __syncthreads();

            // 计算加载进共享内存的 tile_k 个 row 向量和 vec_num 个列向量的距离
            // 每个线程负责 tile_k * vec_num 距离矩阵中一个点的计算
            for (int i = tid; i < tile_k * cand_size; i += tpb) {
                const int local_row = i / cand_size;
                const int global_row = row_tile + local_row, global_col = i % cand_size;
                if (global_row >= vec_num) break;
                const int matrix_idx =
                    batch_id * vec_num * cand_size + global_row * cand_size + global_col;
                if (global_col >= vec_num + old_nbr_nums[global_row]) {
                    cand_nbrs[matrix_idx].id = -1;
                    cand_nbrs[matrix_idx].status = Discarded;
                } else if (global_col >= vec_num) {
                    const int old_nbr_idx = ids[global_row] * bp.max_degree + global_col - vec_num;
                    cand_nbrs[matrix_idx].id = graph[old_nbr_idx];
                    cand_nbrs[matrix_idx].idx = global_col;
                    cand_nbrs[matrix_idx].dist = graph_dist[old_nbr_idx];
                    cand_nbrs[matrix_idx].status = Initial;
                } else if (global_row < global_col) {
                    const int sym_matrix_idx =
                        batch_id * vec_num * cand_size + global_col * cand_size + global_row;
                    if (old_nbr_nums[global_row] > 0 && old_nbr_nums[global_col] > 0) {
                        cand_nbrs[matrix_idx].id = cand_nbrs[sym_matrix_idx].id = -1;
                        cand_nbrs[matrix_idx].status = cand_nbrs[sym_matrix_idx].status =
                            Discarded;
                    } else {
                        const int col_base_id = ids[global_col];
                        const float *row_vec = tile_row_vecs + local_row * bp.dim;
                        // 候选邻居到 pivot 的距离:
                        cand_nbrs[matrix_idx].dist = cand_nbrs[sym_matrix_idx].dist =
                            bp.metric == DIST_METRIC::L2
                                ? l2_distance(row_vec, base, col_base_id, bp.dim)
                                : ip_distance(row_vec, base, col_base_id, bp.dim);
                        cand_nbrs[matrix_idx].status = cand_nbrs[sym_matrix_idx].status = Initial;
                        // 候选邻居在集合中的索引（0~vec_num-1）：
                        cand_nbrs[matrix_idx].idx = global_col;
                        cand_nbrs[matrix_idx].id = col_base_id;  // 候选邻居在 base 中的索引
                        // 矩阵中另一个轴对称的点：
                        cand_nbrs[sym_matrix_idx].idx = global_row;
                        cand_nbrs[sym_matrix_idx].id = ids[global_row];
                    }
                } else if (global_row == global_col)
                    cand_nbrs[matrix_idx].status = Repeated;
            }
            __syncthreads();  // 等待所有线程完成当前 tile 行的计算
        }
    }
}

__global__ void update_new_nbrs_kernel(const int *__restrict__ vec_ids,
                                       const int *__restrict__ sort_idxs, int *__restrict__ graph,
                                       int *__restrict__ graph_deg, float *__restrict__ graph_dist,
                                       CN *__restrict__ cand_nbrs, int vec_num, int batch, BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x,
              cand_size = vec_num + bp.max_degree;
    for (int group_id = bid; group_id < batch; group_id += gridDim.x) {
        for (int i = tid; i < vec_num; i += tpb) {
            int nbr_num = 0;
            const int knn_id = vec_ids[group_id * vec_num + i], idx_start = knn_id * bp.max_degree;
            for (int j = 0; j < cand_size && nbr_num < bp.max_degree; j++) {
                const CN *cand_nbr = get_CN(cand_nbrs, sort_idxs, j, group_id, i, bp.k, cand_size);
                if (cand_nbr->status == Retained || cand_nbr->status == AllRetained) {
                    const int idx = idx_start + nbr_num++;
                    graph[idx] = cand_nbr->id;  // assert(cand_nbr.id != -1);
                    graph_dist[idx] = cand_nbr->dist;
                }
            }
            for (int j = 0; j < cand_size && nbr_num < bp.max_degree; j++) {
                const CN *cand_nbr = get_CN(cand_nbrs, sort_idxs, j, group_id, i, bp.k, cand_size);
                if (cand_nbr->status == AllRetained) break;
                if (cand_nbr->id >= 0 && cand_nbr->status == Discarded) {
                    const int idx = idx_start + nbr_num++;
                    graph[idx] = cand_nbr->id;
                    graph_dist[idx] = cand_nbr->dist;
                }
            }
            graph_deg[knn_id] = nbr_num;  // assert(nbr_num > 0);
        }
    }
}

__global__ void update_new_nbrs_kernel(const int *__restrict__ vec_ids,
                                       const int *__restrict__ sort_idxs, int *__restrict__ graph,
                                       int *__restrict__ graph_deg, float *__restrict__ graph_dist,
                                       CN *__restrict__ cand_nbrs,
                                       int *__restrict__ from_edge_nums, int vec_num, int batch,
                                       BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x,
              cand_size = vec_num + bp.max_degree;
    for (int group_id = bid; group_id < batch; group_id += gridDim.x) {
        for (int i = tid; i < vec_num; i += tpb) {
            int nbr_num = 0;
            const int base_id = vec_ids[group_id * vec_num + i],
                      idx_start = base_id * bp.max_degree;

            for (int old_nbr_i = 0; old_nbr_i < graph_deg[base_id]; old_nbr_i++)
                atomicSub(from_edge_nums + graph[idx_start + old_nbr_i], 1);

            for (int j = 0; j < cand_size && nbr_num < bp.max_degree; j++) {
                const CN *cand_nbr = get_CN(cand_nbrs, sort_idxs, j, group_id, i, bp.k, cand_size);
                if (cand_nbr->status == Retained || cand_nbr->status == AllRetained) {
                    const int idx = idx_start + nbr_num++;
                    graph[idx] = cand_nbr->id;  // assert(cand_nbr.id != -1);
                    graph_dist[idx] = cand_nbr->dist;
                    atomicAdd(from_edge_nums + cand_nbr->id, 1);
                }
            }
            for (int j = 0; j < cand_size && nbr_num < bp.max_degree; j++) {
                const CN *cand_nbr = get_CN(cand_nbrs, sort_idxs, j, group_id, i, bp.k, cand_size);
                if (cand_nbr->status == AllRetained) break;
                if (cand_nbr->id >= 0 && cand_nbr->status == Discarded) {
                    const int idx = idx_start + nbr_num++;
                    graph[idx] = cand_nbr->id;
                    graph_dist[idx] = cand_nbr->dist;
                    atomicAdd(from_edge_nums + cand_nbr->id, 1);
                }
            }
            graph_deg[base_id] = nbr_num;  // assert(nbr_num > 0);
        }
    }
}

__global__ void get_graph_structure_kernel(GSD *gs_res, const int *graph_deg, int base_n, BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    if (bid == 0 && tid == 0) {
        gs_res->iso_num = 0;
        gs_res->total_degree = 0;
    }
    __syncthreads();

    for (int id = bid * tpb + tid; id < base_n; id += gridDim.x * tpb) {
        if (graph_deg[id] == 0)
            atomicAdd(&gs_res->iso_num, 1);
        else
            atomicAdd(&gs_res->total_degree, graph_deg[id]);
    }
}

void GPUFuncs::neibor_aware(const int *vec_ids, int batch, int vec_num, int base_n,
                            cudaStream_t &stream, int *from_edge_nums) {
    const int cand_size = bp.max_degree + vec_num, size = batch * vec_num * cand_size;
    thrust::device_vector<Candidate_Neighbor> cand_nbrs(size);
    thrust::device_vector<int> cand_idxs(size), cand_idxs_sort(size), offsets(batch * vec_num + 1),
        nbr_nums(batch * vec_num);
    thrust::device_vector<float> cand_dists(size), cand_dists_sort(size);

    // 1. 计算每组中节点两两间距
    // 共享内存最大向量存储数量:
    const static size_t max_vec_num =
        (shared_mem_per_block - 2 * vec_num * sizeof(int)) / (bp.dim * sizeof(float));
    // 每个 block 根据共享内存大小限制分块处理对应的 KNN:
    const static size_t tile_k = std::min<size_t>(vec_num, max_vec_num);
    // 每个 block 的实际使用的共享内存大小:
    const static size_t shared_size = tile_k * bp.dim * sizeof(float) + 2 * vec_num * sizeof(int);

    pair_dist_compute_kernel<<<batch, 64, shared_size, stream>>>(
        d_b.data().get(), vec_ids, graph.data().get(), graph_dist.data().get(),
        graph_deg.data().get(), cand_nbrs.data().get(), tile_k, vec_num, batch, bp);

    // 2.1. 针对集合中每个 pivot 节点，
    // 将其候选邻居集（其他 vec_num - 1 个节点）和之前的旧邻居 d_old_nbrs 合并得到总的候选邻居集，
    // 对候选向量集按到 pivot 的距离排序
    idx_dist_extract(cand_nbrs.data().get(), cand_idxs.data().get(), cand_dists.data().get(),
                     stream, size, bp);
    auto res = segmented_sort_pairs(cand_idxs.data().get(), cand_dists.data().get(),
                                    cand_idxs_sort.data().get(), cand_dists_sort.data().get(),
                                    offsets.data().get(), stream, batch * vec_num, cand_size);

    // 2.2. 针对 batch 组 cand_size 近邻数据，
    // 每次迭代线程并行进行 𝑏𝑎𝑡𝑐ℎ * vec_num * (vec_num − 2 − 𝑖) 次淘汰
    // 根据每个 pivot 候选邻居到 pivot 的距离，从第一个开始，
    // 从近到远成为邻居并并行淘汰后面的后续邻居
    for (int i = 0; i < cand_size; ++i)
        candidate_ignore_kernel<<<batch, 128, 0, stream>>>(
            d_b.data().get(), cand_nbrs.data().get(), res.first, nbr_nums.data().get(), i, batch,
            cand_size, vec_num, bp);

    {
        std::unique_lock<std::shared_mutex> lock(graph_mtx);

        if (from_edge_nums == nullptr)
            update_new_nbrs_kernel<<<batch, 64, 0, stream>>>(  // 将新邻居更新到 Graph 中
                vec_ids, res.first, graph.data().get(), graph_deg.data().get(),
                graph_dist.data().get(), cand_nbrs.data().get(), bp.k, batch, bp);
        else
            update_new_nbrs_kernel<<<batch, 64, 0, stream>>>(  // 将新邻居更新到 Graph 中
                vec_ids, res.first, graph.data().get(), graph_deg.data().get(),
                graph_dist.data().get(), cand_nbrs.data().get(), from_edge_nums, bp.k, batch, bp);
    }

    get_graph_structure_kernel<<<1, 1024, 0, stream>>>(new_gs, graph_deg.data().get(), base_n, bp);

    CUDA_CHECK(cudaMemcpyAsync(&gs, new_gs, sizeof(GSD), cudaMemcpyDeviceToHost, stream));
}

GSD GPUFuncs::handle_knn_updates(const int *d_knn_idxs, const int batch) {
    const std::string name = "upd";
    cudaStream_t &stream = streams[name].stream;

    event_record_time_start(name);
    neibor_aware(d_knn_idxs, batch, bp.k, bp.base_n, stream);
    /*
        // 1. 计算每组 kNN 中节点两两间距
        // 共享内存最大向量存储数量:
        const static size_t max_vec_num =
            (shared_mem_per_block - 2 * bp.k * sizeof(int)) / (bp.dim * sizeof(float));
        // 每个 block 根据共享内存大小限制分块处理对应的 KNN:
        const static size_t tile_k = std::min<size_t>(bp.k, max_vec_num);
        // 每个 block 的实际使用的共享内存大小:
        const static size_t shared_size = tile_k * bp.dim * sizeof(float) + 2 * bp.k * sizeof(int);
        {
            std::shared_lock<std::shared_mutex> lock(graph_mtx);
            knn_dist_compute_kernel<<<batch, upd_threads, shared_size, stream>>>(
                d_b.data().get(), d_knn_idxs, graph.data().get(), graph_dist.data().get(),
                graph_deg.data().get(), d_cand_nbrs, tile_k, bp);
        }
        event_record_time_stop(name, "upd_dist_compute");

        // 2.1. 针对每组 KNN 中每个 pivot 节点，
        // 将其候选邻居集（其他 k-1 个节点）和之前的旧邻居 d_old_nbrs 合并得到总的候选邻居集，
        // 对候选向量集按到 pivot 的距离排序
        event_record_time_start(name);
        idx_dist_extract(d_cand_nbrs, d_idxs, d_knn_dists, stream, batch * bp.k * bp.cand_size,
       bp); auto res = segmented_sort_pairs(d_idxs, d_knn_dists, d_idxs_sort, d_knn_dists_sort,
       d_offsets, stream, batch * bp.k, bp.cand_size); event_record_time_stop(name,
       "upd_dist_sort");

        // 2.2. 针对 batch 组 cand_size 近邻数据，每次迭代线程并行进行 𝑏𝑎𝑡𝑐ℎ * 𝑘 * (𝑘 − 2 − 𝑖)
       次淘汰 event_record_time_start(name); CUDA_CHECK(cudaMemsetAsync(d_nbr_num, 0, batch * bp.k
       * sizeof(int), stream));
        // 根据每个 pivot 候选邻居到 pivot
        // 的距离，从第一个开始，从近到远成为邻居并并行淘汰后面的后续邻居
        for (int i = 0; i < bp.cand_size; ++i)
            candidate_ignore_kernel<<<batch, 128, 0, stream>>>(d_b.data().get(), d_cand_nbrs,
                                                               res.first, d_nbr_num, i, batch, bp);
        {
            std::unique_lock<std::shared_mutex> lock(graph_mtx);
            update_new_nbrs_kernel<<<batch, 64, 0, stream>>>(  // 将新邻居更新到 Graph 中
                d_knn_idxs, graph.data().get(), graph_deg.data().get(), graph_dist.data().get(),
                d_cand_nbrs, res.first, bp);
            get_graph_structure_kernel<<<1, 1024, 0, stream>>>(new_gs, graph_deg.data().get(), bp);
        }
        CUDA_CHECK(cudaMemcpyAsync(&gs, new_gs, sizeof(GSD), cudaMemcpyDeviceToHost, stream));
    */
    event_record_time_stop(name, "upd_candidate_ignore");
    return gs;
}
#endif

void GPUFuncs::upd_prepare() {  // 图结构数据内存分配：
    cudaStream_t &stream = streams["upd"].stream;
#ifdef GIG
    CUDA_CHECK(cudaMallocAsync(&new_gs, sizeof(GSD), stream));
#else
    CUDA_CHECK(
        cudaMallocAsync(&d_new_nbr_ids, BATCH * bp.k * bp.max_degree * sizeof(int), stream));
    CUDA_CHECK(
        cudaMallocAsync(&d_new_nbr_dists, BATCH * bp.k * bp.max_degree * sizeof(float), stream));
#endif

    // handle_knn_updates 预分配内存
#ifdef algo0
    const int size = BATCH * bp.k * std::max(bp.k, bp.max_degree + 1);
    bp.cand_size = std::max(bp.k, bp.max_degree + 1);
#else
    // const int size = BATCH * bp.k * (bp.max_degree + bp.k);
#endif
    // CUDA_CHECK(cudaMallocAsync(&d_cand_nbrs, size * sizeof(CN), stream));
    // CUDA_CHECK(cudaMallocAsync(&d_knn_dists, size * sizeof(float), stream));
    // CUDA_CHECK(cudaMallocAsync(&d_idxs, size * sizeof(int), stream));
    // CUDA_CHECK(cudaMallocAsync(&d_idxs_sort, size * sizeof(int), stream));
    // CUDA_CHECK(cudaMallocAsync(&d_knn_dists_sort, size * sizeof(float), stream));
    // CUDA_CHECK(cudaMallocAsync(&d_offsets, (BATCH * bp.k + 1) * sizeof(int), stream));
    // CUDA_CHECK(cudaMallocAsync(&d_nbr_num, BATCH * bp.k * sizeof(int), stream));
}

void GPUFuncs::upd_free() {
    cudaStream_t &stream = streams["upd"].stream;
    // CUDA_CHECK(cudaFreeAsync(d_cand_nbrs, stream));
    // CUDA_CHECK(cudaFreeAsync(d_knn_dists, stream));
    // CUDA_CHECK(cudaFreeAsync(d_idxs, stream));
    // CUDA_CHECK(cudaFreeAsync(d_idxs_sort, stream));
    // CUDA_CHECK(cudaFreeAsync(d_knn_dists_sort, stream));
    // CUDA_CHECK(cudaFreeAsync(d_offsets, stream));
    // CUDA_CHECK(cudaFreeAsync(d_nbr_num, stream));

#ifdef GIG
    CUDA_CHECK(cudaFreeAsync(new_gs, stream));
    CUDA_CHECK(cudaFreeAsync(d_center, stream));
    CUDA_CHECK(cudaFreeAsync(d_dists, stream));
    CUDA_CHECK(cudaFreeAsync(beam, stream));
    CUDA_CHECK(cudaFreeAsync(beam_converge, stream));
    CUDA_CHECK(cudaFreeAsync(beam_converge_num, stream));
#else
    CUDA_CHECK(cudaFreeAsync(d_new_nbr_ids, stream));
    CUDA_CHECK(cudaFreeAsync(d_new_nbr_dists, stream));
#endif
}
};  // namespace efanna2e

// __global__ void knn_dist_compute_kernel(const float *__restrict__ base,
//                                         const int *__restrict__ knns, const int *graph,
//                                         const float *graph_dist, const int *graph_deg,
//                                         CN *cand_nbrs, const int tile_k, BP bp) {
//     const int batch_id = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
//     extern __shared__ unsigned char shared_mem[];
//     uintptr_t smem_ptr = reinterpret_cast<uintptr_t>(shared_mem);  // 基地址
//     // 存储一个 tile 的基础向量，大小为 tile_k * dim:
//     float *tile_row_vecs = reinterpret_cast<float *>(smem_ptr);
//     smem_ptr += tile_k * bp.dim * sizeof(float);  // 偏移字节数
//     int *old_nbr_nums = reinterpret_cast<int *>(smem_ptr);
//     smem_ptr += bp.k * sizeof(int);
//     int *knn_ids = reinterpret_cast<int *>(smem_ptr);

//     for (int i = tid; i < bp.k; i += tpb) {
//         knn_ids[i] = knns[batch_id * bp.k + i];
//         old_nbr_nums[i] = graph_deg[knn_ids[i]];
//     }
//     __syncthreads();

//     for (int row_tile = 0; row_tile < bp.k; row_tile += tile_k) {
//         for (int i = tid; i < tile_k * bp.dim; i += tpb) {  // 加载 tile_k 个 row 向量进共享内存
//             // 线程处理的tile 中第 i / dim 个向量在其所在 kNNs 中的索引：
//             const int global_idx = row_tile + i / bp.dim;
//             if (global_idx >= bp.k) break;
//             tile_row_vecs[i] = base[knn_ids[global_idx] * bp.dim + i % bp.dim];
//         }
//         __syncthreads();

//         // 计算加载进共享内存的 tile_k 个 row 向量和 K 个列向量的距离
//         // 每个线程负责 tile_k * K 距离矩阵中一个点的计算
//         for (int i = tid; i < tile_k * bp.cand_size; i += tpb) {
//             const int local_row = i / bp.cand_size;
//             const int global_row = row_tile + local_row, global_col = i % bp.cand_size;
//             if (global_row >= bp.k) break;
//             const int matrix_idx =
//                 batch_id * bp.k * bp.cand_size + global_row * bp.cand_size + global_col;
//             if (global_col >= bp.k + old_nbr_nums[global_row]) {
//                 cand_nbrs[matrix_idx].id = -1;
//                 cand_nbrs[matrix_idx].status = Discarded;
//             } else if (global_col >= bp.k) {  // assert(old_nbr_nums[global_row] > 0);
//                 const int old_nbr_idx = knn_ids[global_row] * bp.max_degree + global_col - bp.k;
//                 cand_nbrs[matrix_idx].id = graph[old_nbr_idx];
//                 cand_nbrs[matrix_idx].idx = global_col;
//                 cand_nbrs[matrix_idx].dist = graph_dist[old_nbr_idx];
//                 cand_nbrs[matrix_idx].status = Initial;
//             } else if (global_row < global_col) {
//                 const int sym_matrix_idx =
//                     batch_id * bp.k * bp.cand_size + global_col * bp.cand_size + global_row;
// #ifdef algo1
//                 const int col_base_id = knn_ids[global_col];
//                 // 候选邻居到 pivot 的距离:
//                 cand_nbrs[matrix_idx].dist = cand_nbrs[sym_matrix_idx].dist =
//                     metric == DIST_METRIC::L2
//                         ? l2_distance(tile_row_vecs + local_row * dim, base, col_base_id,
//                         bp.dim) : ip_distance(tile_row_vecs + local_row * dim, base,
//                         col_base_id, bp.dim);
//                 cand_nbrs[matrix_idx].idx = global_col;  // 候选邻居在 knns 中的索引（0~k-1）
//                 cand_nbrs[matrix_idx].id = col_base_id;  // 候选邻居在 base 中的索引
//                 cand_nbrs[matrix_idx].status = cand_nbrs[sym_matrix_idx].status = Initial;
//                 // 矩阵中另一个轴对称的点：
//                 cand_nbrs[sym_matrix_idx].idx = global_row;
//                 cand_nbrs[sym_matrix_idx].id = knn_ids[global_row];
// #endif
// #ifdef algo2
//                 if (old_nbr_nums[global_row] > 0 && old_nbr_nums[global_col] > 0) {
//                     cand_nbrs[matrix_idx].id = cand_nbrs[sym_matrix_idx].id = -1;
//                     cand_nbrs[matrix_idx].status = cand_nbrs[sym_matrix_idx].status = Discarded;
//                 } else {
//                     const int col_base_id = knn_ids[global_col];
//                     const float *row_vec = tile_row_vecs + local_row * bp.dim;
//                     // 候选邻居到 pivot 的距离:
//                     cand_nbrs[matrix_idx].dist = cand_nbrs[sym_matrix_idx].dist =
//                         bp.metric == DIST_METRIC::L2
//                             ? l2_distance(row_vec, base, col_base_id, bp.dim)
//                             : ip_distance(row_vec, base, col_base_id, bp.dim);
//                     cand_nbrs[matrix_idx].status = cand_nbrs[sym_matrix_idx].status = Initial;
//                     cand_nbrs[matrix_idx].idx = global_col;  // 候选邻居在 knns
//                     中的索引（0~k-1） cand_nbrs[matrix_idx].id = col_base_id;  // 候选邻居在
//                     base 中的索引
//                     // 矩阵中另一个轴对称的点：
//                     cand_nbrs[sym_matrix_idx].idx = global_row;
//                     cand_nbrs[sym_matrix_idx].id = knn_ids[global_row];
//                 }
// #endif
//             } else if (global_row == global_col)
//                 cand_nbrs[matrix_idx].status = Repeated;
//         }
//         __syncthreads();  // 等待所有线程完成当前 tile 行的计算
//     }
// }

// __global__ void update_new_nbrs_kernel(const int *__restrict__ knns, int *graph, int *graph_deg,
//                                        float *graph_dist, CN *cand_nbrs, const int *sort_idxs,
//                                        BP bp) {
//     const int group_id = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
//     for (int i = tid; i < bp.k; i += tpb) {
//         int nbr_num = 0;
//         const int knn_id = knns[group_id * bp.k + i], idx_start = knn_id * bp.max_degree;
//         for (int j = 0; j < bp.cand_size && nbr_num < bp.max_degree; j++) {
//             const CN *cand_nbr = get_CN(cand_nbrs, sort_idxs, j, group_id, i, bp.k,
//             bp.cand_size); if (cand_nbr->status == Retained || cand_nbr->status == AllRetained)
//             {
//                 const int idx = idx_start + nbr_num++;
//                 graph[idx] = cand_nbr->id;  // assert(cand_nbr.id != -1);
//                 graph_dist[idx] = cand_nbr->dist;
//             }
//         }
//         for (int j = 0; j < bp.cand_size && nbr_num < bp.max_degree; j++) {
//             const CN *cand_nbr = get_CN(cand_nbrs, sort_idxs, j, group_id, i, bp.k,
//             bp.cand_size); if (cand_nbr->status == AllRetained) break;

//             if (cand_nbr->id >= 0 && cand_nbr->status == Discarded) {
//                 const int idx = idx_start + nbr_num++;
//                 graph[idx] = cand_nbr->id;
//                 graph_dist[idx] = cand_nbr->dist;
//             }
//         }
//         graph_deg[knn_id] = nbr_num;  // assert(nbr_num > 0);
//     }
// }

// __global__ void get_graph_structure_kernel(GSD *gs_res, const int *graph_deg, BP bp) {
//     const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
//     if (bid == 0 && tid == 0) {
//         gs_res->iso_num = 0;
//         gs_res->total_degree = 0;
//     }
//     __syncthreads();

//     for (int id = bid * tpb + tid; id < bp.base_n; id += gridDim.x * tpb) {
//         if (graph_deg[id] == 0)
//             atomicAdd(&gs_res->iso_num, 1);
//         else
//             atomicAdd(&gs_res->total_degree, graph_deg[id]);
//     }
// }
