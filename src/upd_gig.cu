#include <cfloat>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>

#include "fileout.h"
#include "gpufuncs.cuh"
#include "utils.cuh"

namespace efanna2e {
__global__ void pair_dist_compute_kernel(const int* __restrict__ vec_ids,
                                         const NBD* __restrict__ node_data,
                                         CN* __restrict__ cand_nbrs, int tile_k, int vec_num,
                                         int batch, BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x,
              cand_size = vec_num + bp.max_degree;
    extern __shared__ float tile_row_vecs[];  // 存储一个 tile 的基础向量，大小为 tile_k * dim

    for (int batch_id = bid; batch_id < batch; batch_id += gridDim.x) {
        const int offset = batch_id * vec_num, *ids = vec_ids + offset,
                  *degs = node_data->degs + offset,
                  *nbrs = node_data->nbrs + offset * bp.max_degree;
        const float *vecs = node_data->vecs + offset * bp.dim,
                    *dists = node_data->dists + offset * bp.max_degree,
                    *nbrs_vec = node_data->nbrs_vec + offset * bp.max_degree * bp.dim,
                    farthest = farthest_dist_d(bp);
        for (int row_tile = 0; row_tile < vec_num; row_tile += tile_k) {
            // 加载 tile_k 个 row 向量进共享内存:
            for (int i = tid; i < tile_k * bp.dim; i += tpb) {
                // 线程处理的tile 中第 i / dim 个向量在其所在 kNNs 中的索引
                const int global_idx = row_tile + i / bp.dim;
                if (global_idx >= vec_num) break;
                tile_row_vecs[i] = vecs[global_idx * bp.dim + i % bp.dim];
            }
            __syncthreads();

            // 计算加载进共享内存的 tile_k 个 row 向量和 vec_num 个列向量的距离
            // 每个线程负责 tile_k * vec_num 距离矩阵中一个点的计算
            for (int i = tid; i < tile_k * cand_size; i += tpb) {
                const int local_row = i / cand_size, global_row = row_tile + local_row;
                if (global_row >= vec_num) break;

                const int global_col = i % cand_size,
                          matrix_idx = offset * cand_size + global_row * cand_size + global_col;
                if (global_col >= vec_num + degs[global_row]) {
                    cand_nbrs[matrix_idx].id = -1;
                    cand_nbrs[matrix_idx].status = Discarded;
                    cand_nbrs[matrix_idx].dist = farthest;
                } else if (global_col >= vec_num) {
                    const int nbr_i = global_col - vec_num,
                              old_nbr_idx = global_row * bp.max_degree + nbr_i;
                    const float* nbr_vec =
                        nbrs_vec + global_row * bp.max_degree * bp.dim + nbr_i * bp.dim;
                    cand_nbrs[matrix_idx].id = nbrs[old_nbr_idx];
                    cand_nbrs[matrix_idx].idx = global_col;
                    cand_nbrs[matrix_idx].dist = dists[old_nbr_idx];
                    cand_nbrs[matrix_idx].status = Initial;

                    for (int d = 0; d < bp.dim; d++) cand_nbrs[matrix_idx].vec[d] = nbr_vec[d];
                } else if (global_row < global_col) {
                    const int sym_matrix_idx =
                        offset * cand_size + global_col * cand_size + global_row;
                    if (degs[global_row] > 0 && degs[global_col] > 0) {
                        cand_nbrs[matrix_idx].id = cand_nbrs[sym_matrix_idx].id = -1;
                        cand_nbrs[matrix_idx].status = cand_nbrs[sym_matrix_idx].status =
                            Discarded;
                        cand_nbrs[matrix_idx].dist = cand_nbrs[sym_matrix_idx].dist = farthest;

                    } else if (ids[global_row] == ids[global_col]) {
                        cand_nbrs[matrix_idx].status = cand_nbrs[sym_matrix_idx].status = Repeated;
                        cand_nbrs[matrix_idx].id = cand_nbrs[sym_matrix_idx].id = -1;
                        cand_nbrs[matrix_idx].dist = cand_nbrs[sym_matrix_idx].dist = farthest;
                    } else {
                        const float *row_vec = tile_row_vecs + local_row * bp.dim,
                                    *col_vec = vecs + global_col * bp.dim;
                        cand_nbrs[matrix_idx].dist = cand_nbrs[sym_matrix_idx].dist =
                            calc_distance(row_vec, col_vec, bp);  // 候选邻居到 pivot 的距离
                        cand_nbrs[matrix_idx].status = cand_nbrs[sym_matrix_idx].status = Initial;
                        cand_nbrs[matrix_idx].idx = global_col;
                        cand_nbrs[matrix_idx].id = ids[global_col];
                        // 矩阵中另一个轴对称的点：
                        cand_nbrs[sym_matrix_idx].idx = global_row;
                        cand_nbrs[sym_matrix_idx].id = ids[global_row];
                        for (int d = 0; d < bp.dim; d++) {
                            cand_nbrs[matrix_idx].vec[d] = col_vec[d];
                            cand_nbrs[sym_matrix_idx].vec[d] = row_vec[d];
                        }
                    }
                } else if (global_row == global_col) {
                    cand_nbrs[matrix_idx].status = Repeated;
                    cand_nbrs[matrix_idx].id = -1;
                    cand_nbrs[matrix_idx].dist = farthest;
                }
            }
            __syncthreads();  // 等待所有线程完成当前 tile 行的计算
        }
    }
}

__global__ void update_new_nbrs_kernel(const int* __restrict__ sort_idxs,
                                       NBD* __restrict__ node_data, CN* __restrict__ cand_nbrs,
                                       int pivot_num, int cand_size, BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    for (int pivot_i = bid * tpb + tid; pivot_i < pivot_num; pivot_i += gridDim.x * tpb) {
        int nbr_num = 0;
        const int nbr_idx_start = pivot_i * bp.max_degree;
        for (int i = 0; i < cand_size && nbr_num < bp.max_degree; i++) {
            const CN* cand_nbr = get_CN(cand_nbrs, sort_idxs, i, pivot_i, cand_size);
            if (cand_nbr->status == Retained) {
                const int idx = nbr_idx_start + nbr_num++;
                node_data->nbrs[idx] = cand_nbr->id;
                node_data->dists[idx] = cand_nbr->dist;
            }
        }
        for (int j = 0; j < cand_size && nbr_num < bp.max_degree; j++) {
            const CN* cand_nbr = get_CN(cand_nbrs, sort_idxs, j, pivot_i, cand_size);
            if (cand_nbr->id >= 0 && cand_nbr->status == Discarded) {
                const int idx = nbr_idx_start + nbr_num++;
                node_data->nbrs[idx] = cand_nbr->id;
                node_data->dists[idx] = cand_nbr->dist;
            }
        }
        assert(nbr_num > 0);
        node_data->degs[pivot_i] = nbr_num;
    }
}

// __global__ void candidate_ignore_kernel(CN* cand_nbrs, int* sort_idxs, int* nbr_num,
//                                         const int new_nbr_i, const int pivot_num, int cand_size,
//                                         BP bp) {
//     const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
//     for (int pivot_i = bid * tpb + tid; pivot_i < pivot_num; pivot_i += gridDim.x * tpb) {
//         CN* new_nbr = get_CN(cand_nbrs, sort_idxs, new_nbr_i, pivot_i, cand_size);
//         // 无效节点或邻居已满，直接淘汰
//         if (new_nbr->id == -1 || nbr_num[pivot_i] >= bp.max_degree)
//             new_nbr->status = Discarded;
//         else if (new_nbr->status == Initial)  // 待定节点，保留且用于淘汰后续节点
//         {
//             new_nbr->status = Retained;
//             nbr_num[pivot_i]++;
//             const int wti_num = cand_size - new_nbr_i - 1;
//             for (int i = 0; i < wti_num; i++) {
//                 CN* wti_cand_nbr = get_CN(cand_nbrs, sort_idxs, i + new_nbr_i + 1, pivot_i,
//                                           cand_size);  // 待淘汰候选邻居
//                 if (wti_cand_nbr->status == Initial) {
//                     // 如果新邻居和待淘汰候选邻居是同一个节点，则直接淘汰
//                     if (new_nbr->id == wti_cand_nbr->id) wti_cand_nbr->status = Repeated;
//                     // 否则，比较 pivot 到待淘汰候选邻居的距离和新邻居到待淘汰候选邻居距离(ip
//                     // 取负)
//                     else if (compare_dist(wti_cand_nbr->dist,
//                                           calc_distance(wti_cand_nbr->vec, new_nbr->vec, bp),
//                                           bp))
//                         wti_cand_nbr->status = Discarded;  // 淘汰
//                 }
//             }
//         }
//     }
// }

__global__ void candidate_ignore_kernel(CN* cand_nbrs, int* sort_idxs, int* nbr_num,
                                        const int new_nbr_i, const int pivot_num, int cand_size,
                                        BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    extern __shared__ int s_flag[];
    int* s_new_nbr_id = s_flag + 1;
    float** s_nbr_vec = (float**)(s_new_nbr_id + 1);
    for (int pivot_i = bid; pivot_i < pivot_num; pivot_i += gridDim.x) {
        if (tid == 0) {
            *s_flag = 0;
            CN* new_nbr = get_CN(cand_nbrs, sort_idxs, new_nbr_i, pivot_i, cand_size);
            // 无效节点或邻居已满，直接淘汰
            if (new_nbr->id > 0 && new_nbr->status == Initial &&
                nbr_num[pivot_i] < bp.max_degree) {
                *s_flag = 1;
                new_nbr->status = Retained;
                nbr_num[pivot_i]++;
                *s_new_nbr_id = new_nbr->id;
                *s_nbr_vec = new_nbr->vec;
            }
        }
        __syncthreads();

        if (*s_flag) {
            for (int i = tid; i < cand_size - new_nbr_i - 1; i += tpb) {
                CN* wti_cand_nbr = get_CN(cand_nbrs, sort_idxs, i + new_nbr_i + 1, pivot_i,
                                          cand_size);  // 待淘汰候选邻居
                if (wti_cand_nbr->status == Initial) {
                    // 如果新邻居和待淘汰候选邻居是同一个节点，则设置为重复
                    if (*s_new_nbr_id == wti_cand_nbr->id) wti_cand_nbr->status = Repeated;
                    // 否则，比较 pivot 到待淘汰候选邻居的距离和新邻居到待淘汰候选邻居距离
                    else if (compare_dist(wti_cand_nbr->dist,
                                          calc_distance(wti_cand_nbr->vec, *s_nbr_vec, bp), bp))
                        wti_cand_nbr->status = Discarded;  // 淘汰
                }
            }
        }
    }
}

void GPUFuncs::neibor_aware(const int* d_vec_ids, NBD* d_node_data, int batch, int vec_num,
                            cudaStream_t& stream) {
    const int cand_size = bp.max_degree + vec_num, size = batch * vec_num * cand_size;

    // 计算每组中节点两两间距，生成候选集（包括旧邻居）
    pair_dist_compute_kernel<<<batch, 32, tile_k * bp.dim * sizeof(float), stream>>>(
        d_vec_ids, d_node_data, cand_nbrs.data().get(), tile_k, vec_num, batch, bp);

    idx_dist_extract(cand_nbrs.data().get(), cand_idxs.data().get(), cand_dists.data().get(),
                     stream, size, bp);

    auto res = segmented_sort_pairs(cand_idxs.data().get(), cand_dists.data().get(),
                                    cand_idxs_sort.data().get(), cand_dists_sort.data().get(),
                                    offsets.data().get(), stream, batch * vec_num, cand_size);

    // 2.2. 针对 batch 组 cand_size 近邻数据，
    // 每次迭代线程并行进行 𝑏𝑎𝑡𝑐ℎ * vec_num * (vec_num − 2 − 𝑖) 次淘汰
    // 根据每个 pivot 候选邻居到 pivot 的距离，从第一个开始，
    // 从近到远成为邻居并并行淘汰后面的后续邻居
    thrust::fill(thrust::cuda::par.on(stream), nbr_nums.begin(), nbr_nums.end(), 0);
    for (int i = 0; i < cand_size; ++i)
        candidate_ignore_kernel<<<1024, 64, sizeof(int) * 2 + sizeof(float*), stream>>>(
            cand_nbrs.data().get(), res.first, nbr_nums.data().get(), i, batch * vec_num,
            cand_size, bp);

    const int threads = 256, blocks = (batch * vec_num + threads - 1) / threads;
    update_new_nbrs_kernel<<<blocks, threads, 0, stream>>>(res.first, d_node_data,
                                                           cand_nbrs.data().get(), vec_num * batch,
                                                           cand_size, bp);  // 更新新邻居
}

float GPUFuncs::handle_knn_updates(const int* d_knn_ids, int batch_id, int batch) {
    const std::string name = "upd";
    cudaStream_t& stream = streams[name].stream;

    auto e = gpu_record_time_start(stream);

    std::vector<int> h_knn_ids(bp.k * batch), cpu_slot_idxs, gpu_slot_idxs;
    CUDA_CHECK(cudaMemcpyAsync(h_knn_ids.data(), d_knn_ids, bp.k * batch * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));

    graph->load_data(h_knn_ids, cpu_slot_idxs, gpu_slot_idxs, d_node_data_buffer, bp.k * batch,
                     stream);

    float load_time;
    e = gpu_record_time_reset(e, stream, load_time);

    neibor_aware(d_knn_ids, d_node_data_buffer, batch, bp.k, stream);

    graph->update_graph_data(gpu_slot_idxs, d_node_data_buffer, bp.k * batch, stream);

    float upd_time = gpu_record_time_stop(e, stream);
#ifdef INFO_PRINT
    fo.print("handle_knn_updates(" + TOS(load_time) + " s, " + TOS(upd_time) + "): batch_id-" +
             TOS(batch_id));
#endif
    return load_time + upd_time;
}

void GPUFuncs::upd_prepare(bool construct) {  // 图结构数据内存分配：
    cudaStream_t& stream = streams["upd"].stream;
    // CUDA_CHECK(cudaMallocAsync(&new_gs, sizeof(GSD), stream));
    // CUDA_CHECK(cudaMallocAsync(&new_vec_ids, BATCH * bp.k * sizeof(int), stream));
    if (construct) {
        h_node_data = new NBD(BATCH * bp.k, bp.max_degree, bp.dim, stream);
        CUDA_CHECK(cudaMallocAsync(&d_node_data_buffer, sizeof(NBD), stream));
        CUDA_CHECK(cudaMemcpyAsync(d_node_data_buffer, h_node_data, sizeof(NBD),
                                   cudaMemcpyHostToDevice, stream));
    }
    // CUDA_CHECK(cudaMallocAsync(&vec_nbrs, BATCH * bp.k * bp.max_degree * sizeof(int), stream));
    // CUDA_CHECK(cudaMallocAsync(&vec_degs, BATCH * bp.k * sizeof(int), stream));
    // CUDA_CHECK(
    //     cudaMallocAsync(&vec_nbr_dists, BATCH * bp.k * bp.max_degree * sizeof(float), stream));
    // CUDA_CHECK(cudaMallocAsync(&finded_idxs, BATCH * bp.k * sizeof(int), stream));
    // CUDA_CHECK(cudaMallocHost(&h_finded_idxs, BATCH * bp.k * sizeof(int)));
    // CUDA_CHECK(cudaMallocHost(&h_knn_ids, BATCH * bp.k * sizeof(int)));

    const int vec_num = construct ? bp.k : bp.k + 1, cand_size = bp.max_degree + vec_num,
              size = BATCH * vec_num * cand_size;
    cand_nbrs.resize(size);
    cand_idxs.resize(size);
    cand_idxs_sort.resize(size);
    offsets.resize(BATCH * vec_num + 1);
    nbr_nums.resize(BATCH * vec_num);
    cand_dists.resize(size);
    cand_dists_sort.resize(size);

    // 共享内存最大向量存储数量:
    int max_vec_num = shared_mem_per_block / (bp.dim * sizeof(float));
    // 每个 block 根据共享内存大小限制分块处理对应的 KNN:
    tile_k = std::min(vec_num, max_vec_num);
}

void GPUFuncs::upd_free() {
    cudaStream_t& stream = streams["upd"].stream;
    // CUDA_CHECK(cudaFreeAsync(new_gs, stream));
    // CUDA_CHECK(cudaFreeAsync(new_vec_ids, stream));
    // CUDA_CHECK(cudaFreeAsync(vec_nbrs, stream));
    // CUDA_CHECK(cudaFreeAsync(vec_degs, stream));
    // CUDA_CHECK(cudaFreeAsync(vec_nbr_dists, stream));
    // CUDA_CHECK(cudaFreeAsync(d_node_data_buffer, stream));
    // CUDA_CHECK(cudaFreeAsync(finded_idxs, stream));
    // CUDA_CHECK(cudaFreeHost(h_finded_idxs));
    // CUDA_CHECK(cudaFreeHost(h_knn_ids));
    if (h_node_data) {
        CUDA_CHECK(cudaFreeAsync(d_node_data_buffer, stream));
        delete h_node_data;
    }
}
};  // namespace efanna2e

// __global__ void graph_vec_find_kernel(
//     const int* __restrict__ graph_ids, const int* __restrict__ graph,
//     const int* __restrict__ graph_deg, const float* __restrict__ graph_dist,
//     const int* __restrict__ target_ids, int* __restrict__ idxs_out, int* __restrict__ vec_nbrs,
//     int* __restrict__ vec_degs, float* __restrict__ vec_nbr_dists, int target_size, int
//     graph_size, BP bp) { const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x; extern
//     __shared__ int s_target_id[]; int* s_idx_out = s_target_id + 1; for (int i = bid; i <
//     target_size; i += gridDim.x) {
//         if (tid == 0) {
//             *s_idx_out = -1;
//             *s_target_id = target_ids[i];
//         }
//         __syncthreads();

//         for (int j = tid; j < graph_size && *s_idx_out == -1; j += tpb) {
//             if (graph_ids[j] == *s_target_id) {
//                 *s_idx_out = j;
//                 break;
//             }
//         }
//         __syncthreads();

//         if (tid == 0) idxs_out[i] = *s_idx_out;
//         if (*s_idx_out != -1) {
//             const int idx = *s_idx_out;
//             vec_degs[i] = graph_deg[idx];
//             for (int j = tid; j < graph_deg[idx]; j += tpb) {
//                 vec_nbrs[i * bp.max_degree + j] = graph[idx * bp.max_degree + j];
//                 vec_nbr_dists[i * bp.max_degree + j] = graph_dist[idx * bp.max_degree + j];
//             }
//         }
//         __syncthreads();
//     }
// }

// int GPUFuncs::LoadNeededData(const int* d_knn_ids, std::vector<int>& h_graph,
//                              std::vector<int>& h_graph_deg, std::vector<float>& h_graph_dist,
//                              int batch) {
//     const std::string name = "upd";
//     cudaStream_t& stream = streams[name].stream;

//     CUDA_CHECK(cudaMemsetAsync(finded_idxs, -1, batch * bp.k * sizeof(int), stream));
//     graph_vec_find_kernel<<<batch, 512, 2 * sizeof(int), stream>>>(
//         graph_ids.data().get(), graph.data().get(), graph_deg.data().get(),
//         graph_dist.data().get(), d_knn_ids, finded_idxs, vec_nbrs, vec_degs, vec_nbr_dists,
//         batch * bp.k, graph_size, bp);

//     CUDA_CHECK(cudaMemcpyAsync(h_finded_idxs, finded_idxs, batch * bp.k * sizeof(int),
//                                cudaMemcpyDeviceToHost, stream));
//     CUDA_CHECK(cudaMemcpyAsync(h_knn_ids, d_knn_ids, batch * bp.k * sizeof(int),
//                                cudaMemcpyDeviceToHost, stream));

//     std::vector<int> h_new_vec_ids;
//     for (int i = 0; i < batch * bp.k; ++i) {
//         if (h_finded_idxs[i] > 0) continue;
//         int id = h_knn_ids[i];
//         h_new_vec_ids.push_back(id);
//         CUDA_CHECK(cudaMemcpyAsync(knn_vecs + i * bp.dim, h_base + id * bp.dim,
//                                    bp.dim * sizeof(float), cudaMemcpyHostToDevice, stream));
//         CUDA_CHECK(cudaMemcpyAsync(vec_nbrs + i * bp.max_degree,
//                                    h_graph.data() + id * bp.max_degree,
//                                    bp.max_degree * sizeof(int), cudaMemcpyHostToDevice,
//                                    stream));
//         CUDA_CHECK(cudaMemcpyAsync(vec_nbr_dists + i * bp.max_degree,
//                                    h_graph_dist.data() + id * bp.max_degree,
//                                    bp.max_degree * sizeof(float), cudaMemcpyHostToDevice,
//                                    stream));
//         CUDA_CHECK(cudaMemcpyAsync(vec_degs + i, h_graph_deg.data() + id, sizeof(int),
//                                    cudaMemcpyHostToDevice, stream));
//     }
//     CUDA_CHECK(cudaMemcpyAsync(new_vec_ids, h_new_vec_ids.data(),
//                                h_new_vec_ids.size() * sizeof(int), cudaMemcpyHostToDevice,
//                                stream));
//     CUDA_CHECK(cudaStreamSynchronize(stream));
//     return h_new_vec_ids.size();
// }

// int GPUFuncs::SaveGraphData(const int* vec_ids, int batch) {
//     const std::string name = "upd";
//     cudaStream_t& stream = streams[name].stream;

//     CUDA_CHECK(cudaMemsetAsync(finded_idxs, -1, batch * bp.k * sizeof(int), stream));
//     graph_vec_find_kernel<<<batch, 512, 2 * sizeof(int), stream>>>(
//         graph_ids.data().get(), graph.data().get(), graph_deg.data().get(),
//         graph_dist.data().get(), d_knn_ids, finded_idxs, vec_nbrs, vec_degs, vec_nbr_dists,
//         batch * bp.k, graph_size, bp);

//     CUDA_CHECK(cudaMemcpyAsync(h_finded_idxs, finded_idxs, batch * bp.k * sizeof(int),
//                                cudaMemcpyDeviceToHost, stream));
//     CUDA_CHECK(cudaMemcpyAsync(h_knn_ids, d_knn_ids, batch * bp.k * sizeof(int),
//                                cudaMemcpyDeviceToHost, stream));

//     std::vector<int> h_new_vec_ids;
//     for (int i = 0; i < batch * bp.k; ++i) {
//         if (h_finded_idxs[i] > 0) continue;
//         int id = h_knn_ids[i];
//         h_new_vec_ids.push_back(id);
//         CUDA_CHECK(cudaMemcpyAsync(knn_vecs + i * bp.dim, h_base + id * bp.dim,
//                                    bp.dim * sizeof(float), cudaMemcpyHostToDevice, stream));
//         CUDA_CHECK(cudaMemcpyAsync(vec_nbrs + i * bp.max_degree,
//                                    h_graph.data() + id * bp.max_degree,
//                                    bp.max_degree * sizeof(int), cudaMemcpyHostToDevice,
//                                    stream));
//         CUDA_CHECK(cudaMemcpyAsync(vec_nbr_dists + i * bp.max_degree,
//                                    h_graph_dist.data() + id * bp.max_degree,
//                                    bp.max_degree * sizeof(float), cudaMemcpyHostToDevice,
//                                    stream));
//         CUDA_CHECK(cudaMemcpyAsync(vec_degs + i, h_graph_deg.data() + id, sizeof(int),
//                                    cudaMemcpyHostToDevice, stream));
//     }
//     CUDA_CHECK(cudaMemcpyAsync(new_vec_ids, h_new_vec_ids.data(),
//                                h_new_vec_ids.size() * sizeof(int), cudaMemcpyHostToDevice,
//                                stream));
//     CUDA_CHECK(cudaStreamSynchronize(stream));
//     return h_new_vec_ids.size();
// }

// __global__ void update_new_nbrs_kernel(const int* __restrict__ sort_idxs,
//                                        NBD* __restrict__ node_data, CN* __restrict__ cand_nbrs,
//                                        int vec_num, int batch, BP bp) {
//     const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x,
//               cand_size = vec_num + bp.max_degree;
//     for (int group_id = bid; group_id < batch; group_id += gridDim.x) {
//         for (int i = tid; i < vec_num; i += tpb) {
//             int nbr_num = 0;
//             // const int base_id = vec_ids[group_id * vec_num + i],
//             const int idx_start = i * bp.max_degree;

//             for (int j = 0; j < cand_size && nbr_num < bp.max_degree; j++) {
//                 const CN* cand_nbr =
//                     get_CN(cand_nbrs, sort_idxs, j, group_id, i, vec_num, cand_size);
//                 if (cand_nbr->status == Retained || cand_nbr->status == AllRetained) {
//                     const int idx = idx_start + nbr_num++;
//                     node_data->nbrs[idx] = cand_nbr->id;  // assert(cand_nbr.id != -1);
//                     node_data->dists[idx] = cand_nbr->dist;
//                 }
//             }
//             for (int j = 0; j < cand_size && nbr_num < bp.max_degree; j++) {
//                 const CN* cand_nbr =
//                     get_CN(cand_nbrs, sort_idxs, j, group_id, i, vec_num, cand_size);
//                 if (cand_nbr->status == AllRetained) break;
//                 if (cand_nbr->id >= 0 && cand_nbr->status == Discarded) {
//                     const int idx = idx_start + nbr_num++;
//                     node_data->nbrs[idx] = cand_nbr->id;
//                     node_data->dists[idx] = cand_nbr->dist;
//                 }
//             }
//             node_data->degs[i] = nbr_num;  // assert(nbr_num > 0);
//         }
//     }
// }

// __global__ void get_graph_structure_kernel(GSD* gs_res, const int* graph_deg, int base_n, BP bp)
// {
//     const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
//     if (bid == 0 && tid == 0) {
//         gs_res->iso_num = 0;
//         gs_res->total_degree = 0;
//     }
//     __syncthreads();

//     for (int id = bid * tpb + tid; id < base_n; id += gridDim.x * tpb) {
//         if (graph_deg[id] == 0)
//             atomicAdd(&gs_res->iso_num, 1);
//         else
//             atomicAdd(&gs_res->total_degree, graph_deg[id]);
//     }
// }
