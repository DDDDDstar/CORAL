#include <cfloat>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>

#include "fileout.h"
#include "gpufuncs.cuh"
#include "utils.cuh"

namespace efanna2e {
__global__ void pair_dist_compute_kernel(const NAPData data, const DeviceView view, int tile_k,
                                         BP bp) {
    return;
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x,
              cand_size = bp.k + bp.max_degree;
    extern __shared__ float tile_row_vecs[];  // 存储一个 tile 的基础向量 [tile_k * dim]
    int* sh_slot_idxs = (int*)(tile_row_vecs + tile_k * bp.dim);  // [k]
    int* sh_vec_idxs = sh_slot_idxs + bp.k;                       // [k]
    CN* cand_nbrs = data.cand_nbrs;
    return;
    for (int batch_id = bid; batch_id < data.batch; batch_id += gridDim.x) {
        const int offset = batch_id * bp.k, *ids = data.d_knn_ids + offset,
                  *nbr_vec_idxs = data.d_nbr_vec_cache_idxs + offset * bp.max_degree;
        return;
        const float farthest = farthest_dist_d(bp);
        for (int i = tid; i < bp.k; i += tpb) {
            int idx = sh_slot_idxs[i] = data.d_cache_idxs[offset + i];
            assert(idx >= 0 && idx < GPU_N);
            idx = sh_vec_idxs[i] = data.d_vec_cache_idxs[offset + i];
            assert(idx >= 0 && idx < GPU_N);
        }
        __syncthreads();
        return;
        for (int row_tile = 0; row_tile < bp.k; row_tile += tile_k) {
            // 加载 tile_k 个 row 向量进共享内存:
            for (int i = tid; i < tile_k * bp.dim; i += tpb) {
                // 线程处理的tile 中第 i / dim 个向量在其所在 kNNs 中的索引
                const int global_idx = row_tile + i / bp.dim;
                if (global_idx >= bp.k) break;
                tile_row_vecs[i] = view.d_vecs[sh_vec_idxs[global_idx] * bp.dim + i % bp.dim];
            }
            __syncthreads();
            return;

            // 计算加载进共享内存的 tile_k 个 row 向量和 bp.k 个列向量的距离
            // 每个线程负责 tile_k * bp.k 距离矩阵中一个点的计算
            for (int i = tid; i < tile_k * cand_size; i += tpb) {
                const int local_row = i / cand_size, global_row = row_tile + local_row;
                if (global_row >= bp.k) break;

                const int row_id = ids[global_row], global_col = i % cand_size,
                          matrix_idx = offset * cand_size + global_row * cand_size + global_col;
                assert(row_id >= 0);
                const Slot& row_slot = view.d_slots[sh_slot_idxs[global_row]];
                const int row_deg = row_slot.deg;
                if (global_col >= bp.k + row_deg || global_row == global_col)
                    cand_nbrs[matrix_idx] = {-1, global_col, Invalid, farthest};
                else if (global_col >= bp.k) {
                    const int nbr_i = global_col - bp.k, nbr_id = row_slot.nbrs[nbr_i];
                    cand_nbrs[matrix_idx] = {row_slot.nbrs[nbr_i],
                                             nbr_vec_idxs[global_row * bp.max_degree + nbr_i],
                                             Initial, row_slot.dists[nbr_i]};
                } else if (global_row < global_col) {
                    const int col_id = ids[global_col];
                    const Slot& col_slot = view.d_slots[sh_slot_idxs[global_col]];
                    const int sym_matrix_idx =
                        offset * cand_size + global_col * cand_size + global_row;
                    assert(row_id != col_id);
                    if (row_deg > 0 && col_slot.deg > 0) {
                        cand_nbrs[matrix_idx].status = cand_nbrs[sym_matrix_idx].status = Invalid;
                        cand_nbrs[matrix_idx].dist = cand_nbrs[sym_matrix_idx].dist = farthest;
                    } else {
                        const int col_vec_idx = sh_vec_idxs[global_col];
                        const float *row_vec = tile_row_vecs + local_row * bp.dim,
                                    *col_vec = view.d_vecs + col_vec_idx * bp.dim;
                        cand_nbrs[matrix_idx].id = col_id;
                        cand_nbrs[matrix_idx].idx = col_vec_idx;
                        cand_nbrs[sym_matrix_idx].id = row_id;
                        cand_nbrs[sym_matrix_idx].idx = sh_vec_idxs[global_row];
                        cand_nbrs[matrix_idx].dist = cand_nbrs[sym_matrix_idx].dist =
                            calc_distance(row_vec, col_vec, bp);
                        cand_nbrs[matrix_idx].status = cand_nbrs[sym_matrix_idx].status = Initial;
                    }
                }
            }
            __syncthreads();  // 等待所有线程完成当前 tile 行的计算
        }
    }
}

__device__ __forceinline__ int atomicTestAndSetBit(uint32_t* bitset, int bit_index) {
    const uint32_t mask = 1u << (bit_index & 31);
    uint32_t* addr = bitset + (bit_index >> 5);
    const uint32_t old = atomicOr(addr, mask);
    return (old & mask) != 0;
}
__global__ void update_new_nbrs_kernel(const NAPData data, const DeviceView view,
                                       CN* __restrict__ cand_nbrs, ull* __restrict__ total_degree,
                                       ull* __restrict__ noniso_num, int pivot_num, int cand_size,
                                       BP bp) {
    const int tid = threadIdx.x, pivot_i = blockIdx.x * blockDim.x + tid;
    __shared__ int delta_degree, delta_noniso_num;
    if (!tid) delta_degree = delta_noniso_num = 0;
    __syncthreads();
    if (pivot_i < pivot_num) {
        const int pivot_idx = data.d_cache_idxs[pivot_i];
        assert(pivot_idx >= 0);
        if (!atomicTestAndSetBit(data.updated, pivot_idx)) {
            int nbr_num = 0, last_id = -1;
            Slot& slot = view.d_slots[pivot_idx];
            const CN* cands = cand_nbrs + pivot_i * cand_size;
            for (int i = 0; i < cand_size && nbr_num < bp.max_degree; i++) {
                const CN& cand_nbr = cands[i];
                if (cand_nbr.status == Retained) {
                    const int cand_nbr_id = cand_nbr.id;
                    if (cand_nbr_id < 0) {
                        printf("update_new_nbrs ERROR: cand_nbr.id < 0 pivot: %d\n", pivot_idx);
                        continue;
                    }
                    if (cand_nbr_id == last_id) {
                        printf("update_new_nbrs ERROR: cand_nbr_id == last_id pivot: %d\n",
                               pivot_idx);
                        continue;
                    }
                    const int idx = nbr_num++;
                    slot.nbrs[idx] = last_id = cand_nbr_id;
                    slot.dists[idx] = cand_nbr.dist;
                } else if (cand_nbr.status == Invalid)
                    break;
            }
            for (int j = 0; j < cand_size && nbr_num < bp.max_degree; j++) {
                const CN& cand_nbr = cands[j];
                const int cand_nbr_id = cand_nbr.id;
                if (cand_nbr.status == Discarded) {
                    if (cand_nbr_id < 0) {
                        printf("update_new_nbrs ERROR: cand_nbr.id < 0 pivot: %d\n", pivot_idx);
                        continue;
                    }
                    if (cand_nbr_id == last_id) {
                        printf("update_new_nbrs ERROR: cand_nbr_id == last_id pivot: %d\n",
                               pivot_idx);
                        continue;
                    }
                    const int idx = nbr_num++;
                    last_id = slot.nbrs[idx] = cand_nbr_id;
                    slot.dists[idx] = cand_nbr.dist;
                } else if (cand_nbr.status == Invalid)
                    break;
            }
            if (nbr_num == 0)
                printf("update_new_nbrs ERROR: nbr_num == 0, pivot: %d\n", slot.node_id);
            view.d_dirty[pivot_idx] = 1;
            const uint32_t pin_id = (uint32_t)data.batch_id;
            atomicCAS(view.d_pin + pivot_idx, pin_id, 0);
            atomicCAS(view.d_vec_pin + data.d_vec_cache_idxs[pivot_i], pin_id, 0);
            const int* nbr_vec_idxs = data.d_nbr_vec_cache_idxs + pivot_i * bp.max_degree;
#pragma unroll 1
            for (int i = 0; i < slot.deg; i++)
                atomicCAS(view.d_vec_pin + nbr_vec_idxs[i], pin_id, 0);

            const int old_deg = slot.deg;
            if (!old_deg) {
                atomicAdd(&delta_noniso_num, 1);
                atomicAdd(&delta_degree, nbr_num);
            } else
                atomicAdd(&delta_degree, nbr_num - old_deg);
            slot.deg = nbr_num;
        }
    }
    __syncthreads();
    if (!tid) {
        atomicAdd(total_degree, (ull)delta_degree);
        atomicAdd(noniso_num, (ull)delta_noniso_num);
    }
}

struct ExtractDistL2 {
    __host__ __device__ float operator()(const CN& nbr) const { return nbr.dist; }
};

struct ExtractDistIP {
    __host__ __device__ float operator()(const CN& nbr) const { return -nbr.dist; }
};

void dist_extract_from_CN(thrust::device_vector<CN>& nbrs, thrust::device_vector<float>& dists,
                          cudaStream_t stream, BP bp) {
    const size_t num = nbrs.size();
    assert(dists.size() == num);
    CUDA_CHECK(cudaStreamQuery(stream));
    CUDA_CHECK(cudaGetLastError());
    if (bp.metric == DIST_METRIC::L2_)
        thrust::transform(thrust::cuda::par.on(stream), nbrs.begin(), nbrs.end(), dists.begin(),
                          ExtractDistL2{});
    else
        thrust::transform(thrust::cuda::par.on(stream), nbrs.begin(), nbrs.end(), dists.begin(),
                          ExtractDistIP{});
}

__global__ void extract_dist_l2_kernel(const CN* nbrs, float* dists, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dists[i] = nbrs[i].dist;
}

void GPUFuncs::cpu_neibor_aware(NAPData* data, const DeviceView view) {
    const int cand_size = bp.max_degree + bp.k, pivot_num = data->batch * bp.k,
              size = pivot_num * cand_size, tpb = 512, blocks = (pivot_num + tpb - 1) / tpb;

    // cudaError_t q = cudaStreamQuery(data->stream);
    // if (q != cudaSuccess && q != cudaErrorNotReady)
    //     fo.eprint("Stream query failed: " + std::string(cudaGetErrorString(q)));
    // CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaStreamSynchronize(data->stream));
    // 计算每组中节点两两间距，生成候选集（包括旧邻居）
    pair_dist_compute_kernel<<<data->batch, tpb,
                               tile_k * bp.dim * sizeof(float) + bp.k * sizeof(int) * 2,
                               data->stream>>>(*data, view, tile_k, bp);
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaStreamSynchronize(data->stream));
    // exit(0);

    // dist_extract_from_CN(data->cand_nbrs, data->cand_dists, data->stream, bp);
    extract_dist_l2_kernel<<<(size + tpb - 1) / tpb, tpb, 0, data->stream>>>(
        data->cand_nbrs, data->cand_dists, size);
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaStreamSynchronize(data->stream));

    auto res = segmented_sort_pairs_nocpy(data->cand_nbrs, data->cand_dists, data->cand_nbrs_sort,
                                          data->cand_dists_sort, data->offsets, data->stream,
                                          pivot_num, cand_size);

    // 2.2. 针对 batch 组 cand_size 近邻数据，
    // 每次迭代线程并行进行 𝑏𝑎𝑡𝑐ℎ * bp.k * (bp.k − 2 − 𝑖) 次淘汰
    // 根据每个 pivot 候选邻居到 pivot 的距离，从第一个开始，
    // 从近到远成为邻居并并行淘汰后面的后续邻居
    candidate_ignore_kernel<<<blocks, tpb, 0, data->stream>>>(view.d_vecs, res.first, pivot_num,
                                                              cand_size, bp);
    CUDA_CHECK(cudaGetLastError());

    thrust::fill(thrust::cuda::par.on(data->stream), data->updated,
                 data->updated + (GPU_N + 31) / 32, 0);
    update_new_nbrs_kernel<<<blocks, tpb, 0, data->stream>>>(
        *data, view, res.first, d_total_degree, d_noniso_num, pivot_num, cand_size,
        bp);  // 更新新邻居
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaStreamSynchronize(data->stream));
}

void GPUFuncs::NAP_loop() {
    CUDA_CHECK(cudaSetDevice(0));
    auto gpu_cache = graph->GPUCache();
    const auto view = gpu_cache->device_view();
    while (true) {
        NAPData* data;
        {
            std::unique_lock<std::shared_mutex> lock(read_mtx);
            read_cv.wait(lock, [this] { return nap_queue->notempty(); });
        }
        if (nap_queue->stop()) break;
        data = nap_queue->data();
        gpu_cache->lookup_all_data(data->d_knn_ids, data->batch * bp.k, data->d_cache_idxs,
                                   data->d_vec_cache_idxs, data->d_nbr_vec_cache_idxs,
                                   data->stream, true);
        cpu_neibor_aware(data, view);
        if (data->batch_id % 10 == 0) {
            data->sync();
            // cudaMemcpyAsync(h_total_degree, d_total_degree, sizeof(ull), cudaMemcpyDeviceToHost,
            //                 data->stream);
            // cudaStreamSynchronize(data->stream);
            // fo.print("batch " + TOS(data->batch_id) +
            //          " avg degree: " + TOS(*h_total_degree * 1.0 / bp.base_n));
        }
        int num;
        {
            std::unique_lock<std::shared_mutex> lock(write_mtx);
            num = nap_queue->read_finish();
        }
        write_cv.notify_one();
    }
    fo.print("NAP loop stop");
}

};  // namespace efanna2e
