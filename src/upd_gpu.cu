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
                                         const Slot* __restrict__ slots,
                                         const float* __restrict__ vec_slots,
                                         CN* __restrict__ cand_nbrs,
                                         float* __restrict__ cand_dists, int tile_k, int vec_num,
                                         int batch, BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x,
              cand_size = vec_num + bp.max_degree;
    extern __shared__ float tile_row_vecs[];  // 存储一个 tile 的基础向量，大小为 tile_k * dim
    for (int batch_id = bid; batch_id < batch; batch_id += gridDim.x) {
        const int offset = batch_id * vec_num, *ids = vec_ids + offset;
        const float farthest = farthest_dist_d(bp);
        for (int row_tile = 0; row_tile < vec_num; row_tile += tile_k) {
            // 加载 tile_k 个 row 向量进共享内存:
            for (int i = tid; i < tile_k * bp.dim; i += tpb) {
                // 线程处理的tile 中第 i / dim 个向量在其所在 kNNs 中的索引
                const int global_idx = row_tile + i / bp.dim;
                if (global_idx >= vec_num) break;
                tile_row_vecs[i] = vec_slots[(size_t)ids[global_idx] * bp.dim + i % bp.dim];
            }
            __syncthreads();

            // 计算加载进共享内存的 tile_k 个 row 向量和 vec_num 个列向量的距离
            // 每个线程负责 tile_k * vec_num 距离矩阵中一个点的计算
            for (int i = tid; i < tile_k * cand_size; i += tpb) {
                const int local_row = i / cand_size, global_row = row_tile + local_row;
                if (global_row >= vec_num) break;

                const int row_id = ids[global_row], global_col = i % cand_size,
                          matrix_idx = offset * cand_size + global_row * cand_size + global_col;
                assert(row_id >= 0);
                const Slot& row_slot = slots[row_id];
                const int row_deg = row_slot.deg;
                if (global_col >= vec_num + row_deg || global_row == global_col) {
                    cand_nbrs[matrix_idx] = {-1, 0, Invalid, farthest};
                    cand_dists[matrix_idx] = farthest;
                } else if (global_col >= vec_num) {
                    const int nbr_i = global_col - vec_num, nbr_id = row_slot.nbrs[nbr_i];
                    const float dist = row_slot.dists[nbr_i];
                    cand_nbrs[matrix_idx] = {nbr_id, nbr_id, Initial, dist};
                    cand_dists[matrix_idx] = dist;
                } else if (global_row < global_col) {
                    const int col_id = ids[global_col];
                    const Slot& col_slot = slots[col_id];
                    const int sym_matrix_idx =
                        offset * cand_size + global_col * cand_size + global_row;
                    if (row_id == col_id) {
                        printf("row_id=col_id=%d\n", row_id);
                        cand_nbrs[matrix_idx].status = cand_nbrs[sym_matrix_idx].status = Invalid;
                        cand_dists[matrix_idx] = cand_dists[sym_matrix_idx] =
                            cand_nbrs[matrix_idx].dist = cand_nbrs[sym_matrix_idx].dist = farthest;
                    } else if (row_deg > 0 && col_slot.deg > 0) {
                        cand_nbrs[matrix_idx].status = cand_nbrs[sym_matrix_idx].status = Invalid;
                        cand_dists[matrix_idx] = cand_dists[sym_matrix_idx] =
                            cand_nbrs[matrix_idx].dist = cand_nbrs[sym_matrix_idx].dist = farthest;
                    } else {
                        const float dist = calc_distance(tile_row_vecs + local_row * bp.dim,
                                                         vec_slots + (size_t)col_id * bp.dim, bp);
                        cand_nbrs[matrix_idx] = {col_id, col_id, Initial, dist};
                        cand_nbrs[sym_matrix_idx] = {row_id, row_id, Initial, dist};
                        cand_dists[matrix_idx] = cand_dists[sym_matrix_idx] = dist;
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

__global__ void update_new_nbrs_kernel(const int* __restrict__ pivot_ids, Slot* __restrict__ slots,
                                       CN* __restrict__ cand_nbrs, uint32_t* __restrict__ updated,
                                       ull* __restrict__ total_degree,
                                       ull* __restrict__ noniso_num, int pivot_num, int cand_size,
                                       BP bp, int need_discard) {
    const int tid = threadIdx.x, pivot_i = blockIdx.x * blockDim.x + tid;
    __shared__ int delta_deg, delta_noniso_num;
    if (!tid) delta_deg = delta_noniso_num = 0;
    __syncthreads();

    if (pivot_i < pivot_num) {
        const int pivot_id = pivot_ids[pivot_i];
        assert(pivot_id >= 0);
        if (!atomicTestAndSetBit(updated, pivot_id)) {
            int nbr_num = 0, last_id = -1;
            Slot& slot = slots[pivot_id];
            const CN* cands = cand_nbrs + pivot_i * cand_size;
            if (need_discard) {
                int retained = 0;
                for (int i = 0; i < cand_size && nbr_num < bp.max_degree; i++) {
                    const CN& cand_nbr = cands[i];
                    const auto status = cand_nbr.status;
                    if (status == Retained) {
                        retained++;
                    } else if (status == Invalid)
                        break;
                }
                assert(retained > 0);
                int discarded_need = bp.max_degree - retained;
                for (int j = 0; j < cand_size && nbr_num < bp.max_degree; j++) {
                    const CN& cand_nbr = cands[j];
                    const int cand_nbr_id = cand_nbr.id;
                    assert(cand_nbr_id >= 0);
                    const auto status = cand_nbr.status;
                    if (status == Retained) {
                        assert(cand_nbr_id != last_id);
                        const int idx = nbr_num++;
                        slot.nbrs[idx] = last_id = cand_nbr_id;
                        slot.dists[idx] = cand_nbr.dist;
                    } else if (status == Discarded && (discarded_need--) > 0) {
                        assert(cand_nbr_id != last_id);
                        const int idx = nbr_num++;
                        last_id = slot.nbrs[idx] = cand_nbr_id;
                        slot.dists[idx] = cand_nbr.dist;
                    } else if (status == Invalid)
                        break;
                }
            } else {
                for (int i = 0; i < cand_size && nbr_num < bp.max_degree; i++) {
                    const CN& cand_nbr = cands[i];
                    const auto status = cand_nbr.status;
                    if (status == Retained) {
                        const int cand_nbr_id = cand_nbr.id;
                        assert(cand_nbr_id >= 0 && cand_nbr_id != last_id);
                        const int idx = nbr_num++;
                        slot.nbrs[idx] = last_id = cand_nbr_id;
                        slot.dists[idx] = cand_nbr.dist;
                    } else if (status == Invalid)
                        break;
                }
            }
            if (nbr_num == 0) printf("update_new_nbrs ERROR: nbr_num == 0, pivot: %d\n", pivot_id);
            const int old_deg = slot.deg;
            if (!old_deg) {
                atomicAdd(&delta_noniso_num, 1);
                atomicAdd(&delta_deg, nbr_num);
            } else
                atomicAdd(&delta_deg, nbr_num - old_deg);
            slot.deg = nbr_num;
        }
    }
    __syncthreads();
    if (!tid) {
        atomicAdd(total_degree, (ull)delta_deg);
        atomicAdd(noniso_num, (ull)delta_noniso_num);
    }
}

__device__ __forceinline__ bool angle_greater_alpha(const float* p, const float* i, const float* j,
                                                    const float norm_ip2, float cos_alpha,
                                                    int dim) {
    float dot = 0;
    float norm_ij2 = 0;

    for (int d = 0; d < dim; d++) {
        const float ij = j[d] - i[d];
        dot += (p[d] - i[d]) * ij;
        norm_ij2 += ij * ij;
    }

    return dot * dot < norm_ip2 * norm_ij2 * cos_alpha * cos_alpha;
}

__device__ __forceinline__ bool prune_by_margin_and_angle(const float* __restrict__ p,
                                                          const float* __restrict__ i,
                                                          const float* __restrict__ j, int dim,
                                                          float dist_pi2, float dist_pj2,
                                                          float lambda, float cos_alpha) {
    float dist_ij2 = 0.f, dot = 0.f;

    for (int d = 0; d < dim; ++d) {
        const float id = i[d], ij = j[d] - id;
        dist_ij2 += ij * ij;
        dot += (p[d] - id) * ij;
    }

    // 1) 距离 margin 判定（平方距离版）
    const float one_minus_lambda = 1.0f - lambda;
    if (dist_ij2 < one_minus_lambda * one_minus_lambda * dist_pj2) return true;

    // 2) 角度判定
    const float dot2 = dot * dot;
    return (dot <= 0.f) || (dot2 < cos_alpha * cos_alpha * dist_pi2 * dist_ij2);
}

__global__ void candidate_ignore_kernel(const float* __restrict__ vecs, CN* __restrict__ cand_nbrs,
                                        const int pivot_num, int cand_size, BP bp, float lamda) {
    const int pivot_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (pivot_i >= pivot_num) return;
    int nbr_num = 0;
    CN* cands = cand_nbrs + pivot_i * cand_size;
    for (int i = 0; i < cand_size && nbr_num < bp.max_degree; i++) {
        CN& nbr = cands[i];
        if (nbr.status == Invalid) break;
        const int nbr_id = nbr.id;
        assert(nbr_id >= 0);
        const auto status = nbr.status;
        if (status == Initial) {
            nbr.status = Retained;
            nbr_num++;
            const float* vec = vecs + (size_t)nbr.idx * bp.dim;
            for (int j = i + 1; j < cand_size; j++) {
                CN& nbr1 = cands[j];
                if (nbr1.status == Invalid) break;
                const int nbr1_id = nbr1.id;
                if (nbr1_id == nbr_id)
                    nbr1.status = Repeated;
                else if (nbr1.status == Initial) {
                    const float* vec1 = vecs + (size_t)nbr1.idx * bp.dim;
                    const float pj_dist = nbr1.dist;
                    if (compare_dist(pj_dist, calc_distance(vec, vec1, bp), bp))
                        nbr1.status = Discarded;

                    // if (prune_by_margin_and_angle(pivot_vec, vec, vec1, bp.dim, nbr.dist,
                    // pj_dist,
                    //                               lamda, 0.17))
                    //     nbr1.status = Discarded;
                }
            }
        } else if (status == Discarded)
            for (int j = i + 1; j < cand_size; j++) {
                CN& nbr1 = cands[j];
                if (nbr_id == nbr1.id)
                    nbr1.status = Repeated;
                else if (nbr1.status == Invalid)
                    break;
            }
    }
    // assert(nbr_num > 0);
}

void dist_extract_from_CN(CN* nbrs, float* dists, cudaStream_t stream, int num, BP bp) {
    thrust::device_ptr<CN> nbrs_ptr(nbrs);
    thrust::device_ptr<float> dists_ptr(dists);
    if (bp.metric == DIST_METRIC::L2_)
        thrust::transform(thrust::cuda::par.on(stream), nbrs_ptr, nbrs_ptr + num, dists_ptr,
                          [] __host__ __device__(const CN& nbr) { return nbr.dist; });
    else
        thrust::transform(thrust::cuda::par.on(stream), nbrs_ptr, nbrs_ptr + num, dists_ptr,
                          [] __host__ __device__(const CN& nbr) { return -nbr.dist; });
}

void GPUFuncs::gpu_neibor_aware(const int* d_vec_ids, int batch, int vec_num,
                                cudaStream_t stream) {
    const int cand_size = bp.max_degree + vec_num, pivot_num = batch * vec_num,
              size = pivot_num * cand_size;

    // 计算每组中节点两两间距，生成候选集（包括旧邻居）
    Slot* slots = graph->get_gpu_slots();
    const float* vec_slots = graph->get_gpu_vec_slots();
    pair_dist_compute_kernel<<<batch, 512, tile_k * bp.dim * sizeof(float), stream>>>(
        d_vec_ids, slots, vec_slots, cand_nbrs.data().get(), cand_dists.data().get(), tile_k,
        vec_num, batch, bp);
    CUDA_CHECK(cudaGetLastError());

    // dist_extract_from_CN(cand_nbrs.data().get(), cand_dists.data().get(), stream, size, bp);

    auto res = segmented_sort_pairs_nocpy(
        cand_nbrs.data().get(), cand_dists.data().get(), cand_nbrs_sort.data().get(),
        cand_dists_sort.data().get(), offsets.data().get(), stream, pivot_num, cand_size,
        bp.metric == DIST_METRIC::IP_);

    // cudaStreamSynchronize(stream);
    // std::string str;
    // std::vector<CN> cands(cand_size);
    // cudaMemcpy(cands.data(), res.first, cand_size * sizeof(CN), cudaMemcpyDeviceToHost);
    // for (auto c : cands) {
    //     str += TOS(c.id) + "," + TOS(c.idx) + "," + TOS(c.dist) + "," + TOS(c.status) + "  ";
    // }
    // fo.print(str);

    // 2.2. 针对 batch 组 cand_size 近邻数据，
    // 每次迭代线程并行进行 𝑏𝑎𝑡𝑐ℎ * vec_num * (vec_num − 2 − 𝑖) 次淘汰
    // 根据每个 pivot 候选邻居到 pivot 的距离，从第一个开始，
    // 从近到远成为邻居并并行淘汰后面的后续邻居
    const int tpb = 512, blocks = (pivot_num + tpb - 1) / tpb;
    candidate_ignore_kernel<<<blocks, tpb, 0, stream>>>(vec_slots, res.first, pivot_num, cand_size,
                                                        bp, 0.0);
    CUDA_CHECK(cudaGetLastError());

    thrust::fill(thrust::cuda::par.on(stream), d_updated.begin(), d_updated.end(), 0);
    update_new_nbrs_kernel<<<blocks, tpb, 0, stream>>>(
        d_vec_ids, slots, res.first, d_updated.data().get(), d_total_degree, d_noniso_num,
        pivot_num, cand_size, bp, PC.need_discard);  // 更新新邻居
    CUDA_CHECK(cudaGetLastError());
}

};  // namespace efanna2e
