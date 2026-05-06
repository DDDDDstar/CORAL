#include <cfloat>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>

#include "fileout.h"
#include "gpufuncs.cuh"
#include "utils.cuh"

namespace efanna2e {

struct OffsetCalculator {
    int k;
    __host__ __device__ OffsetCalculator(int _k) : k(_k) {}
    __host__ __device__ int operator()(int i) const { return i * k; }
};

__device__ __forceinline__ float warp_calc_dist(const float* __restrict__ a,
                                                const float* __restrict__ b, int lane, BP bp) {
    float sum = 0.f;
    for (int d = lane; d < bp.dim; d += 32) {
        const float diff = a[d] - b[d];
        sum += (bp.metric == DIST_METRIC::L2_) ? diff * diff : a[d] * b[d];
    }
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
    return __shfl_sync(0xffffffff, sum, 0);
}

__device__ __forceinline__ bool bitTest(const uint32_t* bitset, int bit_index) {
    // assert(bit_index >= 0);
    return (bitset[bit_index >> 5] & (1u << (bit_index & 31))) != 0;
}

__device__ __forceinline__ int atomicTestAndSetBit(uint32_t* bitset, int bit_index) {
    // assert(bit_index >= 0);
    const uint32_t mask = 1u << (bit_index & 31);
    uint32_t* addr = bitset + (bit_index >> 5);
    const uint32_t old = atomicOr(addr, mask);
    return (old & mask) != 0;
}

// __global__ void k_plus_ones_build_kernel(
//     const float* __restrict__ vec_slots, const Slot* __restrict__ slots,
//     const int* __restrict__ insert_ids, const int* __restrict__ query_knns,
//     const int* __restrict__ offsets, int* __restrict__ new_nbr_ids_out, CN* __restrict__
//     cand_nbrs, CN* __restrict__ ins_cand_nbrs, int total_num, int cand_size, int ins_cand_size,
//     int batch, BP bp) { const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x, warp =
//     tid / warpSize,
//               lane = tid % warpSize, wpb = tpb / warpSize;
//     extern __shared__ float s_ins_vec[];  // [dim]
//     __shared__ int s_num, s_start, s_end, s_ins_id;

//     for (int ins_idx = bid; ins_idx < batch; ins_idx += gridDim.x) {
//         const int* offset = offsets + ins_idx;

//         if (tid == 0) {
//             s_start = offset[0];
//             s_end = offset[1];
//             assert(s_end > s_start && s_end <= total_num);
//             s_num = s_end - s_start;  // knn 数量
//             assert(s_num > 0 && s_num <= ins_cand_size);
//             s_ins_id = insert_ids[ins_idx];
//             assert(s_ins_id >= 0);
//         }
//         __syncthreads();
//         const float* ins_vec = vec_slots + s_ins_id * bp.dim;
//         for (int d = tid; d < bp.dim; d += tpb) s_ins_vec[d] = ins_vec[d];
//         __syncthreads();

//         const int* knn = query_knns + s_start;
//         int* new_nbr_ids = new_nbr_ids_out + s_start;
//         const int pivot_num = s_num;
//         for (int i = tid; i < pivot_num; i += tpb) new_nbr_ids[i] = s_ins_id;

//         const float farthest = farthest_dist_d(bp);
//         CN *cands_start = cand_nbrs + s_start * cand_size,
//            *ins_cands = ins_cand_nbrs + ins_idx * ins_cand_size;
//         for (int i = tid; i < pivot_num; i += tpb) {
//             const int id = knn[i];
//             assert(id >= 0);
//             CN* cands = cands_start + i * cand_size;
//             const float dist = calc_distance(s_ins_vec, vec_slots + id * bp.dim, bp);
//             ins_cands[i] = {id, id, Initial, dist};
//             cands[cand_size - 1] = {s_ins_id, s_ins_id, Initial, dist};

//             const Slot* slot = slots + id;
//             const int deg = slot->deg, *nbrs = slot->nbrs;
//             const float* dists = slot->dists;
//             for (int j = 0; j < cand_size - 1; j++)
//                 if (j < deg) {
//                     const int nbr_id = nbrs[j];
//                     assert(nbr_id >= 0);
//                     cands[j] = {nbr_id, nbr_id, Initial, dists[j]};
//                 } else
//                     cands[j] = {-1, 0, Invalid, farthest};
//         }
//         for (int i = pivot_num + tid; i < ins_cand_size; i += tpb)
//             ins_cands[i] = {-1, 0, Invalid, farthest};
//         __syncthreads();
//     }
// }

__global__ void k_plus_ones_build_kernel(
    const float* __restrict__ vec_slots, const int* __restrict__ insert_ids,
    const int* __restrict__ query_knns, const int* __restrict__ offsets,
    int* __restrict__ new_nbr_ids_out, Slot* __restrict__ slots, CN* __restrict__ ins_cand_nbrs,
    float* __restrict__ ins_cand_dists, int* __restrict__ upd_nbrs, uint32_t* __restrict__ updated,
    int total_num, int ins_cand_size, int batch, BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x, warp = tid / warpSize,
              lane = tid % warpSize, wpb = tpb / warpSize;
    extern __shared__ float s_ins_vec[];  // [dim]
    __shared__ int s_num, s_start, s_end, s_ins_id;

    for (int ins_idx = bid; ins_idx < batch; ins_idx += gridDim.x) {
        const int* offset = offsets + ins_idx;

        if (tid == 0) {
            s_start = offset[0];
            s_end = offset[1];
            assert(s_end > s_start && s_end <= total_num);
            s_num = s_end - s_start;  // knn 数量
            assert(s_num > 0 && s_num <= ins_cand_size);
            s_ins_id = insert_ids[ins_idx];
            assert(s_ins_id >= 0);
        }
        __syncthreads();
        const float* ins_vec = vec_slots + s_ins_id * bp.dim;
        for (int d = tid; d < bp.dim; d += tpb) s_ins_vec[d] = ins_vec[d];
        __syncthreads();

        const int* knn = query_knns + s_start;
        int* new_nbr_ids = new_nbr_ids_out + s_start;
        const int pivot_num = s_num;
        for (int i = tid; i < pivot_num; i += tpb) new_nbr_ids[i] = s_ins_id;

        const float farthest = farthest_dist_d(bp);
        CN* ins_cands = ins_cand_nbrs + ins_idx * ins_cand_size;
        float* ins_ds = ins_cand_dists + ins_idx * ins_cand_size;
        for (int i = tid; i < pivot_num; i += tpb) {
            int* upd_nbr = upd_nbrs + (s_start + i) * 2;
            const int id = knn[i];
            assert(id >= 0);
            const float dist = calc_distance(s_ins_vec, vec_slots + id * bp.dim, bp);
            ins_cands[i] = {id, id, Initial, dist};
            ins_ds[i] = dist;

            if (atomicTestAndSetBit(updated, id)) {
                upd_nbr[0] = upd_nbr[1] = -1;
                continue;
            }

            Slot& slot = slots[id];
            const int deg = slot.deg;
            int pos = 0, valid = 1, out_pos;
            for (; pos < deg; ++pos)
                if (compare_dist(dist, slot.dists[pos], bp)) {
                    if (valid &&
                        compare_dist(dist, calc_distance(s_ins_vec, vec_slots, slot.nbrs[pos], bp),
                                     bp))
                        valid = 0;
                } else
                    break;

            if (deg == bp.max_degree) {
                if (!valid || pos >= deg) {
                    upd_nbr[0] = upd_nbr[1] = -1;
                    continue;
                }
                out_pos = deg - 1;
                for (; out_pos >= pos; out_pos--)
                    if (compare_dist(slot.dists[out_pos],
                                     calc_distance(s_ins_vec, vec_slots, slot.nbrs[out_pos], bp),
                                     bp))
                        break;
                if (out_pos < pos) out_pos = deg - 1;
                upd_nbr[0] = slot.nbrs[out_pos];
            } else {
                upd_nbr[0] = -1;
                out_pos = deg;
                slot.deg = deg + 1;
            }

            for (int j = out_pos; j > pos; --j) {
                slot.nbrs[j] = slot.nbrs[j - 1];
                slot.dists[j] = slot.dists[j - 1];
            }
            upd_nbr[1] = slot.nbrs[pos] = s_ins_id;
            slot.dists[pos] = dist;
        }
        for (int i = pivot_num + tid; i < ins_cand_size; i += tpb) {
            ins_cands[i] = {-1, 0, Invalid, farthest};
            ins_ds[i] = farthest;
        }
        __syncthreads();
    }
}

__global__ void get_from_edge_num_kernel(const int* __restrict__ graph,
                                         const int* __restrict__ graph_deg,
                                         int* __restrict__ from_edge_nums_res, int base_n, BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    for (int base_id = bid; base_id < base_n; base_id += gridDim.x)
        for (int nbr_i = tid; nbr_i < graph_deg[base_id]; nbr_i += tpb)
            atomicAdd(from_edge_nums_res + graph[base_id * bp.max_degree + nbr_i], 1);
}

__global__ void get_from_vectors_kernel(const int* __restrict__ graph,
                                        const int* __restrict__ graph_deg,
                                        const int* __restrict__ del_ids,
                                        int* __restrict__ from_ids_res,
                                        int* __restrict__ from_nums_res, int batch, int base_n,
                                        BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    for (int del_idx = bid; del_idx < batch; del_idx += gridDim.x) {
        for (int base_id = tid; base_id < base_n; base_id += tpb) {
            for (int nbr_i = 0; nbr_i < graph_deg[base_id]; nbr_i++) {
                if (graph[base_id * bp.max_degree + nbr_i] == del_ids[del_idx]) {
                    atomicAdd(from_nums_res + del_idx, 1);
                }
            }
        }
    }
}

__global__ void vector_deletion_patch_kernel(
    const int* __restrict__ del_ids, const int* __restrict__ del_innbr_ids,
    const int* __restrict__ del_innbr_offsets, Slot* __restrict__ slots,
    const float* __restrict__ vec_slots, const uint32_t* __restrict__ deleted,
    int* __restrict__ old_nbrs_out, CN* __restrict__ cand_nbrs, int* __restrict__ pivot_ids,
    int* __restrict__ innbr_num_out, int innbrs_num, int cand_size, int batch, BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x, warp = tid / warpSize,
              lane = tid % warpSize, wpb = tpb / warpSize;
    extern __shared__ float sh_outnbr_vecs[];                              // [dim * deg]
    int *sh_outnbr_ids = (int*)(sh_outnbr_vecs + bp.dim * bp.max_degree),  // [deg]
        *sh_del_ids = sh_outnbr_ids + bp.max_degree;                       // [batch]
    __shared__ int s_outnbr_num;

    for (int del_idx = bid; del_idx < batch; del_idx += gridDim.x) {
        for (int i = tid; i < batch; i += tpb) sh_del_ids[i] = del_ids[i];
        __syncthreads();

        const int del_id = sh_del_ids[del_idx];
        const Slot* del_node = slots + del_id;
        if (tid == 0) s_outnbr_num = del_node->deg;
        __syncthreads();

        const int outnbr_num = s_outnbr_num;
        int* nbrs_out = old_nbrs_out + del_idx * bp.max_degree;
        // 检查出邻居中是否有被删除节点:
        if (tid < outnbr_num) {
            const int outnbr_id = del_node->nbrs[tid];
            nbrs_out[tid] = bitTest(deleted, outnbr_id) ? -1 : outnbr_id;
        } else if (tid < bp.max_degree)
            nbrs_out[tid] = -1;
        __syncthreads();

        for (int i = tid; i < outnbr_num * bp.dim; i += tpb) {
            const int nbr_i = i / bp.dim, d = i % bp.dim;
            if (sh_outnbr_ids[nbr_i] < 0) continue;
            sh_outnbr_vecs[i] = vec_slots[sh_outnbr_ids[nbr_i] * bp.dim + d];
        }
        __syncthreads();

        const int innbr_start = del_innbr_offsets[del_idx],
                  innbr_end = del_innbr_offsets[del_idx + 1];
        for (int innbr_idx = innbr_start + tid; innbr_idx < innbr_end; innbr_idx += tpb) {
            assert(innbr_idx < innbrs_num);
            const int innbr_id = del_innbr_ids[innbr_idx];
            Slot* innbr = slots + innbr_id;
            if (bitTest(deleted, innbr_id)) {  // 如果入邻居也被删除了，则跳过（不更新了）
                innbr->deg = 0;
                continue;
            }
            const int pivot_idx = atomicAdd(innbr_num_out, 1);
            const float* innbr_vec = vec_slots + innbr_id * bp.dim;
            const int old_nbr_num = innbr->deg;
            CN* out_cands = cand_nbrs + pivot_idx * cand_size;
            pivot_ids[pivot_idx] = innbr_id;
            int nbr_i = 0;
            for (int old_nbr_i = 0; old_nbr_i < old_nbr_num; old_nbr_i++) {
                const int id = innbr->nbrs[old_nbr_i];
                assert(id >= 0);
                if (bitTest(deleted, id)) continue;  // 跳过被删除节点!
                out_cands[nbr_i] = {id, id, Initial, innbr->dists[old_nbr_i]};
                nbr_i++;
            }
            for (int outnbr_i = 0; outnbr_i < outnbr_num; outnbr_i++) {
                const int id = sh_outnbr_ids[outnbr_i];
                if (id == -1) continue;
                out_cands[nbr_i] = {
                    id, id, Initial,
                    calc_distance(innbr_vec, sh_outnbr_vecs + outnbr_i * bp.dim, bp)};
                nbr_i++;
            }
            // assert(nbr_i > 0);
            if (nbr_i <= 0 || nbr_i > cand_size) {
                printf("ERROR: nbr_i = %d with cand_size = %d\n", nbr_i, cand_size);
                assert(false);
            }
            const float farthest = farthest_dist_d(bp);
            for (; nbr_i < cand_size; nbr_i++) out_cands[nbr_i] = {-1, 0, Invalid, farthest};
        }
    }
}

__global__ void vector_deletion_patch_kernel(
    const int* __restrict__ del_ids, const int* __restrict__ del_innbr_ids,
    const int* __restrict__ del_innbr_offsets, Slot* __restrict__ slots,
    const float* __restrict__ vec_slots, const uint32_t* __restrict__ deleted,
    CN* __restrict__ cand_nbrs, float* __restrict__ cand_dists, int innbrs_num, int cand_size,
    int batch, BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x, warp = tid / warpSize,
              lane = tid % warpSize, wpb = tpb / warpSize;
    extern __shared__ float sh_outnbr_vecs[];                              // [dim * deg]
    int *sh_outnbr_ids = (int*)(sh_outnbr_vecs + bp.dim * bp.max_degree),  // [deg]
        *sh_del_ids = sh_outnbr_ids + bp.max_degree;                       // [batch]
    __shared__ int s_outnbr_num;

    for (int del_idx = bid; del_idx < batch; del_idx += gridDim.x) {
        for (int i = tid; i < batch; i += tpb) sh_del_ids[i] = del_ids[i];
        __syncthreads();

        const int del_id = sh_del_ids[del_idx];
        // assert(del_id >= 0 && del_id < bp.base_n);

        const Slot* del_node = slots + del_id;
        if (tid == 0) s_outnbr_num = del_node->deg;
        __syncthreads();

        const int outnbr_num = s_outnbr_num;
        // 检查出邻居中是否有被删除节点:
        if (tid < outnbr_num) {
            const int outnbr_id = del_node->nbrs[tid];
            sh_outnbr_ids[tid] = bitTest(deleted, outnbr_id) ? -1 : outnbr_id;
        } else if (tid < bp.max_degree)
            sh_outnbr_ids[tid] = -1;
        __syncthreads();

        for (int i = tid; i < outnbr_num * bp.dim; i += tpb) {
            const int nbr_i = i / bp.dim, d = i % bp.dim;
            if (sh_outnbr_ids[nbr_i] < 0) continue;
            sh_outnbr_vecs[i] = vec_slots[sh_outnbr_ids[nbr_i] * bp.dim + d];
        }
        __syncthreads();

        const int innbr_start = del_innbr_offsets[del_idx],
                  innbr_end = del_innbr_offsets[del_idx + 1];
        for (int innbr_idx = innbr_start + tid; innbr_idx < innbr_end; innbr_idx += tpb) {
            assert(innbr_idx < innbrs_num);
            const int innbr_id = del_innbr_ids[innbr_idx];
            Slot* innbr = slots + innbr_id;
            const float* innbr_vec = vec_slots + innbr_id * bp.dim;
            const int old_nbr_num = innbr->deg;
            CN* out_cands = cand_nbrs + innbr_idx * cand_size;
            float* out_dists = cand_dists + innbr_idx * cand_size;
            int nbr_i = 0;
            for (int old_nbr_i = 0; old_nbr_i < old_nbr_num; old_nbr_i++) {
                const int id = innbr->nbrs[old_nbr_i];
                // assert(id >= 0);
                if (bitTest(deleted, id)) continue;  // 跳过被删除节点!
                const float d = innbr->dists[old_nbr_i];
                out_cands[nbr_i] = {id, id, Initial, d};
                out_dists[nbr_i] = d;
                nbr_i++;
            }
            for (int outnbr_i = 0; outnbr_i < outnbr_num; outnbr_i++) {
                const int id = sh_outnbr_ids[outnbr_i];
                if (id == -1) continue;
                const float d = calc_distance(innbr_vec, sh_outnbr_vecs + outnbr_i * bp.dim, bp);
                out_cands[nbr_i] = {id, id, Initial, d};
                out_dists[nbr_i] = d;
                nbr_i++;
            }
            assert(nbr_i > 0);
            if (nbr_i <= 0 || nbr_i > cand_size) {
                printf("ERROR: nbr_i = %d with cand_size = %d\n", nbr_i, cand_size);
                // atomicAdd(error_code, 1);
                // assert(false);
            }
            const float farthest = farthest_dist_d(bp);
            for (; nbr_i < cand_size; nbr_i++) {
                out_cands[nbr_i] = {-1, 0, Invalid, farthest};
                out_dists[nbr_i] = farthest;
            }
        }
    }
}

__global__ void gather_nbrs_and_deg_kernel(const int* __restrict__ ids,
                                           const Slot* __restrict__ slots,
                                           int* __restrict__ nbrs_out, int* __restrict__ degs_out,
                                           int N, BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    __shared__ int s_deg;
    for (int i = bid; i < N; i += gridDim.x) {
        const Slot* slot = slots + ids[i];
        const int* in = slot->nbrs;
        int* out = nbrs_out + i * bp.max_degree;
        if (!tid) degs_out[i] = s_deg = slot->deg;
        __syncthreads();
        if (tid < bp.max_degree) out[tid] = tid < s_deg ? in[tid] : -1;
    }
}

__global__ void del_node_mark_kerenl(const int* __restrict__ del_ids,
                                     uint32_t* __restrict__ deleted, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int id = del_ids[i];
    atomicOr(deleted + (id >> 5), 1u << (id & 31));
}

__global__ void update_new_nbrs_kernel(
    const int* __restrict__ pivot_ids, const int* __restrict__ new_nbr_ids,
    const CN* __restrict__ cand_nbrs, Slot* __restrict__ slots,
    int* __restrict__ upd_nbrs,  // [pivot_num * 2]: 第一个为移除的 nbr，第二个为新增的 nbr
    uint32_t* __restrict__ updated, ull* __restrict__ total_degree, ull* __restrict__ noniso_num,
    int pivot_num, int cand_size, BP bp, int* error = nullptr) {
    const int tid = threadIdx.x, pivot_i = blockIdx.x * blockDim.x + tid;
    __shared__ int delta_deg, delta_noniso_num;
    if (!tid) delta_deg = delta_noniso_num = 0;
    __syncthreads();

    if (pivot_i < pivot_num) {
        const int pivot_id = pivot_ids[pivot_i], new_nbr_id = new_nbr_ids[pivot_i];
        int* upd_nbr = upd_nbrs + pivot_i * 2;
        assert(pivot_id >= 0);
        if (atomicTestAndSetBit(updated, pivot_id) == 0) {
            int nbr_num = 0, last_id = -1, nbr_xor = 0;
            Slot& slot = slots[pivot_id];
            const int *nbrs = slot.nbrs, deg = slot.deg;
            for (int i = 0; i < deg; i++) nbr_xor ^= nbrs[i];
            if ((bp.max_degree - deg) % 2 == 1) nbr_xor ^= -1;

            const CN* cands = cand_nbrs + pivot_i * cand_size;
            for (int i = 0; i < cand_size && nbr_num < bp.max_degree; i++) {
                // if (atomicAdd(error, 0) > 0) break;
                const CN& cand_nbr = cands[i];
                if (cand_nbr.status == Retained) {
                    const int cand_nbr_id = cand_nbr.id;
                    assert(cand_nbr_id >= 0);
                    if (cand_nbr_id == last_id) {
                        printf("update_new_nbrs ERROR: cand_nbr_id == last_id pivot: %d\n",
                               pivot_id);
                        // atomicAdd(error, 1);
                        break;
                    }
                    const int idx = nbr_num++;
                    slot.nbrs[idx] = last_id = cand_nbr_id;
                    slot.dists[idx] = cand_nbr.dist;
                    nbr_xor ^= cand_nbr_id;
                } else if (cand_nbr.status == Invalid)
                    break;
            }
            for (int j = 0; j < cand_size && nbr_num < bp.max_degree; j++) {
                // if (atomicAdd(error, 0) > 0) break;
                const CN& cand_nbr = cands[j];
                const int cand_nbr_id = cand_nbr.id;
                if (cand_nbr.status == Discarded) {
                    assert(cand_nbr_id >= 0);
                    if (cand_nbr_id == last_id) {
                        printf("update_new_nbrs ERROR: cand_nbr_id == last_id pivot: %d\n",
                               pivot_id);
                        // atomicAdd(error, 1);
                        break;
                    }
                    const int idx = nbr_num++;
                    last_id = slot.nbrs[idx] = cand_nbr_id;
                    slot.dists[idx] = cand_nbr.dist;
                    nbr_xor ^= cand_nbr_id;
                } else if (cand_nbr.status == Invalid)
                    break;
            }
            assert(nbr_num > 0);

            if ((bp.max_degree - nbr_num) % 2 == 1) nbr_xor ^= -1;
            if (!nbr_xor)
                upd_nbr[0] = upd_nbr[1] = -1;
            else {
                upd_nbr[0] = nbr_xor ^ new_nbr_id;  // 被替换掉的旧邻居
                upd_nbr[1] = new_nbr_id;
            }

            const int old_deg = slot.deg;
            if (!old_deg) {
                atomicAdd(&delta_noniso_num, 1);
                atomicAdd(&delta_deg, nbr_num);
            } else
                atomicAdd(&delta_deg, nbr_num - old_deg);
            slot.deg = nbr_num;
        } else
            upd_nbr[0] = upd_nbr[1] = -1;
    }
    __syncthreads();
    if (!tid) {
        atomicAdd(total_degree, (ull)delta_deg);
        atomicAdd(noniso_num, (ull)delta_noniso_num);
    }
}

__global__ void update_and_collect_nbrs_kernel(
    const int* __restrict__ pivot_ids, const CN* __restrict__ cand_nbrs, Slot* __restrict__ slots,
    int* __restrict__ old_nbrs,  // [pivot_num * max_degree]
    int* __restrict__ new_nbrs,  // [pivot_num * max_degree]
    uint32_t* __restrict__ updated, ull* __restrict__ total_degree, ull* __restrict__ noniso_num,
    int pivot_num, int cand_size, BP bp) {
    const int tid = threadIdx.x, pivot_i = blockIdx.x * blockDim.x + tid;
    __shared__ int delta_deg, delta_noniso_num;
    if (!tid) delta_deg = delta_noniso_num = 0;
    __syncthreads();

    if (pivot_i < pivot_num) {
        const int pivot_id = pivot_ids[pivot_i];
        assert(pivot_id >= 0);
        if (!atomicTestAndSetBit(updated, pivot_id)) {
            int nbr_num = 0, last_id = -1;
            const CN* cands = cand_nbrs + pivot_i * cand_size;
            Slot& slot = slots[pivot_id];
            if (old_nbrs != nullptr) {
                int* old_nbrs_out = old_nbrs + pivot_i * bp.max_degree;
                const int old_deg = slot.deg;
                for (int i = 0; i < old_deg; i++) old_nbrs_out[i] = slot.nbrs[i];
                for (int i = old_deg; i < bp.max_degree; i++) old_nbrs_out[i] = -1;
            }
            int* nbrs_out = new_nbrs + pivot_i * bp.max_degree;
            for (int i = 0; i < cand_size && nbr_num < bp.max_degree; i++) {
                const CN& cand_nbr = cands[i];
                if (cand_nbr.status == Retained) {
                    const int cand_nbr_id = cand_nbr.id;
                    if (cand_nbr_id < 0) {
                        printf("ins update_new_nbrs ERROR: cand_nbr.id < 0 pivot: %d\n", pivot_id);
                        continue;
                    }
                    if (cand_nbr_id == last_id) {
                        printf("ins update_new_nbrs ERROR: cand_nbr_id == last_id pivot: %d\n",
                               pivot_id);
                        continue;
                    }
                    const int idx = nbr_num++;
                    nbrs_out[idx] = slot.nbrs[idx] = last_id = cand_nbr_id;
                    slot.dists[idx] = cand_nbr.dist;
                } else if (cand_nbr.status == Invalid)
                    break;
            }
            for (int j = 0; j < cand_size && nbr_num < bp.max_degree; j++) {
                const CN& cand_nbr = cands[j];
                const int cand_nbr_id = cand_nbr.id;
                if (cand_nbr.status == Discarded) {
                    if (cand_nbr_id < 0) {
                        printf("ins update_new_nbrs ERROR: cand_nbr.id < 0 pivot: %d\n", pivot_id);
                        continue;
                    }
                    if (cand_nbr_id == last_id) {
                        printf("ins update_new_nbrs ERROR: cand_nbr_id == last_id pivot: %d\n",
                               pivot_id);
                        continue;
                    }
                    const int idx = nbr_num++;
                    nbrs_out[idx] = last_id = slot.nbrs[idx] = cand_nbr_id;
                    slot.dists[idx] = cand_nbr.dist;
                } else if (cand_nbr.status == Invalid)
                    break;
            }
            assert(nbr_num > 0);
            for (int j = nbr_num; j < bp.max_degree; j++) nbrs_out[j] = -1;
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

__global__ void candidate_ignore_kernel_for_ins(const float* __restrict__ vecs,
                                                CN* __restrict__ cand_nbrs, const int* pivot_ids,
                                                int pivot_num, int cand_size, BP bp) {
    const int pivot_i = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    if (pivot_i >= pivot_num) return;

    __shared__ int nbr_num;
    if (!tid) nbr_num = 0;
    __syncthreads();

    const float* pivot_vec = vecs + pivot_ids[pivot_i] * bp.dim;
    CN* cands = cand_nbrs + pivot_i * cand_size;
    for (int i = 0; i < cand_size && nbr_num < bp.max_degree; i++) {
        CN& nbr = cands[i];
        if (nbr.status == Invalid) break;
        const int nbr_id = nbr.id;
        assert(nbr_id >= 0);
        const auto status = nbr.status;
        if (status == Initial) {
            if (!tid) {
                nbr.status = Retained;
                nbr_num++;
            }
            const float* vec = vecs + nbr_id * bp.dim;
            for (int j = i + 1 + tid; j < cand_size; j += tpb) {
                CN& nbr1 = cands[j];
                if (nbr1.status == Invalid) break;
                const int nbr1_id = nbr1.id;
                if (nbr1_id == nbr_id)
                    nbr1.status = Repeated;
                else if (nbr1.status == Initial) {
                    const float* vec1 = vecs + nbr1_id * bp.dim;
                    const float pj_dist = nbr1.dist;
                    if (compare_dist(pj_dist, calc_distance(vec, vec1, bp), bp))
                        nbr1.status = Discarded;
                }
            }
        } else if (status == Discarded)
            for (int j = i + 1 + tid; j < cand_size; j += tpb) {
                CN& nbr1 = cands[j];
                if (nbr_id == nbr1.id)
                    nbr1.status = Repeated;
                else if (nbr1.status == Invalid)
                    break;
            }

        __syncthreads();
    }
}

__global__ void merge_paired_knns_kernel(const int* __restrict__ knns,
                                         const int* __restrict__ knn_offsets,
                                         const int* __restrict__ search_nbrs, int k, int batch,
                                         int* __restrict__ knns_res,
                                         int* __restrict__ knn_offsets_res) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;

    if (!bid && !tid) knn_offsets_res[0] = 0;

    __shared__ int s_start, s_num;

    for (int i = bid; i < batch; i += gridDim.x) {
        const int start = knn_offsets[i], end = knn_offsets[i + 1], num = end - start;
        if (!tid) {
            s_start = start + i * k;
            knn_offsets_res[i + 1] = end + (i + 1) * k;
            s_num = num + k;
        }
        __syncthreads();

        int* knns_out = knns_res + s_start;
        for (int j = tid; j < s_num; j += tpb)
            knns_out[j] = j < k ? search_nbrs[i * k + j] : knns[start + j - k];
        __syncthreads();
    }
}

void GPUFuncs::del_node_mark(const std::vector<int>& del_ids) {
    const int del_num = del_ids.size();
    cudaStream_t stream = streams["upd"].stream;

    if (d_all_del_ids.size() < del_num) d_all_del_ids.resize(del_num);
    CUDA_CHECK(cudaMemcpyAsync(d_all_del_ids.data().get(), del_ids.data(), sizeof(int) * del_num,
                               cudaMemcpyHostToDevice, stream));

    const int tpb = 512, blocks = (del_num + tpb - 1) / tpb;
    del_node_mark_kerenl<<<blocks, tpb, 0, stream>>>(d_all_del_ids.data().get(),
                                                     d_deleted.data().get(), del_num);
    CUDA_CHECK(cudaGetLastError());
}

float GPUFuncs::vector_delete(const int* del_ids, const std::vector<int>& del_innbr_offsets,
                              const int* del_innbrs, int del_num, int innbrs_num) {
    // std::string input;
    // for (int i = 0; i < 10; i++) input += TOS(del_ids[i]) + " ";
    // input += "\n";
    // for (int i = 0; i <= 10; i++) input += TOS(del_innbr_offsets[i]) + " ";
    // input += "\n";
    // for (int i = 0; i < del_innbr_offsets[10]; i++) input += TOS(del_innbrs[i]) + " ";
    // fo.print("vector_delete input: \n" + input);

    cudaStream_t stream = streams["upd"].stream;

    auto e = gpu_record_time_start(stream);

    int *h_del_old_nbrs = nullptr, *h_del_new_nbrs = nullptr;
    if (d_del_ids.size() < del_num) d_del_ids.resize(del_num);
    if (d_del_innbr_ids.size() < innbrs_num) {
        d_del_innbr_ids.resize(innbrs_num);
        // d_innbr_ids.resize(innbrs_num);
    }
    if (d_del_innbr_offsets.size() < del_num + 1) d_del_innbr_offsets.resize(del_num + 1);

    if (d_del_new_nbrs.size() < innbrs_num * bp.max_degree) {
        d_del_old_nbrs.resize(innbrs_num * bp.max_degree);
        d_del_new_nbrs.resize(innbrs_num * bp.max_degree);
    }
    CUDA_CHECK(cudaMallocHost(&h_del_old_nbrs, sizeof(int) * innbrs_num * bp.max_degree));
    CUDA_CHECK(cudaMallocHost(&h_del_new_nbrs, sizeof(int) * innbrs_num * bp.max_degree));

    CUDA_CHECK(cudaMemcpyAsync(d_del_ids.data().get(), del_ids, sizeof(int) * del_num,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_del_innbr_ids.data().get(), del_innbrs, sizeof(int) * innbrs_num,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_del_innbr_offsets.data().get(), del_innbr_offsets.data(),
                               (del_num + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));

    Slot* slots = graph->get_gpu_slots();
    const float* vec_slots = graph->get_gpu_vec_slots();

    const int threads = 512, blocks = (innbrs_num + threads - 1) / threads;
    const int cand_size = 2 * bp.max_degree;
    const size_t share_size =
        (bp.max_degree + del_num) * sizeof(int) + bp.dim * bp.max_degree * sizeof(float);
    vector_deletion_patch_kernel<<<del_num, threads, share_size, stream>>>(
        d_del_ids.data().get(), d_del_innbr_ids.data().get(), d_del_innbr_offsets.data().get(),
        slots, vec_slots, d_deleted.data().get(), del_cand_nbrs.data().get(),
        del_cand_dists.data().get(), innbrs_num, cand_size, del_num, bp);
    CUDA_CHECK(cudaGetLastError());

    float pre_time;
    e = gpu_record_time_reset(e, stream, pre_time, "vector delete pre");
    const int total_cand_size = innbrs_num * cand_size;
    // dist_extract_from_CN(del_cand_nbrs.data().get(), del_cand_dists.data().get(), stream,
    //                      total_cand_size, bp);
    auto res = segmented_sort_pairs_nocpy(
        del_cand_nbrs.data().get(), del_cand_dists.data().get(), del_cand_nbrs_sort.data().get(),
        del_cand_dists_sort.data().get(), del_offsets.data().get(), stream, innbrs_num, cand_size);

    candidate_ignore_kernel<<<blocks, threads, 0, stream>>>(vec_slots, res.first, innbrs_num,
                                                            cand_size, bp, 0.0);
    CUDA_CHECK(cudaGetLastError());

    thrust::fill(thrust::cuda::par.on(stream), d_updated.begin(), d_updated.end(), 0);
    CUDA_CHECK(cudaStreamWaitEvent(stream, cpy_event));
    update_and_collect_nbrs_kernel<<<blocks, threads, 0, stream>>>(
        d_del_innbr_ids.data().get(), res.first, slots, d_del_old_nbrs.data().get(),
        d_del_new_nbrs.data().get(), d_updated.data().get(), d_total_degree, d_noniso_num,
        innbrs_num, cand_size, bp);
    CUDA_CHECK(cudaGetLastError());

    // gather_nbrs_and_deg_kernel<<<del_num, bp.max_degree, 0, stream>>>(
    //     d_del_innbr_ids.data().get(), slots, d_del_new_nbrs.data().get(),
    //     d_del_new_degs.data().get(), innbrs_num, bp);
    // CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(h_del_old_nbrs, d_del_old_nbrs.data().get(),
                               sizeof(int) * innbrs_num * bp.max_degree, cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(h_del_new_nbrs, d_del_new_nbrs.data().get(),
                               sizeof(int) * innbrs_num * bp.max_degree, cudaMemcpyDeviceToHost,
                               stream));

    const float time = gpu_record_time_stop(e, stream, "vector delete");
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // std::async(std::launch::async, [this, del_ids, del_innbrs, del_num, innbrs_num]() {
    //     graph->delete_nodes(del_ids, del_innbrs, h_del_old_nbrs, h_del_new_nbrs, h_del_new_degs,
    //                         del_num, innbrs_num);
    // });
    executor.submit(
        [this, del_ids, del_innbrs, h_del_old_nbrs, h_del_new_nbrs, del_num, innbrs_num]() {
            graph->delete_nodes(del_ids, del_innbrs, h_del_old_nbrs, h_del_new_nbrs, del_num,
                                innbrs_num);
            CUDA_CHECK(cudaFreeHost(h_del_old_nbrs));
            CUDA_CHECK(cudaFreeHost(h_del_new_nbrs));
        });
    // fo.print("vector delete(" + TOS(pre_time) + "s, " + TOS(time) + "s) : N = " + TOS(del_num) +
    //          ", innbrs_num = " + TOS(innbrs_num));
    return pre_time + time;
}

int GPUFuncs::obtain_paired_knns(const float* h_insert_vectors, const int* ins_ids,
                                 int*& h_knns_out, int batch) {
    cudaStream_t stream = streams["upd"].stream, cpy_stream = streams["upd_cpy"].stream;
    int pivot_num;
    auto start_ids = graph->get_search_start_nodes_fps(PC.search_start_nodes_num, 1);
    if (PC.upd_type == 0) {
        if (d_paired_knns.size() < batch * bp.k) d_paired_knns.resize(batch * bp.k);
        cudaMemcpy(d_ins_vecs, h_insert_vectors, batch * bp.dim * sizeof(float),
                   cudaMemcpyHostToDevice);

        search_gpu(d_ins_vecs, d_paired_knns.data().get(), nullptr, batch, bp.k, start_ids, true);

        pivot_num = batch * bp.k;
        CUDA_CHECK(cudaMallocHost(&h_knns_out, sizeof(int) * pivot_num));
        CUDA_CHECK(cudaMemcpy(h_knns_out, d_paired_knns.data().get(), pivot_num * sizeof(int),
                              cudaMemcpyDeviceToHost));
        // std::string str;
        // for (int i = 0; i < bp.k; i++) str += TOS(h_knns_out[i]) + " ";
        // fo.print("paired knns: " + str);
        query_index->get_query_of_base(d_ins_vecs, h_knns_out, h_paired_query_ids.data(), batch,
                                       bp.k, bp);
        auto fut =
            executor.submit_with_future([this, ins_ids, batch]() {  // 找到配对查询向量的 knns
                query_index->get_knns().get_and_add(h_paired_query_ids, ins_ids, paired_query_knns,
                                                    query_knn_offsets, batch, 1);
            });
        fut.get();
        pivot_num = paired_query_knns.size();
        CUDA_CHECK(cudaMallocHost(&h_knns_out, sizeof(int) * pivot_num));
        memcpy(h_knns_out, paired_query_knns.data(), sizeof(int) * pivot_num);
        if (d_paired_knns.size() < pivot_num) d_paired_knns.resize(pivot_num);
        CUDA_CHECK(cudaMemcpyAsync(d_paired_knns.data().get(), h_knns_out, pivot_num * sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_knn_offsets.data().get(), query_knn_offsets.data(),
                                   (batch + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));

    } else if (PC.upd_type == 1) {
        if (d_paired_knns.size() < batch * bp.k) d_paired_knns.resize(batch * bp.k);
        cudaMemcpyAsync(d_ins_vecs, h_insert_vectors, batch * bp.dim * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
        search_gpu(d_ins_vecs, d_paired_knns.data().get(), nullptr, batch, bp.k, start_ids, false,
                   stream);
        pivot_num = batch * bp.k;
        CUDA_CHECK(cudaMallocHost(&h_knns_out, sizeof(int) * pivot_num));
        CUDA_CHECK(cudaMemcpyAsync(h_knns_out, d_paired_knns.data().get(), pivot_num * sizeof(int),
                                   cudaMemcpyDeviceToHost, cpy_stream));
        thrust::counting_iterator<int> count_iter(0);
        thrust::transform(thrust::cuda::par.on(stream), count_iter, count_iter + batch + 1,
                          d_knn_offsets.begin(), OffsetCalculator(bp.k));
    } else if (PC.upd_type == 2) {
        query_index->Topk_Search_faiss(h_insert_vectors, h_paired_query_ids, batch,
                                       PC.insert_query_k);
        auto fut =
            executor.submit_with_future([this, ins_ids, batch]() {  // 找到配对查询向量的 knns
                query_index->get_knns().get_and_add(h_paired_query_ids, ins_ids, paired_query_knns,
                                                    query_knn_offsets, batch, PC.insert_query_k);
            });
        fut.get();
        pivot_num = paired_query_knns.size();
        CUDA_CHECK(cudaMallocHost(&h_knns_out, sizeof(int) * pivot_num));
        memcpy(h_knns_out, paired_query_knns.data(), sizeof(int) * pivot_num);
        if (d_paired_knns.size() < pivot_num) d_paired_knns.resize(pivot_num);
        CUDA_CHECK(cudaMemcpyAsync(d_paired_knns.data().get(), h_knns_out, pivot_num * sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_knn_offsets.data().get(), query_knn_offsets.data(),
                                   (batch + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));
    } else if (PC.upd_type == 3) {
        const size_t search_knns_num = batch * bp.k;
        if (d_search_knns.size() < search_knns_num) d_search_knns.resize(search_knns_num);

        cudaMemcpyAsync(d_ins_vecs, h_insert_vectors, batch * bp.dim * sizeof(float),
                        cudaMemcpyHostToDevice, stream);

        query_index->Topk_Search_faiss(h_insert_vectors, h_paired_query_ids, batch,
                                       PC.insert_query_k);
        auto fut =
            executor.submit_with_future([this, ins_ids, batch]() {  // 找到配对查询向量的 knns
                query_index->get_knns().get_and_add(h_paired_query_ids, ins_ids, paired_query_knns,
                                                    query_knn_offsets, batch, PC.insert_query_k);
            });

        search_gpu(d_ins_vecs, d_search_knns.data().get(), nullptr, batch, bp.k, start_ids, false,
                   stream);

        fut.get();
        pivot_num = paired_query_knns.size();

        CUDA_CHECK(cudaMallocHost(&h_knns_out, sizeof(int) * (pivot_num + search_knns_num)));
        memcpy(h_knns_out, paired_query_knns.data(), sizeof(int) * pivot_num);

        if (d_paired_knns1.size() < pivot_num) d_paired_knns1.resize(pivot_num);
        CUDA_CHECK(cudaMemcpyAsync(d_paired_knns1.data().get(), h_knns_out,
                                   pivot_num * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_knn_offsets1.data().get(), query_knn_offsets.data(),
                                   (batch + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));

        pivot_num += batch * bp.k;
        if (d_paired_knns.size() < pivot_num) d_paired_knns.resize(pivot_num);
        merge_paired_knns_kernel<<<batch, 256, 0, stream>>>(
            d_paired_knns1.data().get(), d_knn_offsets1.data().get(), d_search_knns.data().get(),
            bp.k, batch, d_paired_knns.data().get(), d_knn_offsets.data().get());
        CUDA_CHECK(cudaMemcpyAsync(h_knns_out, d_paired_knns.data().get(), pivot_num * sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
    }
    return pivot_num;
}

float GPUFuncs::vector_insert(const float* h_insert_vectors, const int* ins_ids, int batch) {
    // fo.eprint("TODO: vector_insert");
    cudaStream_t stream = streams["upd"].stream, cpy_stream = streams["upd_cpy"].stream;

    auto s = now_time();

    int *h_knns = nullptr, *h_upd_nbrs = nullptr, *h_new_nbrs = nullptr;
    const int tpb = 512, cand_size = bp.max_degree + 1,
              ins_cand_size = PC.upd_type == 1 ? bp.k
                                               : (PC.upd_type == 3 ? (PC.insert_query_k + 1) * bp.k
                                                                   : PC.insert_query_k * bp.k);
    const int pivot_num = obtain_paired_knns(h_insert_vectors, ins_ids, h_knns, batch);

    auto s1 = now_time();
    const float get_knns_time = time_diff(s, s1);

    CUDA_CHECK(cudaMallocHost(&h_upd_nbrs, sizeof(int) * pivot_num * 2));
    CUDA_CHECK(cudaMallocHost(&h_new_nbrs, sizeof(int) * batch * bp.max_degree));

    auto e = gpu_record_time_start(stream);
    CUDA_CHECK(cudaMemcpyAsync(d_ins_ids.data().get(), ins_ids, batch * sizeof(int),
                               cudaMemcpyHostToDevice, stream));

    const float* vec_slots = graph->get_gpu_vec_slots();
    Slot* slots = graph->get_gpu_slots();
    thrust::fill(thrust::cuda::par.on(stream), d_updated.begin(), d_updated.end(), 0);
    k_plus_ones_build_kernel<<<batch, tpb, 0, stream>>>(
        vec_slots, d_ins_ids.data().get(), d_paired_knns.data().get(), d_knn_offsets.data().get(),
        d_new_nbr_ids.data().get(), slots, ins_cand_nbrs.data().get(), cand_dists.data().get(),
        d_upd_nbrs.data().get(), d_updated.data().get(), pivot_num, ins_cand_size, batch, bp);
    CUDA_CHECK(cudaGetLastError());

    const int total_cand_size = pivot_num * cand_size, blocks = (pivot_num + tpb - 1) / tpb,
              blocks1 = (batch + tpb - 1) / tpb;

    CUDA_CHECK(cudaEventRecord(upd_event, stream));
    CUDA_CHECK(cudaStreamWaitEvent(cpy_stream, upd_event));
    CUDA_CHECK(cudaMemcpyAsync(h_upd_nbrs, d_upd_nbrs.data().get(), sizeof(int) * pivot_num * 2,
                               cudaMemcpyDeviceToHost, cpy_stream));
    CUDA_CHECK(cudaEventRecord(cpy_event, cpy_stream));

    executor.submit([this, h_knns, h_upd_nbrs, pivot_num]() {
        CUDA_CHECK(cudaEventSynchronize(cpy_event));
        graph->update_innbrs_with_upd_nbrs(h_knns, h_upd_nbrs, pivot_num);
        CUDA_CHECK(cudaFreeHost(h_knns));
        CUDA_CHECK(cudaFreeHost(h_upd_nbrs));
    });

    // dist_extract_from_CN(ins_cand_nbrs.data().get(), cand_dists.data().get(), stream,
    //                      ins_cand_size * batch, bp);
    auto res1 = segmented_sort_pairs_nocpy(
        ins_cand_nbrs.data().get(), cand_dists.data().get(), ins_cand_nbrs_sort.data().get(),
        cand_dists_sort.data().get(), offsets.data().get(), stream, batch, ins_cand_size);

    candidate_ignore_kernel_for_ins<<<batch, tpb, 0, stream>>>(
        vec_slots, res1.first, d_ins_ids.data().get(), batch, ins_cand_size, bp);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamWaitEvent(stream, cpy_event));  // 等待 d_upd_nbrs 数据拷贝完成再复用
    if (d_upd_nbrs.size() < batch * bp.max_degree) d_upd_nbrs.resize(batch * bp.max_degree);
    update_and_collect_nbrs_kernel<<<blocks1, tpb, 0, stream>>>(
        d_ins_ids.data().get(), res1.first, slots, nullptr, d_upd_nbrs.data().get(),
        d_updated.data().get(), d_total_degree, d_noniso_num, batch, ins_cand_size, bp);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(h_new_nbrs, d_upd_nbrs.data().get(),
                               sizeof(int) * batch * bp.max_degree, cudaMemcpyDeviceToHost,
                               stream));

    executor.submit([this, stream, ins_ids, h_new_nbrs, batch]() {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        graph->update_innbrs_with_new_nbrs(ins_ids, h_new_nbrs, batch);
        CUDA_CHECK(cudaFreeHost(h_new_nbrs));
    });

    const float nap_time = gpu_record_time_stop(e, stream, "vector insert NAP");

    // fo.print("vector insert(" + TOS(get_knns_time) + "s, " + TOS(nap_time) +
    //          "s) : pivot_num=" + TOS(pivot_num));
    return get_knns_time + nap_time;
}

float GPUFuncs::vector_insert_base(const float* h_insert_vectors, const int* ins_ids, int batch) {
    fo.eprint("USELESS: vector_insert_base");
    // cudaStream_t stream = streams["upd"].stream, cpy_stream = streams["upd_cpy"].stream;

    // auto s = std::chrono::high_resolution_clock::now();
    // int *h_knns, *h_upd_nbrs, *h_new_nbrs;
    // const int pivot_num = obtain_paired_knns(h_insert_vectors, ins_ids, h_knns, batch);
    // auto s1 = std::chrono::high_resolution_clock::now();
    // const float get_knns_time =
    //     std::chrono::duration_cast<std::chrono::milliseconds>(s1 - s).count() / 1000.0f;

    // // CUDA_CHECK(cudaMallocHost(&h_knns, sizeof(int) * pivot_num));
    // CUDA_CHECK(cudaMallocHost(&h_upd_nbrs, sizeof(int) * pivot_num * 2));
    // CUDA_CHECK(cudaMallocHost(&h_new_nbrs, sizeof(int) * batch * bp.max_degree));

    // auto e = gpu_record_time_start(stream);
    // CUDA_CHECK(cudaMemcpyAsync(d_ins_ids.data().get(), ins_ids, batch * sizeof(int),
    //                            cudaMemcpyHostToDevice, stream));
    // // CUDA_CHECK(cudaMemcpyAsync(h_knns, d_paired_knns.data().get(), pivot_num * sizeof(int),
    // //                            cudaMemcpyDeviceToHost, cpy_stream));

    // // thrust::counting_iterator<int> count_iter(0);
    // // thrust::transform(thrust::cuda::par.on(stream), count_iter, count_iter + batch + 1,
    // //                   d_knn_offsets.begin(), OffsetCalculator(bp.k));

    // const float* vec_slots = graph->get_gpu_vec_slots();
    // Slot* slots = graph->get_gpu_slots();
    // const int tpb = 512, cand_size = bp.max_degree + 1, ins_cand_size = bp.k;
    // k_plus_ones_build_kernel<<<batch, tpb, 0, stream>>>(
    //     vec_slots, slots, d_ins_ids.data().get(), d_paired_knns.data().get(),
    //     d_knn_offsets.data().get(), d_new_nbr_ids.data().get(), cand_nbrs.data().get(),
    //     ins_cand_nbrs.data().get(), pivot_num, cand_size, ins_cand_size, batch, bp);
    // CUDA_CHECK(cudaGetLastError());

    // const int total_cand_size = pivot_num * cand_size, blocks = (pivot_num + tpb - 1) / tpb,
    //           blocks1 = (batch + tpb - 1) / tpb;
    // // fo.print("Insert: pivot_num = " + TOS(pivot_num));

    // dist_extract_from_CN(cand_nbrs.data().get(), cand_dists.data().get(), stream,
    // total_cand_size,
    //                      bp);
    // auto res = segmented_sort_pairs_nocpy(
    //     cand_nbrs.data().get(), cand_dists.data().get(), cand_nbrs_sort.data().get(),
    //     cand_dists_sort.data().get(), offsets.data().get(), stream, pivot_num, cand_size);

    // candidate_ignore_kernel<<<blocks, tpb, 0, stream>>>(vec_slots, res.first, pivot_num,
    // cand_size,
    //                                                     bp, 0.0);
    // CUDA_CHECK(cudaGetLastError());

    // thrust::fill(thrust::cuda::par.on(stream), d_updated.begin(), d_updated.end(), 0);
    // // int *d_error, h_error;
    // // cudaMallocAsync(&d_error, sizeof(int), stream);
    // // cudaMemsetAsync(d_error, 0, sizeof(int), stream);
    // update_new_nbrs_kernel<<<blocks, tpb, 0, stream>>>(
    //     d_paired_knns.data().get(), d_new_nbr_ids.data().get(), res.first, slots,
    //     d_upd_nbrs.data().get(), d_updated.data().get(), d_total_degree, d_noniso_num,
    //     pivot_num, cand_size, bp);  // 更新新邻居
    // CUDA_CHECK(cudaGetLastError());
    // // cudaMemcpyAsync(&h_error, d_error, sizeof(int), cudaMemcpyDeviceToHost, stream);
    // // cudaStreamSynchronize(stream);
    // // if (h_error > 0) fo.eprint("Insert: update error(" + TOS(h_error) + ")!");

    // CUDA_CHECK(cudaEventRecord(upd_event, stream));

    // CUDA_CHECK(cudaStreamWaitEvent(cpy_stream, upd_event));
    // CUDA_CHECK(cudaMemcpyAsync(h_upd_nbrs, d_upd_nbrs.data().get(), sizeof(int) * pivot_num * 2,
    //                            cudaMemcpyDeviceToHost, cpy_stream));
    // CUDA_CHECK(cudaEventRecord(cpy_event, cpy_stream));

    // // std::async(std::launch::async, [this, pivot_num]() {
    // //     CUDA_CHECK(cudaEventSynchronize(cpy_event));
    // //     graph->update_innbrs_with_upd_nbrs(paired_query_knns.data(), h_upd_nbrs, pivot_num);
    // // });
    // executor.submit([this, h_knns, h_upd_nbrs, pivot_num]() {
    //     CUDA_CHECK(cudaEventSynchronize(cpy_event));
    //     graph->update_innbrs_with_upd_nbrs(h_knns, h_upd_nbrs, pivot_num);
    //     CUDA_CHECK(cudaFreeHost(h_knns));
    //     CUDA_CHECK(cudaFreeHost(h_upd_nbrs));
    // });

    // dist_extract_from_CN(ins_cand_nbrs.data().get(), cand_dists.data().get(), stream,
    //                      ins_cand_size * batch, bp);
    // auto res1 = segmented_sort_pairs_nocpy(
    //     ins_cand_nbrs.data().get(), cand_dists.data().get(), ins_cand_nbrs_sort.data().get(),
    //     cand_dists_sort.data().get(), offsets.data().get(), stream, batch, ins_cand_size);

    // candidate_ignore_kernel_for_ins<<<batch, tpb, 0, stream>>>(
    //     vec_slots, res1.first, d_ins_ids.data().get(), batch, ins_cand_size, bp);
    // CUDA_CHECK(cudaGetLastError());

    // CUDA_CHECK(cudaStreamWaitEvent(stream, cpy_event));  // 等待 d_upd_nbrs 数据拷贝完成再复用
    // update_and_collect_nbrs_kernel<<<blocks1, tpb, 0, stream>>>(
    //     d_ins_ids.data().get(), res1.first, slots, nullptr, d_upd_nbrs.data().get(),
    //     d_updated.data().get(), d_total_degree, d_noniso_num, batch, ins_cand_size, bp);
    // CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaMemcpyAsync(h_new_nbrs, d_upd_nbrs.data().get(),
    //                            sizeof(int) * batch * bp.max_degree, cudaMemcpyDeviceToHost,
    //                            stream));

    // // std::async(std::launch::async, [this, stream, ins_ids, batch]() {
    // //     CUDA_CHECK(cudaStreamSynchronize(stream));
    // //     graph->update_innbrs_with_new_nbrs(ins_ids, h_new_nbrs, batch);
    // // });
    // executor.submit([this, stream, ins_ids, h_new_nbrs, batch]() {
    //     CUDA_CHECK(cudaStreamSynchronize(stream));
    //     graph->update_innbrs_with_new_nbrs(ins_ids, h_new_nbrs, batch);
    //     CUDA_CHECK(cudaFreeHost(h_new_nbrs));
    // });

    // const float nap_time = gpu_record_time_stop(e, stream, "vector insert NAP");

    // fo.print("vector insert(" + TOS(get_knns_time) + "s, " + TOS(nap_time) +
    //          "s) : pivot_num=" + TOS(pivot_num));
    // return get_knns_time + nap_time;
}

void GPUFuncs::update_prepare() {
    cudaStream_t& stream = streams["upd"].stream;
    query_index->set_gpu(stream);

    d_del_ids.resize(INNBR_BATCH / 32);
    d_del_innbr_ids.resize(INNBR_BATCH);
    d_del_innbr_offsets.resize(INNBR_BATCH / 32 + 1);
    const int cand_size = 2 * bp.max_degree, total_cand_size = INNBR_BATCH * cand_size;
    del_offsets.resize(INNBR_BATCH + 1);
    del_cand_dists.resize(total_cand_size);
    del_cand_dists_sort.resize(total_cand_size);
    del_cand_nbrs.resize(total_cand_size);
    del_cand_nbrs_sort.resize(total_cand_size);

    del_hash_cap = (INNBR_BATCH / 32) << 2;
    d_del_hashset.resize(del_hash_cap);
    d_deleted.assign((bp.base_n * 2 + 31) / 32, 0);
    // CUDA_CHECK(cudaMallocAsync(&d_innbr_num, sizeof(int), stream));
    // CUDA_CHECK(cudaMallocHost(&h_innbr_num, sizeof(int)));

    const int max_pivot_num =
                  PC.insert_batch * (PC.upd_type == 0 || PC.upd_type == 1
                                         ? bp.k
                                         : (PC.upd_type == 2 ? bp.k * PC.insert_query_k
                                                             : bp.k * (PC.insert_query_k + 1))),
              cand_size1 = bp.max_degree + 1, total_size = cand_size1 * max_pivot_num;
    d_ins_ids.resize(PC.insert_batch);
    d_knn_offsets.resize(PC.insert_batch + 1);
    d_knn_offsets1.resize(PC.insert_batch + 1);
    h_paired_query_ids.resize(PC.insert_batch * PC.insert_query_k);
    d_new_nbr_ids.resize(max_pivot_num);
    d_upd_nbrs.resize(max_pivot_num * 2);
    // CUDA_CHECK(cudaMallocHost(&h_upd_nbrs, sizeof(int) * max_pivot_num * 2));
    // CUDA_CHECK(cudaMallocHost(&h_new_nbrs, sizeof(int) * PC.insert_batch * bp.max_degree));
    // cand_nbrs.resize(total_size);
    // cand_nbrs_sort.resize(total_size);

    ins_cand_nbrs.resize(max_pivot_num);
    ins_cand_nbrs_sort.resize(max_pivot_num);
    cand_dists.resize(max_pivot_num);
    cand_dists_sort.resize(max_pivot_num);
    offsets.resize(PC.insert_batch + 1);

    d_updated.resize((bp.base_n * 2 + 31) / 32);

    cudaMallocAsync(&d_ins_vecs, PC.insert_batch * bp.dim * sizeof(float), stream);

    cudaEventCreate(&del_event);
    cudaEventCreate(&cpy_event);
    cudaEventCreate(&upd_event);
}

void GPUFuncs::update_free() {
    cudaStream_t& stream = streams["upd"].stream;
    if (d_ins_vecs != nullptr) CUDA_CHECK(cudaFreeAsync(d_ins_vecs, stream));
    // CUDA_CHECK(cudaFreeHost(h_innbr_num));
    // CUDA_CHECK(cudaFreeHost(h_upd_nbrs));
    // CUDA_CHECK(cudaFreeHost(h_new_nbrs));
    // CUDA_CHECK(cudaFreeHost(h_del_old_nbrs));
    // CUDA_CHECK(cudaFreeHost(h_del_new_nbrs));
    // CUDA_CHECK(cudaFreeHost(h_del_new_degs));

    cudaEventSynchronize(del_event);
    cudaEventSynchronize(cpy_event);
    cudaEventDestroy(del_event);
    cudaEventDestroy(cpy_event);
    cudaEventDestroy(upd_event);
}

// void GPUFuncs::graph_resize(int new_size) {
//     base_n_upd.fetch_add(new_size);

//     int base_n = base_n_upd.load();
//     graph.resize(base_n * bp.max_degree);  // 扩展 graph
//     graph_dist.resize(base_n * bp.max_degree);
//     graph_deg.resize(base_n);
//     from_edge_nums.resize(base_n);

//     fo.print("graph resize to " + TOS(base_n) + " with new " + TOS(new_size) + " vectors");
// }

};  // namespace efanna2e

// __global__ void find_closest_paired_query_kernel(const int* __restrict__ knns,
//                                                  const int* __restrict__ base_query_ids,
//                                                  const float* __restrict__ queries,
//                                                  const float* __restrict__ insert_vectors,
//                                                  int* __restrict__ paired_query_ids_res, int
//                                                  batch, BP bp) {
//     const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
//     extern __shared__ unsigned char smem[];
//     int* sh_paired_qid = (int*)smem;
//     float* sh_paired_query_dist = (float*)(sh_paired_qid + 1);
//     for (int ins_idx = bid; ins_idx < batch; ins_idx += gridDim.x) {
//         if (!tid) *sh_paired_query_dist = farthest_dist_d(bp);
//         __syncthreads();

//         for (int i = tid; i < bp.k; i += tpb) {
//             const int qid = base_query_ids[knns[ins_idx * bp.k + i]];
//             const float* query = queries + qid * bp.dim;
//             const float dist = calc_distance(query, insert_vectors + ins_idx * bp.dim,
//                                              bp);  // 插入与查询向量的距离
//             atomicClosestFloat(sh_paired_query_dist, dist, bp);
//             __threadfence();  // 刷写全局/共享内存，使其他 block 可见
//             if (*sh_paired_query_dist - dist < EPS) sh_paired_qid[i] = qid;
//         }
//         __syncthreads();

//         if (!tid) paired_query_ids_res[ins_idx] = *sh_paired_qid;
//     }
// }

// __global__ void base_insert_kernel(const float* __restrict__ insert_vectors,
//                                    float* __restrict__ base, int batch, int base_n, BP bp) {
//     const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
//     for (int ins_idx = bid; ins_idx < batch; ins_idx += gridDim.x)
//         for (int d = tid; d < bp.dim; d += tpb)
//             base[(base_n + ins_idx) * bp.dim + d] = insert_vectors[ins_idx * bp.dim + d];
// }
