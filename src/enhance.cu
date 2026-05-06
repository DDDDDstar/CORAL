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

__global__ void check_isolated_kernel(const Slot* __restrict__ slots, int* __restrict__ is_iso,
                                      int N) {
    const int tpb = blockDim.x;
    for (int i = blockIdx.x * tpb + threadIdx.x; i < N; i += gridDim.x * tpb)
        is_iso[i] = slots[i].deg == 0 ? 1 : 0;
}

__global__ void load_vecs_kernel(const float* __restrict__ vecs, const int* __restrict__ ids,
                                 float* __restrict__ vecs_out, int N, int dim) {
    const int tpb = blockDim.x;
    for (int i = blockIdx.x * tpb + threadIdx.x; i < N; i += gridDim.x * tpb) {
        const float* in = vecs + (size_t)ids[i] * dim;
        float* out = vecs_out + (size_t)i * dim;
        for (int j = 0; j < dim; ++j) out[j] = in[j];
    }
}

__global__ void update_nbrs_kernel(Slot* __restrict__ slots, const int* __restrict__ slot_idxs,
                                   const int* __restrict__ new_nbrs,
                                   const float* __restrict__ nbr_dists, int* delta_degree,
                                   int* delta_noniso_num, int N, int max_degree) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    for (int slot_i = bid * tpb + tid; slot_i < N; slot_i += gridDim.x * tpb) {
        const int slot_idx = slot_idxs[slot_i];
        Slot* slot = slots + slot_idx;
        const int* nbrs = new_nbrs + slot_i * max_degree;
        const float* dists = nbr_dists + slot_i * max_degree;

        if (slot->deg == 0) atomicAdd(delta_noniso_num, 1);
        atomicAdd(delta_degree, max_degree - slot->deg);
        slot->deg = max_degree;
        for (int nbr_i = 0; nbr_i < max_degree; nbr_i++) {
            slot->nbrs[nbr_i] = nbrs[nbr_i];
            slot->dists[nbr_i] = dists[nbr_i];
        }
    }
}
__device__ __forceinline__ int atomicTestAndSetBit(uint32_t* bitset, int bit_index) {
    const uint32_t mask = 1u << (bit_index & 31);
    uint32_t* addr = bitset + (bit_index >> 5);
    const uint32_t old = atomicOr(addr, mask);
    return (old & mask) != 0;
}

__global__ void build_candidate_kernel(const int* __restrict__ top_ids,
                                       const int* __restrict__ knn_ids,
                                       const float* __restrict__ knn_dists,
                                       const float* __restrict__ vecs, Slot* __restrict__ slots,
                                       CN* __restrict__ cand_nbrs, float* __restrict__ cand_dists,
                                       uint32_t* __restrict__ updated, int iso_num, BP bp) {
    const int top_i = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x,
              cand_size = bp.k + bp.max_degree;
    if (top_i >= iso_num) return;

    __shared__ int s_top_id;
    if (tid == 0) {
        s_top_id = top_ids[top_i];
        assert(s_top_id >= 0);
    }
    __syncthreads();

    const int off = top_i * cand_size, *knns = knn_ids + top_i * bp.k;
    CN* cands = cand_nbrs + off;
    float* cand_ds = cand_dists + off;
    const float* dists = knn_dists + top_i * bp.k;
    const Slot& top_slot = slots[s_top_id];
    for (int i = tid; i < cand_size; i += tpb) {
        if (i < bp.k) {
            const int knn_id = knns[i];
            assert(knn_id >= 0);
            const float knn_dist = dists[i];
            cands[i] = {knn_id, knn_id, Initial, knn_dist};
            cand_ds[i] = knn_dist;

            if (atomicTestAndSetBit(updated, knn_id)) continue;
            Slot& slot = slots[knn_id];
            const int deg = slot.deg;
            int pos = 0, valid = 1, out_pos;
            for (; pos < deg; ++pos)
                if (compare_dist(knn_dist, slot.dists[pos], bp)) {
                    if (valid &&
                        compare_dist(knn_dist, calc_distance(vecs, slot.nbrs[pos], s_top_id, bp),
                                     bp))
                        valid = 0;
                } else
                    break;

            if (deg == bp.max_degree) {
                if (!valid || pos >= deg) continue;
                out_pos = deg - 1;
                for (; out_pos >= pos; out_pos--)
                    if (compare_dist(slot.dists[out_pos],
                                     calc_distance(vecs, slot.nbrs[out_pos], s_top_id, bp), bp))
                        break;
                if (out_pos < pos) {
                    out_pos = deg - 1;
                }
            } else {
                out_pos = deg;
                slot.deg = deg + 1;
            }

            for (int j = out_pos; j > pos; --j) {
                slot.nbrs[j] = slot.nbrs[j - 1];
                slot.dists[j] = slot.dists[j - 1];
            }
            slot.nbrs[pos] = s_top_id;
            slot.dists[pos] = knn_dist;
        } else {
            const int nbr_i = i - bp.k, nbr_id = top_slot.nbrs[nbr_i];
            const float dist = top_slot.dists[nbr_i];
            cands[i] = {nbr_id, nbr_id, Initial, dist};
            cand_ds[i] = dist;
        }
    }
}

__global__ void build_candidate_kernel(
    const int* __restrict__ iso_ids, const int* __restrict__ knn_ids,
    const float* __restrict__ knn_dists, const float* __restrict__ vecs, Slot* __restrict__ slots,
    CN* __restrict__ iso_cand_nbrs, float* __restrict__ iso_cand_dists,
    uint32_t* __restrict__ updated, uint32_t* __restrict__ cnt, int iso_num, int knn_num, BP bp) {
    const int iso_i = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    if (iso_i >= iso_num) return;

    extern __shared__ float s_iso_vec[];  // [dim]
    __shared__ int s_iso_id;
    if (tid == 0) {
        s_iso_id = iso_ids[iso_i];
        assert(s_iso_id >= 0 && s_iso_id < bp.base_n);
    }
    __syncthreads();
    const float* iso_vec = vecs + (size_t)s_iso_id * bp.dim;
    for (int d = tid; d < bp.dim; d += tpb) s_iso_vec[d] = iso_vec[d];
    __syncthreads();

    const int off = iso_i * knn_num, *knns = knn_ids + off;
    CN* iso_cands = iso_cand_nbrs + off;
    float* cand_dists = iso_cand_dists + off;
    const float* dists = knn_dists + off;
    for (int i = tid; i < knn_num; i += tpb) {
        const int knn_id = knns[i];
        assert(knn_id >= 0 && knn_id < bp.base_n);
        const float knn_dist = dists[i];
        iso_cands[i] = {knn_id, knn_id, Initial, knn_dist};
        cand_dists[i] = knn_dist;

        if (atomicTestAndSetBit(updated, knn_id)) continue;
#ifdef VERBOSE
        atomicAdd(cnt, 1);
#endif
        Slot& slot = slots[knn_id];
        const int deg = slot.deg;
        int pos = 0, valid = 1, out_pos;
        for (; pos < deg; ++pos)
            if (compare_dist(knn_dist, slot.dists[pos], bp)) {
                if (valid &&
                    compare_dist(knn_dist, calc_distance(s_iso_vec, vecs, slot.nbrs[pos], bp), bp))
                    valid = 0;
            } else
                break;

        if (deg == bp.max_degree) {
            if (!valid || pos >= deg) continue;
            out_pos = deg - 1;
            for (; out_pos >= pos; out_pos--)
                if (compare_dist(slot.dists[out_pos],
                                 calc_distance(s_iso_vec, vecs, slot.nbrs[out_pos], bp), bp))
                    break;
            if (out_pos < pos) {
#ifdef VERBOSE
                atomicAdd(cnt + 1, 1);
#endif
                out_pos = deg - 1;
            }
#ifdef VERBOSE
            else
                atomicAdd(cnt + 2, 1);
#endif
        } else {
            out_pos = deg;
            slot.deg = deg + 1;
#ifdef VERBOSE
            atomicAdd(cnt + 3, 1);
#endif
        }

        for (int j = out_pos; j > pos; --j) {
            slot.nbrs[j] = slot.nbrs[j - 1];
            slot.dists[j] = slot.dists[j - 1];
        }
        slot.nbrs[pos] = s_iso_id;
        slot.dists[pos] = knn_dist;
    }
}

__global__ void build_candidate_kernel(
    const int* __restrict__ iso_ids, const int* __restrict__ knn_ids,
    const float* __restrict__ knn_dists, const size_t* __restrict__ knn_offsets,
    const float* __restrict__ vecs, Slot* __restrict__ slots, CN* __restrict__ iso_cand_nbrs,
    float* __restrict__ iso_cand_dists, uint32_t* __restrict__ updated, uint32_t* __restrict__ cnt,
    int iso_num, BP bp, int need_discard) {
    const int iso_i = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    if (iso_i >= iso_num) return;

    extern __shared__ float s_iso_vec[];  // [dim]
    __shared__ int s_iso_id, s, e;
    if (tid == 0) {
        s_iso_id = iso_ids[iso_i];
        assert(s_iso_id >= 0);
        s = (int)knn_offsets[iso_i];
        e = (int)knn_offsets[iso_i + 1];
    }
    __syncthreads();
    const float* iso_vec = vecs + (size_t)s_iso_id * bp.dim;
    for (int d = tid; d < bp.dim; d += tpb) s_iso_vec[d] = iso_vec[d];
    __syncthreads();

    for (int i = tid + s; i < e; i += tpb) {
        const int knn_id = knn_ids[i];
        assert(knn_id >= 0);
        const float knn_dist = knn_dists[i];
        iso_cand_nbrs[i] = {knn_id, knn_id, Initial, knn_dist};
        iso_cand_dists[i] = knn_dist;

        if (atomicTestAndSetBit(updated, knn_id)) continue;
#ifdef VERBOSE
        atomicAdd(cnt, 1);
#endif

        Slot& slot = slots[knn_id];
        const int deg = slot.deg;
        int pos = 0, valid = 1, out_pos;
        if (need_discard) {
            for (; pos < deg; ++pos)
                if (compare_dist(knn_dist, slot.dists[pos], bp)) {
                    if (valid &&
                        compare_dist(knn_dist, calc_distance(s_iso_vec, vecs, slot.nbrs[pos], bp),
                                     bp))
                        valid = 0;
                } else
                    break;

            if (deg == bp.max_degree) {
                if (!valid || pos >= deg) continue;
                out_pos = deg - 1;
                for (; out_pos >= pos; out_pos--)
                    if (compare_dist(slot.dists[out_pos],
                                     calc_distance(s_iso_vec, vecs, slot.nbrs[out_pos], bp), bp))
                        break;
                if (out_pos < pos) {
#ifdef VERBOSE
                    atomicAdd(cnt + 1, 1);
#endif
                    out_pos = deg - 1;
                }
#ifdef VERBOSE
                else
                    atomicAdd(cnt + 2, 1);
#endif
            } else {
                out_pos = deg;
                slot.deg = deg + 1;
#ifdef VERBOSE
                atomicAdd(cnt + 3, 1);
#endif
            }

            for (int j = out_pos; j > pos; --j) {
                slot.nbrs[j] = slot.nbrs[j - 1];
                slot.dists[j] = slot.dists[j - 1];
            }
            slot.nbrs[pos] = s_iso_id;
            slot.dists[pos] = knn_dist;
        } else {
            for (; pos < deg; ++pos)
                if (compare_dist(knn_dist, slot.dists[pos], bp)) {
                    if (compare_dist(knn_dist, calc_distance(s_iso_vec, vecs, slot.nbrs[pos], bp),
                                     bp)) {
                        valid = 0;
                        break;
                    }
                } else
                    break;
            if (!valid || pos >= bp.max_degree) continue;

            out_pos = deg - 1;
            for (; out_pos >= pos; out_pos--)
                if (compare_dist(slot.dists[out_pos],
                                 calc_distance(s_iso_vec, vecs, slot.nbrs[out_pos], bp), bp))
                    break;

            if (out_pos < pos) {
                if (deg == bp.max_degree) {
                    out_pos = deg - 1;
#ifdef VERBOSE
                    atomicAdd(cnt + 1, 1);
#endif
                } else {
                    out_pos = deg;
                    slot.deg = deg + 1;
#ifdef VERBOSE
                    atomicAdd(cnt + 2, 1);
#endif
                }
            }
#ifdef VERBOSE
            else
                atomicAdd(cnt + 3, 1);
#endif
            for (int j = out_pos; j > pos; --j) {
                slot.nbrs[j] = slot.nbrs[j - 1];
                slot.dists[j] = slot.dists[j - 1];
            }
            slot.nbrs[pos] = s_iso_id;
            slot.dists[pos] = knn_dist;
        }
    }
}

__global__ void candidate_ignore_kernel_for_iso(const float* __restrict__ vecs,
                                                CN* __restrict__ cand_nbrs, const int* pivot_ids,
                                                int pivot_num, int cand_size, BP bp) {
    const int pivot_i = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    if (pivot_i >= pivot_num) return;

    __shared__ int nbr_num;
    if (!tid) nbr_num = 0;
    __syncthreads();

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
            const float* vec = vecs + (size_t)nbr_id * bp.dim;
            for (int j = i + 1 + tid; j < cand_size; j += tpb) {
                CN& nbr1 = cands[j];
                if (nbr1.status == Invalid) break;
                const int nbr1_id = nbr1.id;
                if (nbr1_id == nbr_id)
                    nbr1.status = Repeated;
                else if (nbr1.status == Initial) {
                    const float* vec1 = vecs + (size_t)nbr1_id * bp.dim;
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

__global__ void candidate_ignore_kernel_for_iso(const float* __restrict__ vecs,
                                                CN* __restrict__ cand_nbrs, const int* pivot_ids,
                                                const size_t* offsets, int pivot_num, BP bp) {
    const int pivot_i = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    if (pivot_i >= pivot_num) return;

    __shared__ int nbr_num, s, e;
    if (!tid) {
        nbr_num = 0;
        s = (int)offsets[pivot_i];
        e = (int)offsets[pivot_i + 1];
    }
    __syncthreads();

    for (int i = s; i < e && nbr_num < bp.max_degree; i++) {
        CN& nbr = cand_nbrs[i];
        if (nbr.status == Invalid) break;
        const int nbr_id = nbr.id;
        assert(nbr_id >= 0);
        const auto status = nbr.status;
        if (status == Initial) {
            if (!tid) {
                nbr.status = Retained;
                nbr_num++;
            }
            const float* vec = vecs + (size_t)nbr_id * bp.dim;
            for (int j = i + 1 + tid; j < e; j += tpb) {
                CN& nbr1 = cand_nbrs[j];
                if (nbr1.status == Invalid) break;
                const int nbr1_id = nbr1.id;
                if (nbr1_id == nbr_id)
                    nbr1.status = Repeated;
                else if (nbr1.status == Initial) {
                    const float* vec1 = vecs + (size_t)nbr1_id * bp.dim;
                    const float pj_dist = nbr1.dist;
                    if (compare_dist(pj_dist, calc_distance(vec, vec1, bp), bp))
                        nbr1.status = Discarded;
                }
            }
        } else if (status == Discarded)
            for (int j = i + 1 + tid; j < e; j += tpb) {
                CN& nbr1 = cand_nbrs[j];
                if (nbr_id == nbr1.id)
                    nbr1.status = Repeated;
                else if (nbr1.status == Invalid)
                    break;
            }
        __syncthreads();
    }
    // assert(nbr_num > 0);
    if (!tid && nbr_num == 0) printf("nbr_num == 0, s=%d, e=%d\n", s, e);
}

__global__ void cal_knn_dists_kernel(const float* __restrict__ vecs,
                                     const float* __restrict__ queries,
                                     const int* __restrict__ knns,
                                     const size_t* __restrict__ offsets,
                                     float* __restrict__ dists_out, int n, BP bp) {
    const int bid = blockIdx.x, tpb = blockDim.x, tid = threadIdx.x;

    extern __shared__ float s_queries[];
    __shared__ int s, e;
    for (int i = bid; i < n; i += gridDim.x) {
        const float* query = queries + (size_t)i * bp.dim;
        for (int j = tid; j < bp.dim; j += tpb) s_queries[j] = query[j];
        if (!tid) s = offsets[i];
        if (tid == 1) e = offsets[i + 1];
        __syncthreads();
        for (int j = s + tid; j < e; j += tpb) {
            dists_out[j] = calc_distance(s_queries, vecs + (size_t)knns[j] * bp.dim, bp);
        }
    }
}

__global__ void update_iso_nbrs_kernel(const int* __restrict__ pivot_ids,
                                       const size_t* __restrict__ offsets,
                                       Slot* __restrict__ slots, CN* __restrict__ cand_nbrs,
                                       ull* __restrict__ total_degree,
                                       ull* __restrict__ noniso_num, int pivot_num, BP bp) {
    const int tid = threadIdx.x, pivot_i = blockIdx.x * blockDim.x + tid;
    __shared__ int delta_deg, delta_noniso_num;
    if (!tid) delta_deg = delta_noniso_num = 0;
    __syncthreads();

    if (pivot_i < pivot_num) {
        const int pivot_id = pivot_ids[pivot_i];
        // assert(pivot_id >= 0);
        int nbr_num = 0, last_id = -1;
        Slot& slot = slots[pivot_id];
        int retained = 0;
        const int s = offsets[pivot_i], e = offsets[pivot_i + 1];
        for (int i = s; i < e && nbr_num < bp.max_degree; i++) {
            const CN& cand_nbr = cand_nbrs[i];
            const auto status = cand_nbr.status;
            if (status == Retained) {
                retained++;
            } else if (status == Invalid)
                break;
        }
        assert(retained > 0);
        // if (!retained) printf("ERROR: no retained nbrs\n");
        int discarded_need = bp.max_degree - retained;
        for (int j = s; j < e && nbr_num < bp.max_degree; j++) {
            const CN& cand_nbr = cand_nbrs[j];
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
        assert(nbr_num > 0);
        // if (nbr_num == 0) printf("ERROR: nbr_num==0\n");
        const int old_deg = slot.deg;
        if (!old_deg) {
            atomicAdd(&delta_noniso_num, 1);
            atomicAdd(&delta_deg, nbr_num);
        } else
            atomicAdd(&delta_deg, nbr_num - old_deg);
        slot.deg = nbr_num;
    }
    __syncthreads();
    if (!tid) {
        atomicAdd(total_degree, (ull)delta_deg);
        atomicAdd(noniso_num, (ull)delta_noniso_num);
    }
}

__global__ void update_iso_nbrs_kernel(const int* __restrict__ pivot_ids, Slot* __restrict__ slots,
                                       CN* __restrict__ cand_nbrs, ull* __restrict__ total_degree,
                                       ull* __restrict__ noniso_num, int pivot_num, int cand_size,
                                       BP bp) {
    const int tid = threadIdx.x, pivot_i = blockIdx.x * blockDim.x + tid;
    __shared__ int delta_deg, delta_noniso_num;
    if (!tid) delta_deg = delta_noniso_num = 0;
    __syncthreads();

    if (pivot_i < pivot_num) {
        const int pivot_id = pivot_ids[pivot_i];
        assert(pivot_id >= 0);
        int nbr_num = 0, last_id = -1;
        Slot& slot = slots[pivot_id];
        const CN* cands = cand_nbrs + pivot_i * cand_size;
        int retained = 0;
        for (int i = 0; i < cand_size && nbr_num < bp.max_degree; i++) {
            const auto status = cands[i].status;
            if (status == Retained) {
                retained++;
            } else if (status == Invalid)
                break;
        }
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
        assert(nbr_num > 0);
        const int old_deg = slot.deg;
        if (!old_deg) {
            atomicAdd(&delta_noniso_num, 1);
            atomicAdd(&delta_deg, nbr_num);
        } else
            atomicAdd(&delta_deg, nbr_num - old_deg);
        slot.deg = nbr_num;
    }
    __syncthreads();
    if (!tid) {
        atomicAdd(total_degree, (ull)delta_deg);
        atomicAdd(noniso_num, (ull)delta_noniso_num);
    }
}

int Graph::load_isolated_nodes(std::vector<int>& iso_ids, int*& d_iso_ids, float*& d_iso_vecs,
                               bool need_vec, cudaStream_t stream) {
    assert(gtype == GraphType::GPU);

    thrust::device_vector<int> d_is_iso(total_node_size);
    int* h_is_iso;
    int tpb = 256, blocks = std::min(1024, (total_node_size + tpb - 1) / tpb);
    check_isolated_kernel<<<blocks, tpb, 0, stream>>>(gpu_slots, d_is_iso.data().get(),
                                                      total_node_size);

    CUDA_CHECK(cudaMallocHost(&h_is_iso, total_node_size * sizeof(int)));
    CUDA_CHECK(cudaMemcpyAsync(h_is_iso, d_is_iso.data().get(), total_node_size * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    iso_ids.resize(total_node_size);
    std::iota(iso_ids.begin(), iso_ids.end(), 0);
    iso_ids.erase(
        std::remove_if(iso_ids.begin(), iso_ids.end(), [&](int id) { return !h_is_iso[id]; }),
        iso_ids.end());
    const int iso_num = iso_ids.size();
    CUDA_CHECK(cudaFreeHost(h_is_iso));
    if (!need_vec) return iso_num;
    if (iso_num > 0) {
        cudaMallocAsync(&d_iso_ids, (size_t)iso_num * sizeof(int), stream);
        cudaMallocAsync(&d_iso_vecs, (size_t)iso_num * dim * sizeof(float), stream);
        CUDA_CHECK(cudaMemcpyAsync(d_iso_ids, iso_ids.data(), iso_num * sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
        blocks = std::min(1024, (iso_num + tpb - 1) / tpb);
        load_vecs_kernel<<<blocks, tpb, 0, stream>>>(gpu_vec_slots, d_iso_ids, d_iso_vecs, iso_num,
                                                     dim);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return iso_num;
}

int Graph::connect_enhance(const int* d_iso_ids, const int* d_new_nbrs,
                           const float* d_new_nbr_dists, int N, cudaStream_t stream, BP bp) {
    thrust::device_vector<int> d_delta_degree(1, 0), d_delta_noniso_num(1, 0);
    int delta_degree, delta_noniso_num;
    const int tpb = 256, blocks = (N + tpb - 1) / tpb;
    update_nbrs_kernel<<<blocks, tpb, 0, stream>>>(gpu_slots, d_iso_ids, d_new_nbrs,
                                                   d_new_nbr_dists, d_delta_degree.data().get(),
                                                   d_delta_noniso_num.data().get(), N, max_degree);
    CUDA_CHECK(cudaMemcpyAsync(&delta_degree, d_delta_degree.data().get(), sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&delta_noniso_num, d_delta_noniso_num.data().get(), sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    total_degree += delta_degree;
    noniso_num += delta_noniso_num;
    fo.print("graph connect enhance: total_degree-" + TOS(total_degree) + ", noniso_num-" +
             TOS(noniso_num));
    return total_degree;
}

void GPUFuncs::enhance_neibor_aware_gpu(const int* d_top_ids, const int* d_search_nbr_ids,
                                        const float* d_search_nbr_dists, int batch,
                                        cudaStream_t stream) {
    const int tpb = 256, cand_size = bp.k + bp.max_degree;
    Slot* slots = graph->get_gpu_slots();
    const float* vecs = graph->get_gpu_vec_slots();
    thrust::fill(d_enhance_updated.begin(), d_enhance_updated.end(), 0);
    build_candidate_kernel<<<batch, tpb, 0, stream>>>(
        d_top_ids, d_search_nbr_ids, d_search_nbr_dists, vecs, slots, d_cand_nbrs.data().get(),
        d_cand_dists.data().get(), d_enhance_updated.data().get(), batch, bp);
    CUDA_CHECK(cudaGetLastError());

    auto res1 = segmented_sort_pairs_nocpy(d_cand_nbrs.data().get(), d_cand_dists.data().get(),
                                           d_cand_nbrs_sort.data().get(),
                                           d_cand_dists_sort.data().get(), d_offsets, stream,
                                           batch, cand_size, bp.metric == DIST_METRIC::IP_);

    candidate_ignore_kernel_for_iso<<<batch, tpb, 0, stream>>>(vecs, res1.first, d_top_ids, batch,
                                                               cand_size, bp);
    CUDA_CHECK(cudaGetLastError());

    update_iso_nbrs_kernel<<<(batch + tpb - 1) / tpb, tpb, 0, stream>>>(
        d_top_ids, slots, res1.first, d_total_degree, d_noniso_num, batch, cand_size, bp);
    CUDA_CHECK(cudaGetLastError());
}

void GPUFuncs::enhance_neibor_aware_gpu(const int* d_iso_ids, const int* d_search_nbr_ids,
                                        const float* d_search_nbr_dists, int batch, int knn_num,
                                        cudaStream_t stream) {
    const int tpb = 128;
    Slot* slots = graph->get_gpu_slots();
    const float* vecs = graph->get_gpu_vec_slots();
    thrust::fill(thrust::cuda::par.on(stream), d_enhance_updated.begin(), d_enhance_updated.end(),
                 0);
    thrust::device_vector<uint32_t> cnt(4, 0);
    std::vector<uint32_t> h_cnt(4, 0);
    build_candidate_kernel<<<batch, tpb, bp.dim * sizeof(float), stream>>>(
        d_iso_ids, d_search_nbr_ids, d_search_nbr_dists, vecs, slots, d_iso_cand_nbrs.data().get(),
        d_cand_dists.data().get(), d_enhance_updated.data().get(), cnt.data().get(), batch,
        knn_num, bp);
    CUDA_CHECK(cudaGetLastError());
    // cudaStreamSynchronize(stream);
    // cudaMemcpy(h_cnt.data(), cnt.data().get(), sizeof(uint32_t) * 4, cudaMemcpyDeviceToHost);
    auto res1 = segmented_sort_pairs_nocpy(d_iso_cand_nbrs.data().get(), d_cand_dists.data().get(),
                                           d_iso_cand_nbrs_sort.data().get(),
                                           d_cand_dists_sort.data().get(), d_offsets, stream,
                                           batch, knn_num, bp.metric == DIST_METRIC::IP_);

    candidate_ignore_kernel_for_iso<<<batch, tpb, 0, stream>>>(vecs, res1.first, d_iso_ids, batch,
                                                               knn_num, bp);
    CUDA_CHECK(cudaGetLastError());

    update_iso_nbrs_kernel<<<(batch + tpb - 1) / tpb, tpb, 0, stream>>>(
        d_iso_ids, slots, res1.first, d_total_degree, d_noniso_num, batch, knn_num, bp);
    CUDA_CHECK(cudaGetLastError());
}

void GPUFuncs::enhance_neibor_aware_gpu(const int* d_iso_ids, const int* d_search_nbr_ids,
                                        const float* d_search_nbr_dists, const size_t* d_offsets,
                                        int batch, int total_num, cudaStream_t stream) {
    const int tpb = 512;
    Slot* slots = graph->get_gpu_slots();
    const float* vecs = graph->get_gpu_vec_slots();
    thrust::fill(thrust::cuda::par.on(stream), d_enhance_updated.begin(), d_enhance_updated.end(),
                 0);
    thrust::device_vector<uint32_t> cnt(4, 0);
    std::vector<uint32_t> h_cnt(4, 0);
    build_candidate_kernel<<<batch, 128, 0, stream>>>(
        d_iso_ids, d_search_nbr_ids, d_search_nbr_dists, d_offsets, vecs, slots,
        d_iso_cand_nbrs.data().get(), d_cand_dists.data().get(), d_enhance_updated.data().get(),
        cnt.data().get(), batch, bp, PC.need_discard);
    CUDA_CHECK(cudaGetLastError());
#ifdef VERBOSE
    cudaStreamSynchronize(stream);
    cudaMemcpy(h_cnt.data(), cnt.data().get(), sizeof(uint32_t) * 4, cudaMemcpyDeviceToHost);
#endif
    // fo.iprint("build candidate: " + TOS(h_cnt[0]) + " " + TOS(h_cnt[1]) + " " + TOS(h_cnt[2]) +
    //           " " + TOS(h_cnt[3]));

    // dist_extract_from_CN(d_iso_cand_nbrs.data().get(), d_cand_dists.data().get(), stream,
    //                      total_num, bp);
    auto res1 = segmented_sort_pairs_nocpy_off(
        d_iso_cand_nbrs.data().get(), d_cand_dists.data().get(), d_iso_cand_nbrs_sort.data().get(),
        d_cand_dists_sort.data().get(), d_offsets, stream, batch, total_num);

    candidate_ignore_kernel_for_iso<<<batch, tpb, 0, stream>>>(vecs, res1.first, d_iso_ids,
                                                               d_offsets, batch, bp);
    CUDA_CHECK(cudaGetLastError());

    update_iso_nbrs_kernel<<<(batch + tpb - 1) / tpb, tpb, 0, stream>>>(
        d_iso_ids, d_offsets, slots, res1.first, d_total_degree, d_noniso_num, batch, bp);
    CUDA_CHECK(cudaGetLastError());
}

void GPUFuncs::gpu_connect_enhance() {
    assert(graph->Type() == GraphType::GPU);
    cudaStream_t stream = streams["search"].stream;
    if (PC.enhance_mode == 1) query_index->set_gpu(stream);
    auto e = gpu_record_time_start(stream);
    float search_time, total_search_time = 0, nap_time, total_nap_time = 0;
    // 1. load isolated nodes
    float load_iso_time;
    std::vector<int> iso_ids;
    int* d_iso_ids = nullptr;
    float* d_iso_vecs = nullptr;
    int iso_num = graph->load_isolated_nodes(iso_ids, d_iso_ids, d_iso_vecs, true, stream);
    cudaMemcpy(h_noniso_num, d_noniso_num, sizeof(ull), cudaMemcpyDeviceToHost);
    fo.iprint("load_isolated_nodes: isolated nodes: " + TOS(iso_num) +
              ", expected: " + TOS(bp.base_n - *h_noniso_num));

    e = gpu_record_time_reset(e, stream, load_iso_time, "load isolated nodes");

    if (PC.enhance_mode == 1) {
        const int iso_batch = PC.enhance_nap_batch;
        for (int i = 0; i < iso_num; i += iso_batch) {
            const int batch = std::min(iso_batch, iso_num - i);
            std::vector<int> knns;
            query_index->knn_Search_faiss(d_iso_vecs + i * bp.dim, iso_ids.data() + i, knns,
                                          h_offsets, batch);
            const int total_knn_num = knns.size();
            std::string str;
            for (int j = 0; j < bp.k * 10; j++) str += TOS(knns[j]) + " ";
            fo.print("search_and_nap: avg_knn_num=" + TOS(total_knn_num * 1.0 / batch) + "\n" +
                     str);
            d_search_res.assign(knns.begin(), knns.end());
            if (d_search_dist_res.size() < total_knn_num) {
                d_search_dist_res.resize(total_knn_num);
                d_iso_cand_nbrs.resize(total_knn_num);
                d_iso_cand_nbrs_sort.resize(total_knn_num);
                d_cand_dists.resize(total_knn_num);
                d_cand_dists_sort.resize(total_knn_num);
            }

            CUDA_CHECK(cudaMemcpyAsync(d_offsets, h_offsets, (batch + 1) * sizeof(size_t),
                                       cudaMemcpyHostToDevice, stream));

            cal_knn_dists_kernel<<<batch, 128, 0, stream>>>(
                graph->get_gpu_vec_slots(), d_iso_vecs + i * bp.dim, d_search_res.data().get(),
                d_offsets, d_search_dist_res.data().get(), batch, bp);
            cudaGetLastError();

            enhance_neibor_aware_gpu(d_iso_ids + i, d_search_res.data().get(),
                                     d_search_dist_res.data().get(), d_offsets, batch,
                                     total_knn_num, stream);

            fo.print("search_and_nap finished with i=" + TOS(i + batch) + "/" + TOS(iso_num));
        }
    } else if (PC.enhance_mode == 0) {
        const int search_batch = PC.enhance_search_batch, nap_batch = PC.enhance_nap_batch;
        search_prepare(search_batch, 256);
        d_search_res.resize(search_batch * bp.k);
        d_search_dist_res.resize(search_batch * bp.k);
        d_iso_cand_nbrs.resize(nap_batch * bp.k);
        d_iso_cand_nbrs_sort.resize(nap_batch * bp.k);
        d_cand_dists.resize(nap_batch * bp.k);
        d_cand_dists_sort.resize(nap_batch * bp.k);

        for (int i = 0; i < iso_num; i += search_batch) {
            const int batch = std::min(search_batch, iso_num - i);
            auto start_ids = graph->get_search_start_nodes_fps(PC.search_start_nodes_num, 1);
            search_gpu(d_iso_vecs + (size_t)i * bp.dim, d_search_res.data().get(),
                       d_search_dist_res.data().get(), batch, bp.k, start_ids, false, stream);
            for (int j = 0; j < batch; j += nap_batch) {
                const int batch1 = std::min(nap_batch, batch - j);
                enhance_neibor_aware_gpu(d_iso_ids + i + j, d_search_res.data().get() + j * bp.k,
                                         d_search_dist_res.data().get() + j * bp.k, batch1, bp.k,
                                         stream);
            }
            fo.print("search_and_nap finished with i=" + TOS(i + batch) + "/" + TOS(iso_num));
        }
    } else {
        iso_num = 0;
        search_prepare(std::min(PC.top_hit_num, PC.enhance_search_batch), 256);
    }
    cudaFreeAsync(d_iso_ids, stream);
    cudaFreeAsync(d_iso_vecs, stream);

    float iso_enhance_time;
    e = gpu_record_time_reset(e, stream, iso_enhance_time, "enhance isolated nodes");

    if (PC.test_search) test_search(graph->get_start_ids());

    // iso_num = graph->load_isolated_nodes(iso_ids, d_iso_ids, d_iso_vecs, false, stream);
    // fo.iprint("load_isolated_nodes test: isolated nodes: " + TOS(iso_num));

    if (PC.top_hit_num > iso_num) {
        const int search_batch = PC.enhance_search_batch, nap_batch = PC.enhance_nap_batch,
                  top_n = PC.top_hit_num - iso_num, tpb = 512;

        int* d_top_ids = nullptr;
        float* d_top_vecs = nullptr;
        cudaMalloc(&d_top_ids, top_n * sizeof(int));
        cudaMalloc(&d_top_vecs, (size_t)search_batch * bp.dim * sizeof(float));
        d_search_res.resize(search_batch * bp.k);
        d_search_dist_res.resize(search_batch * bp.k);
        const int cand_size = bp.k + bp.max_degree;
        d_cand_nbrs.resize(nap_batch * cand_size);
        d_cand_nbrs_sort.resize(nap_batch * cand_size);
        d_cand_dists.resize(nap_batch * cand_size);
        d_cand_dists_sort.resize(nap_batch * cand_size);
        get_top_hit_nodes(d_top_ids, top_n);
        for (int i = 0; i < top_n; i += search_batch) {
            const int search_n = std::min(search_batch, top_n - i);
            if (PC.test_search) search_prepare(search_n, 256);

            const int* d_top_ids_batch = d_top_ids + i;
            load_vecs_kernel<<<(search_n + tpb - 1) / tpb, tpb>>>(
                graph->get_gpu_vec_slots(), d_top_ids_batch, d_top_vecs, search_n, bp.dim);

            auto start_ids = graph->get_search_start_nodes_fps(PC.search_start_nodes_num, 1);
            search_gpu(d_top_vecs, d_search_res.data().get(), d_search_dist_res.data().get(),
                       search_n, bp.k, start_ids, false, stream);
            for (int j = 0; j < search_n; j += nap_batch) {
                const int batch = std::min(nap_batch, search_n - j);
                enhance_neibor_aware_gpu(d_top_ids_batch + j, d_search_res.data().get() + j * bp.k,
                                         d_search_dist_res.data().get() + j * bp.k, batch, stream);
            }
            fo.print("top enchance finished with i=" + TOS(i + search_n) + "/" + TOS(top_n));
            if (PC.test_search) test_search(graph->get_start_ids());
        }
        cudaFreeAsync(d_top_ids, stream);
        cudaFreeAsync(d_top_vecs, stream);
    }

    float top_enhance_time = gpu_record_time_stop(e, stream, "enhance top nodes");
    fo.iprint("connect_enhance finished(" + TOS(load_iso_time) + "s, " + TOS(iso_enhance_time) +
              "s, " + TOS(top_enhance_time) + "s): iso_num-" + TOS(iso_num));
}

};  // namespace efanna2e