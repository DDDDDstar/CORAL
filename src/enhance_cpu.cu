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

__device__ __forceinline__ int atomicTestAndSetBit(uint32_t* bitset, int bit_index) {
    const uint32_t mask = 1u << (bit_index & 31);
    uint32_t* addr = bitset + (bit_index >> 5);
    const uint32_t old = atomicOr(addr, mask);
    return (old & mask) != 0;
}
__global__ void update_new_nbrs_kernel(const Enhance_Data data, const DeviceView view,
                                       CN* __restrict__ cand_nbrs, uint32_t* __restrict__ updated,
                                       ull* __restrict__ total_degree,
                                       ull* __restrict__ noniso_num, int pivot_num, int cand_size,
                                       BP bp) {
    const int tid = threadIdx.x, pivot_i = blockIdx.x * blockDim.x + tid;
    __shared__ int delta_deg, delta_noniso_num;
    if (!tid) delta_deg = delta_noniso_num = 0;
    __syncthreads();

    if (pivot_i < pivot_num) {
        const int pivot_idx = data.d_idxs[pivot_i];
        assert(pivot_idx >= 0);
        if (!atomicTestAndSetBit(updated, pivot_idx)) {
            int nbr_num = 0, last_id = -1;
            Slot& slot = view.d_slots[pivot_idx];
            const CN* cands = cand_nbrs + pivot_i * cand_size;
            for (int i = 0; i < cand_size && nbr_num < bp.max_degree; i++) {
                const CN& cand_nbr = cands[i];
                if (cand_nbr.status == Retained) {
                    const int cand_nbr_id = cand_nbr.id;
                    assert(cand_nbr_id >= 0 && cand_nbr_id != last_id);
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
                    assert(cand_nbr_id >= 0 && cand_nbr_id != last_id);
                    const int idx = nbr_num++;
                    last_id = slot.nbrs[idx] = cand_nbr_id;
                    slot.dists[idx] = cand_nbr.dist;
                } else if (cand_nbr.status == Invalid)
                    break;
            }
            assert(nbr_num > 0);
            view.d_dirty[pivot_idx] = 1;
            const uint32_t pin_id = data.pin_id;
            atomicCAS(view.d_pin + pivot_idx, pin_id, 0);
            atomicCAS(view.d_vec_pin + data.d_vec_idxs[pivot_i], pin_id, 0);

            const int* nbr_vec_idxs = data.d_nbr_vec_idxs + pivot_i * bp.max_degree;
#pragma unroll 1
            for (int i = 0; i < slot.deg; i++)
                atomicCAS(view.d_vec_pin + nbr_vec_idxs[i], pin_id, 0);

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
    if (tid == 0) {
        atomicAdd(total_degree, delta_deg);
        atomicAdd(noniso_num, delta_noniso_num);
    }
}

__global__ void build_candidate_kernel(const DeviceView view, const Enhance_Data data,
                                       CN* __restrict__ cand_nbrs,      // iso_num * k * (deg + 1)
                                       CN* __restrict__ iso_cand_nbrs,  // iso_num * k
                                       int iso_num, int cand_size, int iso_cand_size, BP bp) {
    const int iso_i = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    if (iso_i >= iso_num) return;

    __shared__ int s_iso_id, s_iso_vec_idx;
    if (tid == 0) {
        s_iso_id = data.d_iso_ids[iso_i];
        s_iso_vec_idx = data.d_iso_vec_idxs[iso_i];
        assert(s_iso_id >= 0);
        assert(s_iso_vec_idx >= 0);
    }
    __syncthreads();

    const int off = iso_i * iso_cand_size, *idxs = data.d_idxs + off,
              *vec_idxs = data.d_vec_idxs + off,
              *nbr_vec_idxs = data.d_nbr_vec_idxs + off * bp.max_degree;
    const float *dists = data.d_search_nbr_dists + off, farthest = farthest_dist_d(bp);
    CN* iso_cands = iso_cand_nbrs + off;
    for (int i = tid; i < bp.k * cand_size; i += tpb) {
        const int s_nbr_i = i / cand_size, cand_i = i % cand_size,
                  *nbr_vec_idxs_ = nbr_vec_idxs + s_nbr_i * bp.max_degree;
        const Slot& nbr_slot = view.d_slots[idxs[s_nbr_i]];
        CN* cand_nbrs_start = cand_nbrs + iso_i * bp.k * cand_size + s_nbr_i * cand_size;
        if (cand_i < nbr_slot.deg) {
            const int old_nbr_id = nbr_slot.nbrs[cand_i];
            assert(old_nbr_id >= 0);
            cand_nbrs_start[cand_i] = {old_nbr_id, nbr_vec_idxs_[cand_i], Initial,
                                       nbr_slot.dists[cand_i]};
        } else if (cand_i < bp.max_degree)
            cand_nbrs_start[cand_i] = {-1, 0, Invalid, farthest};
        else {
            const float dist = dists[s_nbr_i];
            cand_nbrs_start[cand_i] = {s_iso_id, s_iso_vec_idx, Initial, dist};
            iso_cands[s_nbr_i] = {nbr_slot.node_id, vec_idxs[s_nbr_i], Initial, dist};
        }
        __syncthreads();
    }
}

__global__ void update_iso_nbrs_kernel(const Enhance_Data data, const DeviceView view,
                                       CN* __restrict__ cand_nbrs, ull* __restrict__ total_degree,
                                       ull* __restrict__ noniso_num, int pivot_num, int cand_size,
                                       BP bp) {
    const int tid = threadIdx.x, pivot_i = blockIdx.x * blockDim.x + tid;
    __shared__ int delta_deg, delta_noniso_num;
    if (!tid) delta_deg = delta_noniso_num = 0;
    __syncthreads();

    if (pivot_i < pivot_num) {
        const int pivot_idx = data.d_iso_idxs[pivot_i];
        assert(pivot_idx >= 0);
        int nbr_num = 0, last_id = -1;
        Slot& slot = view.d_slots[pivot_idx];
        const CN* cands = cand_nbrs + pivot_i * cand_size;
        for (int i = 0; i < cand_size && nbr_num < bp.max_degree; i++) {
            const CN& cand_nbr = cands[i];
            if (cand_nbr.status == Retained) {
                const int cand_nbr_id = cand_nbr.id;
                assert(cand_nbr_id >= 0 && cand_nbr_id != last_id);
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
                assert(cand_nbr_id >= 0 && cand_nbr_id != last_id);
                const int idx = nbr_num++;
                last_id = slot.nbrs[idx] = cand_nbr_id;
                slot.dists[idx] = cand_nbr.dist;
            } else if (cand_nbr.status == Invalid)
                break;
        }
        assert(nbr_num > 0);
        view.d_dirty[pivot_idx] = 1;
        const uint32_t pin_id = data.pin_id;
        atomicCAS(view.d_pin + pivot_idx, pin_id, 0);
        atomicCAS(view.d_vec_pin + data.d_iso_vec_idxs[pivot_i], pin_id, 0);

        atomicAdd(&delta_noniso_num, 1);
        atomicAdd(&delta_deg, nbr_num);
        slot.deg = nbr_num;
    }
    __syncthreads();
    if (tid == 0) {
        atomicAdd(total_degree, delta_deg);
        atomicAdd(noniso_num, delta_noniso_num);
    }
}

void GPUFuncs::enhance_neibor_aware_cpu(const DeviceView& view, Enhance_Data& data) {
    const int cand_size = bp.max_degree + 1, pivot_num = data.batch * bp.k,
              cand_total_size = pivot_num * cand_size, tpb = 512;
    build_candidate_kernel<<<data.batch, tpb, 0, data.stream>>>(
        view, data, d_cand_nbrs.data().get(), d_iso_cand_nbrs.data().get(), data.batch, cand_size,
        bp.k, bp);
    CUDA_CHECK(cudaGetLastError());

    dist_extract_from_CN(d_cand_nbrs.data().get(), d_cand_dists.data().get(), data.stream,
                         cand_total_size, bp);
    auto res = segmented_sort_pairs_nocpy(
        d_cand_nbrs.data().get(), d_cand_dists.data().get(), d_cand_nbrs_sort.data().get(),
        d_cand_dists_sort.data().get(), d_offsets, data.stream, pivot_num, cand_size);

    const int blocks = (pivot_num + tpb - 1) / tpb;
    candidate_ignore_kernel<<<blocks, tpb, 0, data.stream>>>(view.d_vecs, res.first, pivot_num,
                                                             cand_size, bp);
    CUDA_CHECK(cudaGetLastError());

    thrust::fill(thrust::cuda::par.on(data.stream), d_enhance_updated.begin(),
                 d_enhance_updated.end(), 0);
    update_new_nbrs_kernel<<<blocks, tpb, 0, data.stream>>>(
        data, view, res.first, d_enhance_updated.data().get(), d_total_degree, d_noniso_num,
        pivot_num, cand_size, bp);

    dist_extract_from_CN(d_iso_cand_nbrs.data().get(), d_cand_dists.data().get(), data.stream,
                         pivot_num, bp);
    auto res1 = segmented_sort_pairs_nocpy(
        d_iso_cand_nbrs.data().get(), d_cand_dists.data().get(), d_iso_cand_nbrs_sort.data().get(),
        d_cand_dists_sort.data().get(), d_offsets, data.stream, data.batch, bp.k);

    candidate_ignore_kernel_for_iso<<<data.batch, tpb, 0, data.stream>>>(view.d_vecs, res1.first,
                                                                         data.batch, bp.k, bp);
    CUDA_CHECK(cudaGetLastError());

    update_iso_nbrs_kernel<<<(data.batch + tpb - 1) / tpb, tpb, 0, data.stream>>>(
        data, view, res1.first, d_total_degree, d_noniso_num, data.batch, bp.k, bp);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(data.ev, data.stream));
}

void GPUFuncs::search_and_nap(int num, GpuClockCache* gpu_cache, const DeviceView& view) {
    // CUDA_CHECK(cudaMemcpy(d_iso_ids, h_iso_ids, num * sizeof(int), cudaMemcpyHostToDevice));
    // CUDA_CHECK(
    //     cudaMemcpy(d_iso_vecs, h_iso_vecs, num * bp.dim * sizeof(float),
    //     cudaMemcpyHostToDevice));
    // Search(d_iso_vecs, d_search_res, d_search_dist_res, num, bp.k);
    // int ping = 0, nap_batch = PC.enhance_nap_batch;
    // uint32_t pin_id = 0;
    // edatas[0].prepare(d_iso_ids, d_search_res, d_search_dist_res, std::min(nap_batch, num),
    //                   pin_id++);
    // gpu_cache->ensure_self_from_device_ids(d_iso_ids, edatas[0].batch, edatas[0].pin_id, true,
    //                                        edatas[0].stream);
    // gpu_cache->ensure_cached_from_device_ids(d_search_res, edatas[0].batch * bp.k,
    //                                          edatas[0].pin_id, true, edatas[0].stream);
    // gpu_cache->lookup_slots_vecs(d_iso_ids, edatas[0].batch, edatas[0].d_iso_idxs,
    //                              edatas[0].d_iso_vec_idxs, edatas[0].stream, true);
    // gpu_cache->lookup_all_data(d_search_res, edatas[0].batch * bp.k, edatas[0].d_idxs,
    //                            edatas[0].d_vec_idxs, edatas[0].d_nbr_vec_idxs, edatas[0].stream,
    //                            true);
    // for (int j = 0; j < num; j += nap_batch, ping = 1 - ping) {
    //     auto &cdata = edatas[ping], &ldata = edatas[1 - ping];
    //     enhance_neibor_aware_cpu(view, cdata);
    //     const int nj = j + nap_batch;
    //     if (nj >= num) break;

    //     ldata.prepare(d_iso_ids + nj, d_search_res + nj * bp.max_degree,
    //                   d_search_dist_res + nj * bp.max_degree, std::min(nap_batch, num - nj),
    //                   pin_id++);
    //     gpu_cache->ensure_self_from_device_ids(ldata.d_iso_ids, ldata.batch, ldata.pin_id, true,
    //                                            ldata.stream);
    //     gpu_cache->ensure_cached_from_device_ids(d_search_res + nj * bp.max_degree,
    //                                              ldata.batch * bp.max_degree, ldata.pin_id,
    //                                              false, ldata.stream);
    //     gpu_cache->lookup_slots_vecs(ldata.d_iso_ids, ldata.batch, ldata.d_iso_idxs,
    //                                  ldata.d_iso_vec_idxs, ldata.stream, true);
    //     gpu_cache->lookup_all_data(ldata.d_search_nbr_ids, ldata.batch * bp.max_degree,
    //                                ldata.d_idxs, nullptr, ldata.d_nbr_vec_idxs, ldata.stream,
    //                                false);
    //     CUDA_CHECK(cudaStreamWaitEvent(ldata.stream, cdata.ev));  // 防止两个 stream 的 NAP 并发
    // }
}
void GPUFuncs::cpu_connect_enhance() {
    // assert(graph->Type() == GraphType::CPU);
    // cudaStream_t stream = streams["search"].stream;
    // auto gpu_cache = graph->GPUCache();
    // const auto view = gpu_cache->device_view();
    // gpu_cache->flush_all();
    // const auto& cpu_slots = graph->get_cpu_slots();
    // const float* cpu_vecs = graph->get_cpu_vec_slots();
    // const int search_batch = PC.enhance_batch;
    // int iso_idx = 0;
    // uint32_t pin_id = 0;
    // for (int i = 0; i < bp.base_n; i++) {
    //     if (cpu_slots[i].deg > 0) continue;
    //     h_iso_ids[iso_idx] = i;
    //     memcpy(h_iso_vecs + iso_idx * bp.dim, cpu_vecs + i * bp.dim, bp.dim * sizeof(float));
    //     if ((++iso_idx) == search_batch) {
    //         search_and_nap(search_batch, gpu_cache, view);
    //         iso_idx = 0;
    //     }
    // }
    // if (iso_idx > 0) search_and_nap(iso_idx, gpu_cache, view);
}

};  // namespace efanna2e