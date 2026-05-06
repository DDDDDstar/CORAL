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
__global__ void build_candidate_kernel(
    // const Enhance_Data1 data,
    const int* __restrict__ d_iso_ids, const float* __restrict__ d_search_dists,
    const float* __restrict__ d_dists2iso, Slot* __restrict__ d_search_slots,
    CN* __restrict__ iso_cand_nbrs, float* __restrict__ iso_cand_dists,
    ull* __restrict__ total_degree, int iso_num, BP bp) {
    const int iso_i = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    if (iso_i >= iso_num) return;

    __shared__ int s_iso_id, s_delta_deg;
    if (tid == 0) {
        s_iso_id = d_iso_ids[iso_i];
        assert(s_iso_id >= 0);
        s_delta_deg = 0;
    }
    __syncthreads();

    const int off = iso_i * bp.k;
    Slot* search_slots = d_search_slots + off;
    const float *dists = d_search_dists + off, *dists2iso = d_dists2iso + off * bp.max_degree;
    CN* iso_cands = iso_cand_nbrs + off;
    float* cand_dists = iso_cand_dists + off;
    for (int i = tid; i < bp.k; i += tpb) {
        const float *dists2iso_ = dists2iso + i * bp.max_degree, search_dist = dists[i];
        Slot& slot = search_slots[i];
        iso_cands[i] = {slot.node_id, off + i, Initial, search_dist};
        cand_dists[i] = search_dist;

        const int deg = slot.deg;
        int pos = 0, valid = 1, out_pos;
        for (; pos < deg; ++pos)
            if (compare_dist(search_dist, slot.dists[pos], bp)) {
                if (valid && compare_dist(search_dist, dists2iso_[pos], bp)) valid = 0;
            } else
                break;

        if (deg == bp.max_degree) {
            if (!valid) continue;
            out_pos = deg - 1;
            for (; out_pos >= pos; out_pos--)
                if (compare_dist(slot.dists[out_pos], dists2iso_[out_pos], bp)) break;
            if (out_pos < pos) out_pos = deg - 1;
        } else {
            out_pos = deg;
            slot.deg = deg + 1;
            atomicAdd(&s_delta_deg, 1);
        }

        for (int j = out_pos; j > pos; --j) {
            slot.nbrs[j] = slot.nbrs[j - 1];
            slot.dists[j] = slot.dists[j - 1];
        }
        slot.nbrs[pos] = s_iso_id;
        slot.dists[pos] = search_dist;
    }
    __syncthreads();
    if (!tid) atomicAdd(total_degree, s_delta_deg);
}

__global__ void build_candidate_kernel(
    const int* __restrict__ d_iso_ids, const float* __restrict__ d_search_dists,
    const float* __restrict__ d_dists2iso, const Slot* __restrict__ d_top_slots,
    Slot* __restrict__ d_search_slots, CN* __restrict__ iso_cand_nbrs,
    float* __restrict__ iso_cand_dists, ull* __restrict__ total_degree, int iso_num, int max_num,
    BP bp) {
    const int iso_i = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x,
              cand_size = bp.k + bp.max_degree;
    if (iso_i >= iso_num) return;

    __shared__ int s_iso_id, s_delta_deg;
    if (tid == 0) {
        s_iso_id = d_iso_ids[iso_i];
        assert(s_iso_id >= 0);
        s_delta_deg = 0;
    }
    __syncthreads();

    const int off = iso_i * cand_size, search_off = iso_i * bp.k, nbr_off = iso_i * bp.max_degree;
    Slot* search_slots = d_search_slots + search_off;
    const Slot& top_slot = d_top_slots[iso_i];
    const float *dists = d_search_dists + search_off,
                *dists2iso = d_dists2iso + search_off * bp.max_degree,
                farthest = farthest_dist_d(bp);
    CN* iso_cands = iso_cand_nbrs + off;
    float* cand_dists = iso_cand_dists + off;
    const int search_vec_idx_start = max_num * bp.max_degree, top_slot_deg = top_slot.deg;
    for (int i = tid; i < cand_size; i += tpb) {
        if (i >= bp.k && i < bp.k + top_slot_deg) {
            const int nbr_i = i - bp.k, nbr_id = top_slot.nbrs[nbr_i];
            const float dist = top_slot.dists[nbr_i];
            iso_cands[i] = {nbr_id, nbr_off + nbr_i, Initial, dist};
            cand_dists[i] = dist;
        } else if (i < bp.k) {
            const float *dists2iso_ = dists2iso + i * bp.max_degree, search_dist = dists[i];
            Slot& slot = search_slots[i];
            iso_cands[i] = {slot.node_id, search_vec_idx_start + search_off + i, Initial,
                            search_dist};
            cand_dists[i] = search_dist;

            const int deg = slot.deg;
            int pos = 0, valid = 1, out_pos;
            for (; pos < deg; ++pos)
                if (compare_dist(search_dist, slot.dists[pos], bp)) {
                    if (valid && compare_dist(search_dist, dists2iso_[pos], bp)) valid = 0;
                } else
                    break;

            if (deg == bp.max_degree) {
                if (!valid) continue;
                out_pos = deg - 1;
                for (; out_pos >= pos; out_pos--)
                    if (compare_dist(slot.dists[out_pos], dists2iso_[out_pos], bp)) break;
                if (out_pos < pos) out_pos = deg - 1;
            } else {
                out_pos = deg;
                slot.deg = deg + 1;
                atomicAdd(&s_delta_deg, 1);
            }

            for (int j = out_pos; j > pos; --j) {
                slot.nbrs[j] = slot.nbrs[j - 1];
                slot.dists[j] = slot.dists[j - 1];
            }
            slot.nbrs[pos] = s_iso_id;
            slot.dists[pos] = search_dist;
        } else {
            iso_cands[i] = {-1, -1, Invalid, farthest};
            cand_dists[i] = farthest;
        }
    }
    __syncthreads();
    if (!tid) atomicAdd(total_degree, s_delta_deg);
}

__global__ void candidate_ignore_kernel_for_iso(const float* __restrict__ vecs,
                                                CN* __restrict__ cand_nbrs, const int pivot_num,
                                                int cand_size, BP bp) {
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
            const float* vec = vecs + nbr.idx * bp.dim;
            for (int j = i + 1 + tid; j < cand_size; j += tpb) {
                CN& nbr1 = cands[j];
                if (nbr1.status == Invalid) break;
                if (nbr1.id == nbr_id)
                    nbr1.status = Repeated;
                else if (nbr1.status == Initial &&
                         compare_dist(nbr1.dist, calc_distance(vec, vecs, nbr1.idx, bp), bp))
                    nbr1.status = Discarded;
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

__global__ void enhance_iso_nbrs_kernel(
    // const Enhance_Data1 data,
    const int* __restrict__ d_iso_ids, Slot* __restrict__ d_iso_slots, CN* __restrict__ cand_nbrs,
    ull* __restrict__ total_degree, ull* __restrict__ noniso_num, int pivot_num, int cand_size,
    BP bp) {
    const int tid = threadIdx.x, pivot_i = blockIdx.x * blockDim.x + tid;
    __shared__ int delta_deg, delta_noniso_num;
    if (!tid) delta_deg = delta_noniso_num = 0;
    __syncthreads();

    if (pivot_i < pivot_num) {
        int nbr_num = 0, last_id = -1;
        // Slot& slot = data.d_iso_slots[pivot_i];
        Slot& slot = d_iso_slots[pivot_i];
        slot.node_id = d_iso_ids[pivot_i];
        const CN* cands = cand_nbrs + pivot_i * cand_size;
        int retained = 0;
        for (int i = 0; i < cand_size && nbr_num < bp.max_degree; i++) {
            const CN& cand_nbr = cands[i];
            const auto status = cand_nbr.status;
            if (status == Retained) {
                retained++;
                // const int cand_nbr_id = cand_nbr.id;
                // assert(cand_nbr_id >= 0 && cand_nbr_id != last_id);
                // const int idx = nbr_num++;
                // slot.nbrs[idx] = last_id = cand_nbr_id;
                // slot.dists[idx] = cand_nbr.dist;
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

void GPUFuncs::enhance_neibor_aware_disk(Enhance_Data1& data) {
    const int cand_size = bp.max_degree + 1, pivot_num = data.batch * bp.k,
              cand_total_size = pivot_num * cand_size, tpb = 512;
    build_candidate_kernel<<<data.batch, 128, 0, data.stream>>>(
        data.d_iso_ids, data.d_search_dists, data.d_dists2query, data.d_search_slots,
        d_iso_cand_nbrs.data().get(), d_cand_dists.data().get(), d_total_degree, data.batch, bp);
    CUDA_CHECK(cudaGetLastError());

    auto res1 = segmented_sort_pairs_nocpy(
        d_iso_cand_nbrs.data().get(), d_cand_dists.data().get(), d_iso_cand_nbrs_sort.data().get(),
        d_cand_dists_sort.data().get(), d_offsets, data.stream, data.batch, bp.k);

    candidate_ignore_kernel_for_iso<<<data.batch, tpb, 0, data.stream>>>(
        data.d_search_vecs, res1.first, data.batch, bp.k, bp);
    CUDA_CHECK(cudaGetLastError());

    enhance_iso_nbrs_kernel<<<(data.batch + tpb - 1) / tpb, tpb, 0, data.stream>>>(
        data.d_iso_ids, data.d_iso_slots, res1.first, d_total_degree, d_noniso_num, data.batch,
        bp.k, bp);
    CUDA_CHECK(cudaGetLastError());
}

void GPUFuncs::enhance_neibor_aware_disk_top(Enhance_Data1& data) {
    const int cand_size = bp.max_degree + bp.k, tpb = 256;
    build_candidate_kernel<<<data.batch, 128, 0, data.stream>>>(
        data.d_iso_ids, data.d_search_dists, data.d_dists2query, data.d_iso_slots,
        data.d_search_slots, d_iso_cand_nbrs.data().get(), d_cand_dists.data().get(),
        d_total_degree, data.batch, PC.enhance_nap_batch, bp);
    CUDA_CHECK(cudaGetLastError());

    auto res1 = segmented_sort_pairs_nocpy(
        d_iso_cand_nbrs.data().get(), d_cand_dists.data().get(), d_iso_cand_nbrs_sort.data().get(),
        d_cand_dists_sort.data().get(), d_offsets, data.stream, data.batch, cand_size);

    candidate_ignore_kernel_for_iso<<<data.batch, tpb, 0, data.stream>>>(
        data.d_vecs, res1.first, data.batch, cand_size, bp);
    CUDA_CHECK(cudaGetLastError());

    enhance_iso_nbrs_kernel<<<(data.batch + tpb - 1) / tpb, tpb, 0, data.stream>>>(
        data.d_iso_ids, data.d_iso_slots, res1.first, d_total_degree, d_noniso_num, data.batch,
        cand_size, bp);
    CUDA_CHECK(cudaGetLastError());
}

inline bool bitTest(const uint32_t* bitset, int bit_index) {
    return (bitset[bit_index >> 5] & (1u << (bit_index & 31))) != 0;
}

int GPUFuncs::collect_iso_nodes(int max_num) {
    static int i = 0, total = 0;
    int iso_idx = 0;
    const uint32_t* visited = graph->get_visited();
    for (; i < bp.base_n; i++) {
        if (bitTest(visited, i)) continue;
        total++;
        h_iso_ids[iso_idx] = i;
        if ((++iso_idx) == max_num) {
            fo.print("Enhance: iso=" + TOS(total) + ", total=" + TOS(i));
            return iso_idx;
        }
    }
    fo.print("Enhance: iso=" + TOS(total) + ", total=" + TOS(i));
    return iso_idx;
}

void GPUFuncs::disk_connect_enhance() {
    float *d_iso_vecs = nullptr, *h_iso_vecs = nullptr;
    d_iso_cand_nbrs.resize(PC.enhance_nap_batch * bp.k);
    d_iso_cand_nbrs_sort.resize(PC.enhance_nap_batch * bp.k);
    d_cand_dists.resize(PC.enhance_nap_batch * bp.k);
    d_cand_dists_sort.resize(PC.enhance_nap_batch * bp.k);

    if (PC.enhance_mode == 1) {
        // graph->enhance_prepare(PC.enhance_nap_batch * bp.k);
        // cudaMalloc(&d_iso_vecs, PC.enhance_nap_batch * bp.dim * sizeof(float));
        // cudaMallocHost(&h_iso_vecs, PC.enhance_nap_batch * bp.dim * sizeof(float));
        // query_index->set_gpu(nullptr);

        // int iso_num = collect_iso_nodes(PC.enhance_nap_batch);

        // graph->load_vecs(h_iso_ids, iso_num, h_iso_vecs);
        // CUDA_CHECK(cudaMemcpy(d_iso_vecs, h_iso_vecs, iso_num * bp.dim * sizeof(float),
        //                       cudaMemcpyHostToDevice));

        // query_index->knn_Search_faiss(d_iso_vecs, h_search_ids, iso_num, 1);

        // edatas1[0].prepare(h_iso_ids, iso_num);
        // graph->load_calc_vecs_dists(d_iso_vecs, h_search_ids, iso_num, bp.k,
        //                             edatas1[0].d_search_vecs, edatas1[0].d_search_dists,
        //                             edatas1[0].stream);
        // graph->load_search_data(h_search_ids, d_iso_vecs, nullptr, edatas1[0].batch, bp.k,
        //                         edatas1[0].d_search_slots, nullptr, edatas1[0].d_dists2query,
        //                         edatas1[0].stream, false);
        // edatas1[0].record(0);

        // int ping = 0;
        // while (iso_num > 0) {
        //     auto &cdata = edatas1[ping], &ldata = edatas1[1 - ping];

        //     cdata.wait(ldata.cev);  // 防止两个 stream 的 NAP 并发
        //     enhance_neibor_aware_disk(cdata);
        //     cdata.record(1);
        //     cdata.need_write = true;

        //     iso_num = collect_iso_nodes(PC.enhance_nap_batch);
        //     if (!iso_num) {
        //         cdata.write(h_iso_slots, h_search_slots);
        //         graph->write_data(h_iso_slots, cdata.batch);
        //         graph->write_data(h_search_slots, cdata.batch * bp.k);
        //         break;
        //     }

        //     ldata.wait(cdata.lev);  // 防止两个 stream 的 load 并发
        //     if (ldata.need_write) {
        //         ldata.write(h_iso_slots, h_search_slots);
        //         graph->write_data(h_iso_slots, ldata.batch);
        //         graph->write_data(h_search_slots, ldata.batch * bp.k);
        //     }

        //     graph->load_vecs(h_iso_ids, iso_num, h_iso_vecs);
        //     CUDA_CHECK(cudaMemcpy(d_iso_vecs, h_iso_vecs, iso_num * bp.dim * sizeof(float),
        //                           cudaMemcpyHostToDevice));

        //     query_index->knn_Search_faiss(d_iso_vecs, h_search_ids, iso_num, 1);

        //     ldata.prepare(h_iso_ids, iso_num);
        //     graph->load_calc_vecs_dists(d_iso_vecs, h_search_ids, iso_num, bp.k,
        //                                 ldata.d_search_vecs, ldata.d_search_dists,
        //                                 ldata.stream);

        //     graph->load_search_data(h_search_ids, d_iso_vecs, nullptr, ldata.batch, bp.k,
        //                             ldata.d_search_slots, nullptr, ldata.d_dists2query,
        //                             ldata.stream, false);
        //     ldata.record(0);

        //     ping = 1 - ping;
        // }
    } else if (PC.enhance_mode == 0) {
        graph->search_prepare(bp.k);

        const int search_batch = PC.enhance_search_batch, nap_batch = PC.enhance_nap_batch;
        cudaMalloc(&d_iso_vecs, search_batch * bp.dim * sizeof(float));
        cudaMallocHost(&h_iso_vecs, search_batch * bp.dim * sizeof(float));
        search_prepare(search_batch, 256);
        d_search_res.resize(search_batch * bp.k);
        d_search_dist_res.resize(search_batch * bp.k);

        int iso_num = collect_iso_nodes(search_batch);
        while (iso_num) {
            graph->load_vecs(h_iso_ids, iso_num, h_iso_vecs);
            CUDA_CHECK(cudaMemcpy(d_iso_vecs, h_iso_vecs, iso_num * bp.dim * sizeof(float),
                                  cudaMemcpyHostToDevice));
            Search(d_iso_vecs, d_search_res.data().get(), d_search_dist_res.data().get(), iso_num,
                   graph->get_start_ids(), bp.k);

            const int first_batch = std::min(nap_batch, iso_num);
            edatas1[0]->prepare(h_iso_ids, first_batch);
            cudaMemcpy(h_search_ids, d_search_res.data().get(), first_batch * bp.k * sizeof(int),
                       cudaMemcpyDeviceToHost);
            // std::string str;
            // for (int i = 0; i < 2 * bp.k; i++) str += TOS(h_search_ids[i]) + " ";
            // fo.print("enhance search: " + str);
            // graph->load_vecs(h_search_ids, first_batch * bp.k, edatas1[0].h_search_vecs);
            graph->load_search_data(h_search_ids, d_iso_vecs, nullptr, first_batch, bp.k,
                                    edatas1[0]->d_search_slots, edatas1[0]->h_search_vecs,
                                    edatas1[0]->d_dists2query, edatas1[0]->stream, true);
            edatas1[0]->load(d_search_dist_res.data().get());
            edatas1[0]->record(0);

            int ping = 0;
            for (int i = 0; i < iso_num; i += nap_batch) {
                const int batch = std::min(nap_batch, iso_num - i);
                auto &cdata = *edatas1[ping], &ldata = *edatas1[1 - ping];

                cdata.wait(ldata.cev);  // 防止两个 stream 的 NAP 并发
                enhance_neibor_aware_disk(cdata);
                cdata.record(1);
                cdata.need_write = true;

                const int next_i = i + batch;
                if (next_i >= iso_num) {
                    cdata.write(h_iso_slots, h_search_slots);
                    graph->write_data(h_iso_slots, cdata.batch);
                    graph->write_data(h_search_slots, cdata.batch * bp.k);
                    break;
                }

                const int next_batch = std::min(nap_batch, iso_num - next_i);
                ldata.wait(cdata.lev);  // 防止两个 stream 的 load 并发
                if (ldata.need_write) {
                    ldata.write(h_iso_slots, h_search_slots);
                    graph->write_data(h_iso_slots, ldata.batch);
                    graph->write_data(h_search_slots, ldata.batch * bp.k);
                }

                ldata.prepare(h_iso_ids + next_i, next_batch);
                cudaMemcpy(h_search_ids, d_search_res.data().get() + next_i * bp.k,
                           next_batch * bp.k * sizeof(int), cudaMemcpyDeviceToHost);
                // graph->load_vecs(h_search_ids, next_batch * bp.k, ldata.h_search_vecs);

                graph->load_search_data(h_search_ids, d_iso_vecs + next_i * bp.dim, nullptr,
                                        ldata.batch, bp.k, ldata.d_search_slots,
                                        ldata.h_search_vecs, ldata.d_dists2query, ldata.stream,
                                        true);
                ldata.load(d_search_dist_res.data().get() + next_i * bp.k);
                ldata.record(0);

                ping = 1 - ping;
            }

            if (PC.test_search) {
                auto res = test_search(graph->get_start_ids());

                const float cost_time = res.get_time(),
                            recall = res.calc_recall(TEST_SEARCH_QUERY_SIZE * bp.k),
                            avg_hops = res.avg_hops(TEST_SEARCH_QUERY_SIZE);

                fo.iprint("Testing recall-" + TOS(recall) + "%, avg_hops-" + TOS(avg_hops) +
                          ", cost_time-" + TOS(cost_time) + "s");
            }

            iso_num = collect_iso_nodes(search_batch);
        }
    }
    cudaFree(d_iso_vecs);
    cudaFreeHost(h_iso_vecs);

    if (PC.top_hit_num > 0) {
        graph->search_prepare(bp.k);

        const int search_batch = PC.enhance_search_batch, nap_batch = PC.enhance_nap_batch,
                  total_cand = nap_batch * (bp.k + bp.max_degree), top_n = PC.top_hit_num,
                  tpb = 512;
        search_prepare(search_batch, 1024);
        d_iso_cand_nbrs.resize(total_cand);
        d_iso_cand_nbrs_sort.resize(total_cand);
        d_cand_dists.resize(total_cand);
        d_cand_dists_sort.resize(total_cand);

        std::vector<int> top_ids;
        ivf_builder->get_top_hit_nodes(top_ids, top_n);
        std::string str;
        for (int i = 0; i < 100; i++) str += TOS(top_ids[i]) + " ";
        fo.print("Enhance for top hit nodes: " + TOS(PC.top_hit_num) + ", example: " + str);

        float *d_top_vecs = nullptr, *h_top_vecs = nullptr;
        cudaMalloc(&d_top_vecs, search_batch * bp.dim * sizeof(float));
        cudaMallocHost(&h_top_vecs, search_batch * bp.dim * sizeof(float));
        d_search_res.resize(search_batch * bp.k);
        d_search_dist_res.resize(search_batch * bp.k);

        if (PC.test_search) {
            auto res = test_search(graph->get_start_ids());

            const float cost_time = res.get_time(),
                        recall = res.calc_recall(TEST_SEARCH_QUERY_SIZE * bp.k),
                        avg_hops = res.avg_hops(TEST_SEARCH_QUERY_SIZE);

            fo.iprint("Testing recall-" + TOS(recall) + "%, avg_hops-" + TOS(avg_hops) +
                      ", cost_time-" + TOS(cost_time) + "s");
        }

        for (int i = 0; i < top_n; i += search_batch) {
            const int search_n = std::min(search_batch, top_n - i);
            const int* h_top_ids = top_ids.data() + i;
            graph->load_vecs(h_top_ids, search_n, h_top_vecs);
            CUDA_CHECK(cudaMemcpy(d_top_vecs, h_top_vecs, search_n * bp.dim * sizeof(float),
                                  cudaMemcpyHostToDevice));
            auto start_ids = graph->get_search_start_nodes_fps(PC.search_start_nodes_num, 0);
            auto res = Search(d_top_vecs, d_search_res.data().get(),
                              d_search_dist_res.data().get(), search_n, start_ids, bp.k);
            fo.print("Search time: " + TOS(res.get_time()) + "s, search_n: " + TOS(search_n));

            const int first_batch = std::min(nap_batch, search_n);
            edatas1[0]->prepare(h_top_ids, first_batch);
            graph->load_data(h_top_ids, first_batch, edatas1[0]->h_iso_slots,
                             edatas1[0]->h_top_nbr_vecs, false);
            cudaMemcpy(edatas1[0]->d_iso_slots, edatas1[0]->h_iso_slots,
                       first_batch * sizeof(Slot), cudaMemcpyHostToDevice);
            cudaMemcpy(h_search_ids, d_search_res.data().get(), first_batch * bp.k * sizeof(int),
                       cudaMemcpyDeviceToHost);
            // graph->load_vecs(h_search_ids, first_batch * bp.k, edatas1[0].h_search_vecs);
            graph->load_search_data(h_search_ids, d_top_vecs, nullptr, first_batch, bp.k,
                                    edatas1[0]->d_search_slots, edatas1[0]->h_search_vecs,
                                    edatas1[0]->d_dists2query, edatas1[0]->stream, true);
            edatas1[0]->load(d_search_dist_res.data().get(), false);
            edatas1[0]->record(0);

            int ping = 0;
            for (int j = 0; j < search_n; j += nap_batch) {
                const int batch = std::min(nap_batch, search_n - j);
                auto &cdata = *edatas1[ping], &ldata = *edatas1[1 - ping];

                cdata.wait(ldata.cev);  // 防止两个 stream 的 NAP 并发
                enhance_neibor_aware_disk_top(cdata);
                cdata.record(1);
                cdata.need_write = true;

                const int next_j = j + batch;
                if (next_j >= search_n) {
                    cdata.write(h_iso_slots, h_search_slots);
                    graph->write_data(h_iso_slots, cdata.batch);
                    graph->write_data(h_search_slots, cdata.batch * bp.k);
                    break;
                }

                const int next_batch = std::min(nap_batch, search_n - next_j);
                ldata.wait(cdata.lev);  // 防止两个 stream 的 load 并发
                if (ldata.need_write) {
                    ldata.write(h_iso_slots, h_search_slots);
                    graph->write_data(h_iso_slots, ldata.batch);
                    graph->write_data(h_search_slots, ldata.batch * bp.k);
                }

                ldata.prepare(h_top_ids + next_j, next_batch);
                graph->load_data(h_top_ids + next_j, next_batch, ldata.h_iso_slots,
                                 ldata.h_top_nbr_vecs, false);
                cudaMemcpy(ldata.d_iso_slots, ldata.h_iso_slots, next_batch * sizeof(Slot),
                           cudaMemcpyHostToDevice);
                cudaMemcpy(h_search_ids, d_search_res.data().get() + next_j * bp.k,
                           next_batch * bp.k * sizeof(int), cudaMemcpyDeviceToHost);
                // graph->load_vecs(h_search_ids, next_batch * bp.k, ldata.h_search_vecs);

                graph->load_search_data(h_search_ids, d_top_vecs + next_j * bp.dim, nullptr,
                                        ldata.batch, bp.k, ldata.d_search_slots,
                                        ldata.h_search_vecs, ldata.d_dists2query, ldata.stream,
                                        true);
                ldata.load(d_search_dist_res.data().get() + next_j * bp.k, false);
                ldata.record(0);

                ping = 1 - ping;
            }
            fo.print("top enchance finished with i=" + TOS(i + search_n) + "/" + TOS(top_n));
            if (PC.test_search) {
                auto res = test_search(graph->get_start_ids());

                const float cost_time = res.get_time(),
                            recall = res.calc_recall(TEST_SEARCH_QUERY_SIZE * bp.k),
                            avg_hops = res.avg_hops(TEST_SEARCH_QUERY_SIZE);

                fo.iprint("Testing recall-" + TOS(recall) + "%, avg_hops-" + TOS(avg_hops) +
                          ", cost_time-" + TOS(cost_time) + "s");
            }
        }
        cudaFree(d_top_vecs);
        cudaFreeHost(h_top_vecs);
    }
}

void GPUFuncs::enhance_prepare() {
    cudaMallocHost(&h_offsets, (PC.enhance_nap_batch + 1) * sizeof(size_t));
    cudaMalloc(&d_offsets, (PC.enhance_nap_batch + 1) * sizeof(size_t));
    auto gt = graph_type();

    if (gt == GraphType::GPU)
        d_enhance_updated.resize((bp.base_n + 31) / 32);
    else {
        const int search_batch = PC.enhance_nap_batch;
        // search_prepare(search_batch);
        CUDA_CHECK(cudaMallocHost(&h_iso_ids, PC.enhance_search_batch * sizeof(int)));
        // CUDA_CHECK(cudaMallocHost(&h_iso_vecs, search_batch * bp.dim * sizeof(float)));
        // CUDA_CHECK(cudaMalloc(&d_iso_vecs, search_batch * bp.dim * sizeof(float)));
        // CUDA_CHECK(cudaMalloc(&d_iso_ids, search_batch * sizeof(int)));
        // CUDA_CHECK(cudaMalloc(&d_search_res, search_batch * bp.k * sizeof(int)));
        // CUDA_CHECK(cudaMalloc(&d_search_dist_res, search_batch * bp.k * sizeof(float)));

        CUDA_CHECK(cudaMallocHost(&h_search_ids, search_batch * bp.k * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&h_search_slots, PC.enhance_nap_batch * bp.k * sizeof(Slot)));
        CUDA_CHECK(cudaMallocHost(&h_iso_slots, PC.enhance_nap_batch * sizeof(Slot)));
        // CUDA_CHECK(
        //     cudaMallocHost(&h_search_vecs, PC.enhance_nap_batch * bp.k * bp.dim *
        //     sizeof(float)));

        if (!MEM_MODE) {
            d_enhance_updated.resize((GPU_N + 31) / 32);
            edatas.emplace_back(PC.enhance_nap_batch, bp);
            edatas.emplace_back(PC.enhance_nap_batch, bp);
        } else {
            edatas1.emplace_back(new Enhance_Data1(PC.enhance_nap_batch, bp));
            edatas1.emplace_back(new Enhance_Data1(PC.enhance_nap_batch, bp));
        }
    }
}
void GPUFuncs::enhance_free() {
    cuda_check_last_error("enhance free");
    thrust::device_vector<CN>().swap(d_cand_nbrs);
    thrust::device_vector<float>().swap(d_cand_dists);
    thrust::device_vector<float>().swap(d_cand_dists_sort);
    thrust::device_vector<uint32_t>().swap(d_enhance_updated);
    thrust::device_vector<int>().swap(d_search_res);
    thrust::device_vector<float>().swap(d_search_dist_res);

    if (h_iso_ids) cudaFreeHost(h_iso_ids);
    // if (h_iso_vecs) cudaFreeHost(h_iso_vecs);
    // if (d_iso_vecs) cudaFree(d_iso_vecs);
    // if (d_iso_ids) cudaFree(d_iso_ids);
    if (d_offsets) cudaFree(d_offsets);
    if (h_search_ids) cudaFreeHost(h_search_ids);
    if (h_search_slots) cudaFreeHost(h_search_slots);
    // if (h_search_vecs) cudaFreeHost(h_search_vecs);
    if (h_offsets) cudaFreeHost(h_offsets);

    edatas.clear();
    if (edatas1.size())
        for (auto ed : edatas1) delete ed;
    edatas1.clear();
    graph->enhance_free();
}
};  // namespace efanna2e
