#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <cfloat>
#include <cmath>
#include <cub/cub.cuh>
#include <iostream>
#include <string>
#include <utility>

#include "fileout.h"
#include "gpufuncs.cuh"
#include "graph.cuh"
#include "utils.cuh"

namespace efanna2e {
__device__ __forceinline__ float warp_calc_dist(const float* __restrict__ base,
                                                const float* __restrict__ query, int lane, BP bp) {
    float sum = 0.f;
    for (int d = lane; d < bp.dim; d += 32) {
        float diff = query[d] - base[d];
        sum += (bp.metric == DIST_METRIC::L2_) ? diff * diff : query[d] * base[d];
    }
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
    return __shfl_sync(0xffffffff, sum, 0);
}

std::pair<int*, float*> GPUFuncs::test_query_load(const std::vector<float>& query_data,
                                                  const std::vector<int>& gt_data, int k) {
    gtk = k;
    const size_t size = query_data.size();
    const int Q = size / bp.dim;
    cudaMalloc(&test_query_data, size * sizeof(float));
    cudaMalloc(&test_query_knns, Q * gtk * sizeof(int));
    cudaMemcpy(test_query_data, query_data.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(test_query_knns, gt_data.data(), Q * gtk * sizeof(int), cudaMemcpyHostToDevice);

    if (mode_ != Mode::CE) {
        const int* ivf_res = ivf_builder->search(query_data.data(), Q, false);
        std::string str;
        for (int i = 0; i < bp.k; i++) str += TOS(ivf_res[i]) + " ";
        fo.print("ivf_res=" + str);
        int hits = 0;
        for (int i = 0; i < Q; ++i)
            for (int j = 0; j < bp.k; ++j)
                for (int l = 0; l < gtk; ++l)
                    if (ivf_res[i * bp.k + j] == gt_data[i * gtk + l]) {
                        ++hits;
                        break;
                    }
        fo.print("test_query_load: recall=" + TOS(100.0 * hits / (Q * bp.k)) + "%");
    }
    return std::make_pair(test_query_knns, test_query_data);
}

std::pair<int*, float*> GPUFuncs::test_query_prepare(const std::vector<float>& query_data) {
    gtk = bp.k;
    const size_t size = query_data.size();
    const int Q = size / bp.dim;
    cudaMalloc(&test_query_data, size * sizeof(float));
    cudaMalloc(&test_query_knns, Q * bp.k * sizeof(int));
    cudaMemcpy(test_query_data, query_data.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    float time = 0.0f;
    MyFaiss* myfaiss = nullptr;
    std::vector<int> knns_res;
    int hit = 0;
    if ((size_t)bp.base_n * bp.dim > GPU_N * 200) {
        myfaiss = new MyFaiss(bp.dim, bp.k, bp.metric, false);
        myfaiss->Add(h_base, bp.base_n);
        knns_res.resize(Q * bp.k);
        myfaiss->CPUSearch(query_data.data(), knns_res.data(), Q);
        cudaMemcpy(test_query_knns, knns_res.data(), Q * bp.k * sizeof(int),
                   cudaMemcpyHostToDevice);
        delete myfaiss;
    } else
        for (int i = 0; i < Q; i += PC.batch) {
            const int batch = std::min(PC.batch, Q - i);
            knn_compute_faiss(test_query_data + i * bp.dim, test_query_knns + i * bp.k, batch,
                              time);
        }
    return std::make_pair(test_query_knns, test_query_data);
}

__device__ __forceinline__ int beam_insert(int* hash_beam, int key, int cap) {
#ifdef VERBOSE
    assert(key >= 0);
#endif
    // const int mask = cap - 1;
    // const uint32_t h = hash1((uint32_t)key) % cap;
    // const uint32_t h = key % cap;
    for (int i = 0; i < cap; ++i) {  // linear probing
        const int old = atomicCAS(hash_beam + (key + i) % cap, -1, key);
        if (old == -1) return 1;   // 成功插入
        if (old == key) return 0;  // 已存在
    }
    assert(false);
    return -1;
}
__global__ void beam_search_kernel(const int* __restrict__ start_node_ids,
                                   const float* __restrict__ queries, const DeviceView view,
                                   const int* __restrict__ start_vec_idxs,  // [M]
                                   Beam_Data beam_data, int* __restrict__ qids,
                                   int* __restrict__ ids_res, float* __restrict__ dists_res,
                                   int* __restrict__ hops, int* __restrict__ unprocessed_qid,
                                   int* __restrict__ finish_num, uint32_t pin_id, int k, int Q,
                                   int M, BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x, wpb = tpb / warpSize;
    const int lane = tid % warpSize;  // warp 内线程号
    const int warp = tid / warpSize;  // block 内 warp 号
    const float farthest = farthest_dist_d(bp), closest = closest_dist_d(bp);

    extern __shared__ float sh_query[];                       // dim
    float* sh_dists = sh_query + bp.dim;                      // tpb
    float* sh_beam_dists = sh_dists + tpb;                    // beam_capacity
    int* sh_idxs = (int*)(sh_beam_dists + bp.beam_capacity);  // tpb
    int* sh_beam_ids = sh_idxs + tpb;                         // beam_capacity

    __shared__ int s_qid, s_new_cand_idx, s_beam_num, s_hop;
    __shared__ float s_farthest_dist;  // 当前 beam 中最远距离
    __shared__ unsigned short s_flag;

    int* beam_hash = beam_data.d_beams_hash + bid * bp.hash_n;
    uint8_t* beam_expanded = beam_data.d_expanded + bid * bp.beam_capacity;
    int* ensure_ids = beam_data.d_beam_ids + bid * bp.beam_capacity;
    int* beam_ids = beam_data.d_ids + bid * bp.beam_size;
    float* beam_dists = beam_data.d_dists + bid * bp.beam_size;
    const int *beam_idxs = beam_data.d_idxs + bid * bp.beam_capacity,
              *beam_nbr_vec_idxs =
                  beam_data.d_nbr_vec_idxs + bid * bp.beam_capacity * bp.max_degree;

    if (tid == 0) s_qid = qids[bid];
    __syncthreads();

    if (s_qid >= Q) return;

    if (s_qid < 0) {
        if (tid == 0) {
            s_qid = atomicAdd(unprocessed_qid, 1);  // 获取新的 qid
            if (s_qid >= Q) {
                atomicAdd(finish_num, 1);
            } else {
                beam_data.d_nums[bid] = M;
                beam_data.d_farthests[bid] = farthest;
            }
        }
        __syncthreads();
        if (s_qid >= Q) return;  // 所有 qid 都处理完了
        // 初始化：
        for (int d = tid; d < bp.dim; d += tpb) sh_query[d] = queries[s_qid * bp.dim + d];
        for (int i = tid; i < bp.beam_capacity; i += tpb) beam_expanded[i] = false;
        for (int i = tid; i < bp.hash_n; i += tpb) beam_hash[i] = -1;
        __syncthreads();
        for (int start_i = warp; start_i < M; start_i += wpb) {
            int vec_idx = -1;
            if (lane == 0) {
                const int id = ensure_ids[start_i] = beam_ids[start_i] = start_node_ids[start_i];
                vec_idx = start_vec_idxs[start_i];
                beam_insert(beam_hash, id, bp.hash_n);
            }
            vec_idx = __shfl_sync(0xffffffff, vec_idx, 0);
            const float dist =
                warp_calc_dist(view.d_vecs + (size_t)vec_idx * bp.dim, sh_query, lane, bp);
            if (lane == 0) beam_dists[start_i] = dist;
        }
        for (int i = M + tid; i < bp.beam_capacity; i += tpb) ensure_ids[i] = beam_ids[i] = -1;
        return;
    }

    if (!tid) {
        s_flag = 1;
        s_hop = 0;
        s_new_cand_idx = s_beam_num = beam_data.d_nums[bid];
        s_farthest_dist = beam_data.d_farthests[bid];
    }
    __syncthreads();
    for (int i = tid; i < s_beam_num; i += tpb) {
        sh_beam_ids[i] = beam_ids[i];
        sh_beam_dists[i] = beam_dists[i];
    }
    for (int d = tid; d < bp.dim; d += tpb) sh_query[d] = queries[s_qid * bp.dim + d];
    __syncthreads();

    for (int beam_i = warp; beam_i < s_beam_num; beam_i += wpb) {
        const int idx = beam_idxs[beam_i];
        int deg = -1;
        if (lane == 0 && !beam_expanded[beam_i]) {
#ifdef VERBOSE
            assert(idx >= 0);
#endif
            deg = view.d_slots[idx].deg;
            beam_expanded[beam_i] = true;
        }
        deg = __shfl_sync(0xffffffff, deg, 0);
        if (deg <= 0) continue;

        const int *nbrs = view.d_slots[idx].nbrs,
                  *nbr_idxs = beam_nbr_vec_idxs + beam_i * bp.max_degree;
        for (int nbr_i = lane; nbr_i < deg; nbr_i += warpSize) {  // warp 内遍历邻居
            // bool valid = false;
            const int nbrid = nbrs[nbr_i], nbr_vec_idx = nbr_idxs[nbr_i];
            // float dist;
            if (beam_insert(beam_hash, nbrid, bp.hash_n) == 1) {
                const float dist = calc_distance(sh_query, view.d_vecs, nbr_vec_idx, bp);
                if (compare_dist(s_farthest_dist, dist, bp)) {
                    const int pos = atomicAdd(&s_new_cand_idx, 1);
                    beam_ids[pos] = nbrid;
                    beam_dists[pos] = dist;
                }
            }
            atomicCAS(view.d_vec_pin + nbr_vec_idx, pin_id, 0);
            // warp 压缩写入 beam
            // const unsigned active = __activemask(), mask = __ballot_sync(active, valid);
            // const int cnt = __popc(mask);
            // if (cnt > 0) {
            //     int base_idx = -1;
            //     if (lane == 0) base_idx = atomicAdd(&s_new_cand_idx, cnt);
            //     base_idx = __shfl_sync(active, base_idx, 0);
            //     if (valid) {
            //         const int idx = base_idx + __popc(mask & ((1u << lane) - 1));
            //         beam_ids[idx] = nbrid;
            //         beam_dists[idx] = dist;
            //     }
            // }
        }
        if (!lane) atomicCAS(view.d_pin + idx, pin_id, 0);
    }
    __syncthreads();

    const int total = s_new_cand_idx;
    if (total == s_beam_num) {  // 收敛, 筛选 top-k 的节点
#ifdef VERBOSE
        assert(s_beam_num == bp.beam_capacity);
#endif
        int* ids_out = ids_res + s_qid * k;
        float* dists_out = dists_res ? dists_res + s_qid * k : nullptr;
        for (int i = 0; i < k; i++) {
            int local_idx = -1;
            float local_best = farthest;
            for (int j = tid; j < bp.beam_capacity; j += tpb) {
                if (sh_beam_ids[j] < 0) continue;
                const float dist = sh_beam_dists[j];
                if (compare_dist(local_best, dist, bp)) {
                    local_best = dist;
                    local_idx = j;
                }
            }
            sh_dists[tid] = local_best;
            sh_idxs[tid] = local_idx;
            __syncthreads();

            for (int offset = tpb >> 1; offset > 0; offset >>= 1) {
                if (tid < offset && compare_dist(sh_dists[tid], sh_dists[tid + offset], bp)) {
                    sh_dists[tid] = sh_dists[tid + offset];
                    sh_idxs[tid] = sh_idxs[tid + offset];
                }
                __syncthreads();
            }
            if (tid == 0) {
                const int ith_idx = sh_idxs[0];
                ids_out[i] = sh_beam_ids[ith_idx];
                if (dists_out) dists_out[i] = sh_dists[0];
                sh_beam_ids[ith_idx] = -1;
            }
            __syncthreads();
        }
        for (int i = tid; i < bp.beam_capacity; i += tpb) ensure_ids[i] = -1;
        if (tid == 0) {
            hops[s_qid] += s_hop;
            s_qid = -1;
        }
    } else if (total < bp.beam_capacity) {
        for (int i = s_beam_num + tid; i < total; i += tpb) ensure_ids[i] = beam_ids[i];
        if (tid == 0) {
            atomicAdd(&s_hop, total - s_beam_num);
            beam_data.d_nums[bid] = total;
        }
    } else {  // 筛选 top-beam_capacity 的节点
        float farthest_d;
        for (int i = 0; i < bp.beam_capacity; i++) {
            int local_idx = -1;
            float local_best = farthest;
            for (int j = tid; j < total; j += tpb) {
                if (beam_ids[j] < 0) continue;
                const float dist = beam_dists[j];
                if (compare_dist(local_best, dist, bp)) {
                    local_best = dist;
                    local_idx = j;
                }
            }
            sh_dists[tid] = local_best;
            sh_idxs[tid] = local_idx;
            __syncthreads();

            for (int offset = tpb >> 1; offset > 0; offset >>= 1) {
                if (tid < offset && compare_dist(sh_dists[tid], sh_dists[tid + offset], bp)) {
                    sh_dists[tid] = sh_dists[tid + offset];
                    sh_idxs[tid] = sh_idxs[tid + offset];
                }
                __syncthreads();
            }
            if (tid == 0) {
                const int ith_idx = sh_idxs[0];
                const int id = beam_ids[ith_idx];
                // if (i > 0 && id == sh_beam_ids[i - 1]) {
                //     beam_ids[ith_idx] = -1;
                //     beam_dists[ith_idx] = farthest;
                //     i--;
                //     continue;
                // }
                sh_beam_ids[i] = id;
                sh_beam_dists[i] = beam_dists[ith_idx];
                beam_dists[ith_idx] = farthest;
                beam_ids[ith_idx] = -1;
                beam_expanded[i] = ith_idx < s_beam_num ? beam_expanded[ith_idx] : false;
            }
            __syncthreads();
        }

        for (int i = tid; i < bp.beam_capacity; i += tpb) {
            ensure_ids[i] = beam_ids[i] = sh_beam_ids[i];
            beam_dists[i] = sh_beam_dists[i];
        }
        if (tid == 0) {
            beam_data.d_farthests[bid] = sh_dists[0];
            atomicAdd(&s_hop, total - s_beam_num);
            beam_data.d_nums[bid] = bp.beam_capacity;
        }
    }
}

Test_Result GPUFuncs::search_cpu(const float* queries, int* ids_res, float* dists_res, int Q,
                                 int k, int reset) {
    if (k <= 0) k = bp.k;
    const int M = PC.search_start_nodes_num, Q1 = Q / 2, Q2 = Q - Q1;
    auto gpu_cache = graph->GPUCache();
    auto view = gpu_cache->device_view();

    auto start_node_ids = graph->get_search_start_nodes_fps(M, reset);

    thrust::device_vector<int> d_start_node_ids(M), d_unprocessed_qid(1, 0), d_hops(Q, 0);
    CUDA_CHECK(cudaMemcpy(d_start_node_ids.data().get(), start_node_ids.data(), M * sizeof(int),
                          cudaMemcpyHostToDevice));

    for (auto data : search_data) data->reset();

    auto s = std::chrono::high_resolution_clock::now();

    gpu_cache->ensure_cached_from_device_ids(d_start_node_ids.data().get(), M, UINT32_MAX, true,
                                             streams["search"].stream);
    gpu_cache->lookup_vecs(d_start_node_ids.data().get(), M, d_start_vec_idxs.data().get(),
                           nullptr);

    const int tpb = 512, shared_size = (bp.dim + tpb + bp.beam_capacity) * sizeof(float) +
                                       (tpb + bp.beam_capacity * 3) * sizeof(int);
    uint32_t step = 0;
    int ping = 0, not_converge[2] = {1, 1};
    while (not_converge[0] || not_converge[1]) {
        auto sdata = search_data[ping];
        if (not_converge[ping]) {
            beam_search_kernel<<<sdata->batch_, tpb, shared_size, sdata->stream_>>>(
                d_start_node_ids.data().get(), queries, view, d_start_vec_idxs.data().get(),
                sdata->beam_data_, sdata->d_qids_, ids_res, dists_res, d_hops.data().get(),
                d_unprocessed_qid.data().get(), sdata->d_finish_num_, step, k, sdata->Q_, M, bp);
            CUDA_CHECK(cudaGetLastError());
        }
        if (step++ && not_converge[1 - ping]) {
            auto ldata = search_data[1 - ping];
            if (ldata->converge())
                not_converge[1 - ping] = 0;
            else {
                const int n = ldata->batch_ * bp.beam_capacity;
                int* d_ensure_ids = ldata->beam_data_.d_beam_ids;
                gpu_cache->ensure_cached_from_device_ids(d_ensure_ids, n, step, false,
                                                         ldata->stream_);
                gpu_cache->lookup_all_data(d_ensure_ids, n, ldata->beam_data_.d_idxs, nullptr,
                                           ldata->beam_data_.d_nbr_vec_idxs, ldata->stream_,
                                           false);
            }
        }
        ping = 1 - ping;
    }

    gpu_cache->unpin_from_ids(d_start_node_ids.data().get(), M, UINT32_MAX,
                              streams["search"].stream);

    const float time = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::high_resolution_clock::now() - s)
                           .count() /
                       1000.0;
    int* hops;
    cudaMallocHost(&hops, Q * sizeof(int));
    cudaMemcpy(hops, d_hops.data().get(), Q * sizeof(int), cudaMemcpyDeviceToHost);
    size_t total_hop = 0;
    for (int i = 0; i < Q; i++) total_hop += hops[i];
    return Test_Result(-1, total_hop, time);
}

void GPUFuncs::search_prepare(int Q, int beam_capacity) {
    bp.beam_capacity = beam_capacity;
    bp.beam_size = bp.beam_capacity * bp.max_degree;

    search_free();
    if (PC.search_in_cpu) return;

    if (graph_type() == GraphType::GPU) {
        search_data.push_back(new Search_Data<Beam_Data>(Q, bp, GraphType::GPU));
    } else if (!MEM_MODE) {
        size_t hash_cap = 1 << 16;
        while (hash_cap < bp.beam_size * 5) hash_cap <<= 1;
        bp.hash_n = hash_cap;
        d_start_vec_idxs.resize(PC.search_start_nodes_num);
        search_data.push_back(new Search_Data<Beam_Data>(Q, bp, GraphType::CPU));
        search_data.push_back(new Search_Data<Beam_Data>(Q, bp, GraphType::CPU));
        // beam_data.resize((size_t)batch * bp.beam_size);
    } else {
        size_t hash_cap = 1 << 16;
        while (hash_cap < bp.beam_size * 5) hash_cap <<= 1;
        bp.hash_n = hash_cap;
        search_data1.push_back(new Search_Data<Beam_Data1>(Q, bp, GraphType::CPU));
        search_data1.push_back(new Search_Data<Beam_Data1>(Q, bp, GraphType::CPU));

        CUDA_CHECK(
            cudaMallocHost(&h_start_vecs, PC.search_start_nodes_num * bp.dim * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&h_beam_ids, SEARCH_BATCH * beam_capacity * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&h_slot_idxs, SEARCH_BATCH * beam_capacity * sizeof(int)));
        graph->search_prepare(beam_capacity);
        // CUDA_CHECK(cudaMallocHost(&h_slots, SEARCH_BATCH * beam_capacity * sizeof(Slot)));
        // CUDA_CHECK(cudaMallocHost(
        //     &h_nbr_vecs, SEARCH_BATCH * beam_capacity * bp.max_degree * bp.dim *
        //     sizeof(float)));
        // for (auto data : search_data1) {
        //     data->beam_data_.h_beam_ids = h_beam_ids;
        //     data->beam_data_.h_slots = h_slots;
        //     data->beam_data_.h_nbr_vecs = h_nbr_vecs;
        // }
    }
    // fo.print("search prepared: Q=" + TOS(Q) + ", beam_capacity=" + TOS(beam_capacity));
}

void GPUFuncs::search_free() {
    if (search_data.size()) {
        for (auto data : search_data) delete data;
        search_data.clear();
    }
    if (search_data1.size()) {
        for (auto data : search_data1) delete data;
        search_data1.clear();
    }

    if (h_start_vecs) CUDA_CHECK(cudaFreeHost(h_start_vecs));
    if (h_beam_ids) CUDA_CHECK(cudaFreeHost(h_beam_ids));
    if (h_slot_idxs) CUDA_CHECK(cudaFreeHost(h_slot_idxs));
    // if (h_nbr_vecs) CUDA_CHECK(cudaFreeHost(h_nbr_vecs));
}
}  // namespace efanna2e
