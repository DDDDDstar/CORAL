#include <cfloat>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>

#include "fileout.h"
#include "graph.cuh"
#include "utils.cuh"

namespace efanna2e {
__global__ void calc_query_dists_kernel(const float* __restrict__ queries,
                                        const int* __restrict__ qids,
                                        const float* __restrict__ nbr_vecs,
                                        Slot* __restrict__ slots, int batch, int capacity,
                                        int max_deg, int dim, DIST_METRIC m) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x,
              warp = (bid * tpb + tid) / warpSize, lane = tid % warpSize,
              total_warp = gridDim.x * tpb / warpSize;

    for (int i = warp; i < batch * capacity; i += total_warp) {
        Slot* slot = slots + i;
        const float* vecs = nbr_vecs + i * max_deg * dim;
        int qid = -1, deg = -1;
        if (!lane) {
            qid = qids[i / capacity];
            deg = slot->deg;
        }
        deg = __shfl_sync(0xffffffff, deg, 0);
        if (deg <= 0) continue;
        qid = __shfl_sync(0xffffffff, qid, 0);
        const float* query = queries + qid * dim;
        for (int j = lane; j < deg; j += warpSize)
            slot->dists[j] = calc_distance(query, vecs + j * dim, dim, m);
    }
}

__global__ void calc_query_dists_kernel(const float* __restrict__ queries,
                                        const int* __restrict__ qids,
                                        const int* __restrict__ qidxs,
                                        const float* __restrict__ nbr_vecs,
                                        Slot* __restrict__ slots, int N, int max_deg, int dim,
                                        DIST_METRIC m) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x,
              warp = (bid * tpb + tid) / warpSize, lane = tid % warpSize,
              total_warp = gridDim.x * tpb / warpSize;

    for (int i = warp; i < N; i += total_warp) {
        Slot* slot = slots + i;
        const float* vecs = nbr_vecs + i * max_deg * dim;
        int qid = -1, deg = -1;
        if (!lane) {
            qid = qids[qidxs[i]];
            deg = slot->deg;
        }
        deg = __shfl_sync(0xffffffff, deg, 0);
        if (deg <= 0) continue;
        qid = __shfl_sync(0xffffffff, qid, 0);
        const float* query = queries + qid * dim;
        for (int j = lane; j < deg; j += warpSize)
            slot->dists[j] = calc_distance(query, vecs + j * dim, dim, m);
    }
}

__global__ void calc_query_dists_kernel(const float* __restrict__ queries,
                                        const int* __restrict__ qidxs,
                                        const float* __restrict__ nbr_vecs,
                                        Slot* __restrict__ slots, float* __restrict__ dists2query,
                                        int N, int max_deg, int dim, DIST_METRIC m) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x,
              warp = (bid * tpb + tid) / warpSize, lane = tid % warpSize,
              total_warp = gridDim.x * tpb / warpSize;

    for (int i = warp; i < N; i += total_warp) {
        Slot* slot = slots + i;
        const float* vecs = nbr_vecs + i * max_deg * dim;
        int deg = -1;
        if (!lane) deg = slot->deg;
        deg = __shfl_sync(0xffffffff, deg, 0);
        if (deg <= 0) continue;
        const float* query = queries + qidxs[i] * dim;
        float* dists_out = dists2query + i * max_deg;
        for (int j = lane; j < deg; j += warpSize)
            dists_out[j] = calc_distance(query, vecs + j * dim, dim, m);
    }
}

__global__ void calc_query_dists_kernel(const float* __restrict__ queries,
                                        const float* __restrict__ nbr_vecs,
                                        Slot* __restrict__ slots, float* __restrict__ dists2query,
                                        int batch, int capacity, int max_deg, int dim,
                                        DIST_METRIC m) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x,
              warp = (bid * tpb + tid) / warpSize, lane = tid % warpSize,
              total_warp = gridDim.x * tpb / warpSize;

    for (int i = warp; i < batch * capacity; i += total_warp) {
        Slot* slot = slots + i;
        const float* vecs = nbr_vecs + i * max_deg * dim;
        int deg = -1;
        if (!lane) deg = slot->deg;
        deg = __shfl_sync(0xffffffff, deg, 0);
        const float* query = queries + (i / capacity) * dim;
        float* dists_out = dists2query + i * max_deg;
        for (int j = lane; j < deg; j += warpSize)
            dists_out[j] = calc_distance(query, vecs + j * dim, dim, m);
    }
}

__global__ void cal_knn_dists_kernel(const float* __restrict__ queries,
                                     const float* __restrict__ knns, float* __restrict__ dists_out,
                                     int n, int k, int dim, DIST_METRIC m) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x, total = n * k;

    for (int i = tid; i < total; i += blockDim.x * gridDim.x) {
        const int qi = i / k;  // 第几个 query
        dists_out[i] = calc_distance(queries + (size_t)qi * dim, knns + (size_t)i * dim, dim, m);
    }
}

void Graph::write_back_all_gpu_slots() {
    if (gtype == GraphType::GPU)
        CUDA_CHECK(cudaMemcpy(&cpu_slots[0], gpu_slots, total_node_size * sizeof(Slot),
                              cudaMemcpyDeviceToHost));
    else if (!MEM_MODE)
        gpu_cache->flush_all();
}

void Graph::write_data_cpu(const Slot* slots, int N) {
    for (int i = 0; i < N; i++) {
        const Slot* slot = slots + i;
        const int id = slot->node_id;
        assert(id >= 0);
        setBit(visited.data(), id);
        Slot* dst = &cpu_slots[id];
        const int new_deg = slot->deg, old_deg = dst->deg;
        if (new_deg <= 0) {
            fo.print("Graph::write_data_cpu new_deg <= 0: id=" + TOS(id) +
                     ", old_deg=" + TOS(old_deg) + ", new_deg=" + TOS(new_deg) + "\n");
            continue;
        }
        if (!old_deg) {
            noniso_num++;
            total_degree += new_deg;
        } else
            total_degree += new_deg - old_deg;
        std::memcpy(dst, slot, sizeof(Slot));
    }
}

void Graph::write_data_disk(const Slot* slots, int N) {
    for (int i = 0; i < N; i++) {
        const Slot* slot = slots + i;
        const int id = slot->node_id;
        assert(id >= 0);
        auto it = cpu_cache_dir.find(id);
        if (it == cpu_cache_dir.end()) continue;
        setBit(visited.data(), id);
        const int slot_idx = it->second;
        if (slot_states[slot_idx].ref()) {
            Slot* dst = &cpu_slots[slot_idx];
            const int new_deg = slot->deg, old_deg = dst->deg;
            assert(new_deg > 0);
            if (!old_deg) {
                noniso_num++;
                total_degree += new_deg;
            } else
                total_degree += new_deg - old_deg;
            std::memcpy(dst, slot, sizeof(Slot));
            slot_states[slot_idx].write();
        }
    }
}

void Graph::load_calc_vecs_dists_cpu(const float* d_queries, const int* node_ids, int batch, int k,
                                     float* d_vecs_out, float* d_dists_out, cudaStream_t st) {
    const int N = batch * k;
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        const int id = node_ids[i];
        assert(id >= 0);
        std::memcpy(h_vecs + (size_t)i * dim, h_base + (size_t)id * dim, sizeof(float) * dim);
    }
    CUDA_CHECK(
        cudaMemcpyAsync(d_vecs_out, h_vecs, N * dim * sizeof(float), cudaMemcpyHostToDevice, st));
    cal_knn_dists_kernel<<<2048, 512, 0, st>>>(d_queries, d_vecs_out, d_dists_out, batch, k, dim,
                                               metric);
}

void Graph::load_calc_vecs_dists_cpu(const float* d_queries, const int* node_ids,
                                     const int* h_offsets, const int* d_offsets, int batch,
                                     float* d_vecs_out, float* d_dists_out, cudaStream_t st) {
    const int N = batch * k;
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        const int id = node_ids[i];
        assert(id >= 0);
        std::memcpy(h_vecs + (size_t)i * dim, h_base + (size_t)id * dim, sizeof(float) * dim);
    }
    CUDA_CHECK(
        cudaMemcpyAsync(d_vecs_out, h_vecs, N * dim * sizeof(float), cudaMemcpyHostToDevice, st));
    cal_knn_dists_kernel<<<2048, 512, 0, st>>>(d_queries, d_vecs_out, d_dists_out, batch, k, dim,
                                               metric);
}

void Graph::load_calc_vecs_dists_disk(const float* d_queries, const int* node_ids, int batch,
                                      int k, float* d_vecs_out, float* d_dists_out,
                                      cudaStream_t st) {
    const int N = batch * k;
    std::unordered_set<int> miss;
    for (int i = 0; i < N; i++) {
        const int id = node_ids[i];
        assert(id >= 0);
        auto it = cpu_cache_vec_dir.find(id);
        if (it == cpu_cache_vec_dir.end())
            miss.insert(id);
        else
            vec_slot_states[it->second].load();
    }
    load_miss_vecs(miss);

    for (int i = 0; i < N; i++) {
        const int id = node_ids[i];
        auto it = cpu_cache_vec_dir.find(id);
        assert(it != cpu_cache_vec_dir.end());
        std::memcpy(h_vecs + (size_t)i * dim, cpu_vec_slots.data() + (size_t)(it->second) * dim,
                    sizeof(float) * dim);
        vec_slot_states[it->second].used();
    }

    CUDA_CHECK(
        cudaMemcpyAsync(d_vecs_out, h_vecs, N * dim * sizeof(float), cudaMemcpyHostToDevice, st));
    cal_knn_dists_kernel<<<2048, 512, 0, st>>>(d_queries, d_vecs_out, d_dists_out, batch, k, dim,
                                               metric);
}

void Graph::load_vecs_cpu(const int* node_ids, int N, float* vecs_out) {
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        const int id = node_ids[i];
        assert(id >= 0);
        std::memcpy(vecs_out + (size_t)i * dim, h_base + (size_t)id * dim, sizeof(float) * dim);
    }
}

void Graph::load_vecs_disk(const int* node_ids, int N, float* vecs_out) {
    std::unordered_set<int> miss;
    for (int i = 0; i < N; i++) {
        const int id = node_ids[i];
        assert(id >= 0);
        auto it = cpu_cache_vec_dir.find(id);
        if (it == cpu_cache_vec_dir.end())
            miss.insert(id);
        else
            vec_slot_states[it->second].load();
    }

    load_miss_vecs(miss);
    for (int i = 0; i < N; i++) {
        const int id = node_ids[i];
        auto it = cpu_cache_vec_dir.find(id);
        assert(it != cpu_cache_vec_dir.end());
        std::memcpy(vecs_out + (size_t)i * dim, cpu_vec_slots.data() + (size_t)(it->second) * dim,
                    sizeof(float) * dim);
        vec_slot_states[it->second].used();
    }
}

void Graph::calc_query_dists(const float* d_queries, const int* d_qids, int N, Slot* d_slots_out,
                             float* d_dists2query_out, cudaStream_t st) {
    CUDA_CHECK(
        cudaMemcpyAsync(d_slots_out, h_slots, N * sizeof(Slot), cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(d_nbr_vecs, h_nbr_vecs, N * max_degree * dim * sizeof(float),
                               cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(d_qidxs, h_qidxs, N * sizeof(int), cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaEventRecord(ev, st));
    if (d_qids != nullptr)
        calc_query_dists_kernel<<<blocks, tpb, 0, st>>>(d_queries, d_qids, d_qidxs, d_nbr_vecs,
                                                        d_slots_out, N, max_degree, dim, metric);
    else {
        assert(d_dists2query_out != nullptr);
        calc_query_dists_kernel<<<blocks, tpb, 0, st>>>(d_queries, d_qidxs, d_nbr_vecs,
                                                        d_slots_out, d_dists2query_out, N,
                                                        max_degree, dim, metric);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(ev));
}

void Graph::load_search_data_cpu(const int* beam_ids, const float* d_queries, const int* d_qids,
                                 int beam_batch, int capacity, Slot* d_slots_out, int* h_slot_idxs,
                                 float* d_dists2query_out, cudaStream_t st) {
    assert(cap >= capacity);
    const int BATCH = 2 * blocks * tpb / max_degree;

    int slot_idx = 0, d_slot_idx = 0;
    for (int i = 0; i < beam_batch; i++) {
        const int* ids = beam_ids + i * capacity;
        for (int j = 0; j < capacity; j++) {
            const int id = ids[j];
            if (id < 0) {
                h_slot_idxs[i * capacity + j] = -1;
                continue;
            }
            const int idx = slot_idx++;
            Slot& slot = h_slots[idx];
            h_slot_idxs[i * capacity + j] = d_slot_idx + idx;
            h_qidxs[idx] = i;
            slot = cpu_slots[id];
            assert(id == slot.node_id);
            float* vecs = h_nbr_vecs + idx * max_degree * dim;
            for (int d = 0; d < slot.deg; d++)
                std::memcpy(vecs + d * dim, h_base + (size_t)slot.nbrs[d] * dim,
                            sizeof(float) * dim);
            if (slot_idx == BATCH) {
                float* d_dists_ptr = d_dists2query_out
                                         ? (d_dists2query_out + (size_t)d_slot_idx * max_degree)
                                         : nullptr;
                calc_query_dists(d_queries, d_qids, BATCH, d_slots_out + d_slot_idx, d_dists_ptr,
                                 st);
                d_slot_idx += BATCH;
                slot_idx = 0;
            }
        }
    }
    if (slot_idx) {
        float* d_dists_ptr =
            d_dists2query_out ? (d_dists2query_out + (size_t)d_slot_idx * max_degree) : nullptr;
        calc_query_dists(d_queries, d_qids, slot_idx, d_slots_out + d_slot_idx, d_dists_ptr, st);
    }

    CUDA_CHECK(cudaStreamSynchronize(st));
}

void Graph::load_search_data_disk(const int* beam_ids, const float* d_queries, const int* d_qids,
                                  int beam_batch, int capacity, Slot* d_slots_out,
                                  int* h_slot_idxs, float* d_dists2query_out, cudaStream_t st) {
    assert(cap >= capacity);

    // 与 cpu_v2 保持一致：按“有效 slot 数”分批，而不是按 query 分批
    const int BATCH = 2 * blocks * tpb / max_degree;
    const int N = beam_batch * capacity;

    // 先把所有缺失的 slot 元数据补到 CPU cache
    std::unordered_set<int> miss, vec_miss;
    for (int i = 0; i < N; i++) {
        const int id = beam_ids[i];
        if (id < 0) continue;

        auto it = cpu_cache_dir.find(id);
        if (it == cpu_cache_dir.end())
            miss.insert(id);
        else
            slot_states[it->second].load();
    }
    load_miss_slots(miss);

    int slot_idx = 0;    // 当前 h_slots / h_nbr_vecs buffer 已填入多少个有效 slot
    int d_slot_idx = 0;  // 当前已经写入到 d_slots_out 的有效 slot 总数

    auto flush_batch = [&](int cur_slot_num) {
        if (cur_slot_num <= 0) return;

        // 先把当前批次缺失的邻居向量补齐到 CPU vec cache
        load_miss_vecs(vec_miss);
        vec_miss.clear();

        // 根据已经准备好的 h_slots，拷贝其邻居向量到 h_nbr_vecs
#pragma omp parallel for
        for (int idx = 0; idx < cur_slot_num; idx++) {
            const Slot& slot = h_slots[idx];
            float* vecs = h_nbr_vecs + (size_t)idx * max_degree * dim;
            for (int d = 0; d < slot.deg; d++) {
                const int nbr_id = slot.nbrs[d];
                auto it = cpu_cache_vec_dir.find(nbr_id);
                assert(it != cpu_cache_vec_dir.end());
                std::memcpy(vecs + (size_t)d * dim,
                            cpu_vec_slots.data() + (size_t)it->second * dim, sizeof(float) * dim);
                vec_slot_states[it->second].used();
            }
        }

        float* d_dists_ptr =
            d_dists2query_out ? (d_dists2query_out + (size_t)d_slot_idx * max_degree) : nullptr;
        calc_query_dists(d_queries, d_qids, cur_slot_num, d_slots_out + d_slot_idx, d_dists_ptr,
                         st);
        d_slot_idx += cur_slot_num;
    };

    for (int i = 0; i < beam_batch; i++) {
        const int* ids = beam_ids + i * capacity;
        for (int j = 0; j < capacity; j++) {
            const int id = ids[j];
            if (id < 0) {
                h_slot_idxs[i * capacity + j] = -1;
                continue;
            }
            auto it = cpu_cache_dir.find(id);
            assert(it != cpu_cache_dir.end());
            const int idx = slot_idx++;
            Slot& slot = h_slots[idx];

            // 原始 beam 位置 -> 压紧后的 d_slots_out 下标
            h_slot_idxs[i * capacity + j] = d_slot_idx + idx;

            // 记录该压紧 slot 属于哪个 query
            h_qidxs[idx] = i;

            slot = cpu_slots[it->second];
            assert(id == slot.node_id);
            slot_states[it->second].used();

            // 先只检查邻居向量是否在 cache 中；缺失的先记录，等 flush 时统一加载
            for (int d = 0; d < slot.deg; d++) {
                const int nbr_id = slot.nbrs[d];
                assert(nbr_id >= 0);

                auto vit = cpu_cache_vec_dir.find(nbr_id);
                if (vit == cpu_cache_vec_dir.end())
                    vec_miss.insert(nbr_id);
                else
                    vec_slot_states[vit->second].load();
            }

            // 当前压紧 buffer 满了，立刻处理
            if (slot_idx == BATCH) {
                flush_batch(BATCH);
                slot_idx = 0;
            }
        }
    }

    // 处理最后一个不满 BATCH 的尾批
    if (slot_idx) flush_batch(slot_idx);

    CUDA_CHECK(cudaStreamSynchronize(st));
}

void Graph::load_search_data_cpu(const int* beam_ids, const float* d_queries, const int* d_qids,
                                 int beam_batch, int capacity, Slot* d_slots_out,
                                 float* h_vecs_out, float* d_dists2query_out, cudaStream_t st,
                                 bool need_vec) {
    assert(cap >= capacity);
    const int BATCH = 2 * blocks * tpb / capacity / max_degree;
    for (int i = 0; i < beam_batch; i += BATCH) {
        const int batch = std::min(BATCH, beam_batch - i), slot_size = batch * capacity;
#pragma omp parallel for
        for (int j = 0; j < batch; j++) {
            const int* ids = beam_ids + (i + j) * capacity;
            float* vecs_out = need_vec ? h_vecs_out + (size_t)(i + j) * capacity * dim : nullptr;

            Slot* slots = h_slots + j * capacity;
            float* nbr_vecs = h_nbr_vecs + j * capacity * max_degree * dim;
            for (int l = 0; l < capacity; l++) {
                const int id = ids[l];
                Slot& slot = slots[l];
                if (id < 0) {
                    slot.deg = 0;
                    continue;
                }
                slot = cpu_slots[id];
                // slot.search_copy(cpu_slots[id]);
                if (need_vec)
                    std::memcpy(vecs_out + l * dim, h_base + (size_t)id * dim,
                                sizeof(float) * dim);
                assert(id == slot.node_id);
                float* vecs = nbr_vecs + l * max_degree * dim;
                for (int d = 0; d < slot.deg; d++)
                    std::memcpy(vecs + d * dim, h_base + (size_t)slot.nbrs[d] * dim,
                                sizeof(float) * dim);
            }
        }
        Slot* d_slots = d_slots_out + i * capacity;
        CUDA_CHECK(cudaMemcpyAsync(d_slots, h_slots, slot_size * sizeof(Slot),
                                   cudaMemcpyHostToDevice, st));
        CUDA_CHECK(cudaMemcpyAsync(d_nbr_vecs, h_nbr_vecs,
                                   slot_size * max_degree * dim * sizeof(float),
                                   cudaMemcpyHostToDevice, st));
        CUDA_CHECK(cudaStreamSynchronize(st));
        if (d_qids != nullptr)
            calc_query_dists_kernel<<<blocks, tpb, 0, st>>>(d_queries, d_qids + i, d_nbr_vecs,
                                                            d_slots, batch, capacity, max_degree,
                                                            dim, metric);
        else {
            assert(d_dists2query_out != nullptr);
            calc_query_dists_kernel<<<blocks, tpb, 0, st>>>(
                d_queries + i * dim, d_nbr_vecs, d_slots,
                d_dists2query_out + i * capacity * max_degree, batch, capacity, max_degree, dim,
                metric);
        }
        CUDA_CHECK(cudaGetLastError());
        // CUDA_CHECK(cudaStreamSynchronize(st));
    }
    // CUDA_CHECK(
    //     cudaMemcpy(h_slots, d_slots_out, capacity * sizeof(Slot), cudaMemcpyDeviceToDevice));
}

void Graph::load_search_data_disk(const int* beam_ids, const float* d_queries, const int* d_qids,
                                  int beam_batch, int capacity, Slot* d_slots_out,
                                  float* h_vecs_out, float* d_dists2query_out, cudaStream_t st,
                                  bool need_vec) {
    assert(cap >= capacity);
    const int BATCH = blocks * tpb / capacity / max_degree, N = beam_batch * capacity;

    std::unordered_set<int> miss, vec_miss;
    for (int i = 0; i < N; i++) {
        const int id = beam_ids[i];
        if (id < 0) continue;
        auto it = cpu_cache_dir.find(id);
        if (it == cpu_cache_dir.end())
            miss.insert(id);
        else
            slot_states[it->second].load();

        if (need_vec) {
            auto vit = cpu_cache_vec_dir.find(id);
            if (vit == cpu_cache_vec_dir.end())
                vec_miss.insert(id);
            else
                vec_slot_states[vit->second].load();
        }
    }

    load_miss_slots(miss);
    if (need_vec) {
        load_miss_vecs(vec_miss);
        vec_miss.clear();
    }

    for (int i = 0; i < beam_batch; i += BATCH) {
        const int batch = std::min(BATCH, beam_batch - i), slot_size = batch * capacity;
        Slot* d_slots = d_slots_out + i * capacity;
        for (int j = 0; j < batch; j++) {
            const int* ids = beam_ids + (i + j) * capacity;
            Slot* slots = h_slots + j * capacity;
            float* vecs_out = need_vec ? h_vecs_out + (size_t)(i + j) * capacity * dim : nullptr;
            for (int l = 0; l < capacity; l++) {
                const int id = ids[l];
                Slot& slot = slots[l];
                if (id < 0) {
                    slot.deg = 0;
                    continue;
                }
                auto it = cpu_cache_dir.find(id);
                assert(it != cpu_cache_dir.end());
                slot = cpu_slots[it->second];
                // slot.search_copy(cpu_slots[it->second]);
                slot_states[it->second].used();
                assert(id == slot.node_id);

                if (need_vec) {
                    auto vit = cpu_cache_vec_dir.find(id);
                    assert(vit != cpu_cache_vec_dir.end());
                    std::memcpy(vecs_out + l * dim,
                                cpu_vec_slots.data() + (size_t)vit->second * dim,
                                sizeof(float) * dim);
                }
                for (int d = 0; d < slot.deg; d++) {
                    const int nbr_id = slot.nbrs[d];
                    assert(nbr_id >= 0);
                    auto it = cpu_cache_vec_dir.find(nbr_id);
                    if (it == cpu_cache_vec_dir.end())
                        vec_miss.insert(nbr_id);
                    else
                        vec_slot_states[it->second].load();
                }
            }
        }
        load_miss_vecs(vec_miss);
        vec_miss.clear();
        CUDA_CHECK(cudaMemcpyAsync(d_slots, h_slots, slot_size * sizeof(Slot),
                                   cudaMemcpyHostToDevice, st));
        for (int j = 0; j < batch; j++) {
            const int* ids = beam_ids + (i + j) * capacity;
            const Slot* slots = h_slots + j * capacity;
            float* nbr_vecs = h_nbr_vecs + j * capacity * max_degree * dim;
            for (int l = 0; l < capacity; l++) {
                if (ids[l] < 0) continue;
                const Slot& slot = slots[l];
                float* vecs = nbr_vecs + l * max_degree * dim;
                for (int d = 0; d < slot.deg; d++) {
                    const int nbr_id = slot.nbrs[d];
                    auto it = cpu_cache_vec_dir.find(nbr_id);
                    assert(it != cpu_cache_vec_dir.end());
                    std::memcpy(vecs + d * dim, cpu_vec_slots.data() + (size_t)(it->second) * dim,
                                sizeof(float) * dim);
                    vec_slot_states[it->second].used();
                }
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(st));
        CUDA_CHECK(cudaMemcpy(d_nbr_vecs, h_nbr_vecs, slot_size * max_degree * dim * sizeof(float),
                              cudaMemcpyHostToDevice));
        if (d_qids != nullptr)
            calc_query_dists_kernel<<<blocks, tpb, 0, st>>>(d_queries, d_qids + i, d_nbr_vecs,
                                                            d_slots, batch, capacity, max_degree,
                                                            dim, metric);
        else {
            assert(d_dists2query_out != nullptr);
            calc_query_dists_kernel<<<blocks, tpb, 0, st>>>(
                d_queries + i * dim, d_nbr_vecs, d_slots,
                d_dists2query_out + i * capacity * max_degree, batch, capacity, max_degree, dim,
                metric);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(st));
    }
}

void Graph::load_data_cpu(const int* ids, int N, Slot* slots_out, float* vecs_out, bool need_vec,
                          bool need_nbr) {
    float* nbr_vecs_out = need_vec ? vecs_out + (size_t)N * dim : vecs_out;
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        Slot* slot = slots_out + i;
        const int id = ids[i];
        if (id < 0) continue;
        std::memcpy(slot, &cpu_slots[id], sizeof(Slot));
        assert(id == slot->node_id);
        if (need_vec)
            std::memcpy(vecs_out + (size_t)i * dim, h_base + (size_t)id * dim,
                        sizeof(float) * dim);

        if (need_nbr) {
            float* nbr_vecs_out_ = nbr_vecs_out + (size_t)i * max_degree * dim;
            for (int j = 0; j < slot->deg; j++) {
                const int nbr_id = slot->nbrs[j];
                assert(nbr_id >= 0);
                std::memcpy(nbr_vecs_out_ + j * dim, h_base + (size_t)nbr_id * dim,
                            sizeof(float) * dim);
            }
        }
    }
}

void Graph::load_data_disk(const int* ids, int N, Slot* slots_out, float* vecs_out, bool need_vec,
                           bool need_nbr) {
    float* nbr_vecs_out = need_vec ? vecs_out + (size_t)N * dim : vecs_out;

    std::unordered_set<int> miss, vec_miss;
    for (int i = 0; i < N; i++) {
        const int id = ids[i];
        if (id < 0) continue;
        auto it = cpu_cache_dir.find(id);
        if (it == cpu_cache_dir.end())
            miss.insert(id);
        else
            slot_states[it->second].load();
        if (need_vec) {
            auto vec_it = cpu_cache_vec_dir.find(id);
            if (vec_it == cpu_cache_vec_dir.end())
                vec_miss.insert(id);
            else
                vec_slot_states[vec_it->second].load();
        }
    }

    load_miss_slots(miss);

    for (int i = 0; i < N; i++) {
        Slot* slot = slots_out + i;
        const int id = ids[i];
        if (id < 0) continue;
        auto it = cpu_cache_dir.find(id);
        assert(it != cpu_cache_dir.end());
        std::memcpy(slot, &cpu_slots[it->second], sizeof(Slot));
        assert(id == slot->node_id);

        if (need_nbr)
            for (int j = 0; j < slot->deg; j++) {
                const int nbr_id = slot->nbrs[j];
                assert(nbr_id >= 0);
                auto it = cpu_cache_vec_dir.find(nbr_id);
                if (it == cpu_cache_vec_dir.end())
                    vec_miss.insert(nbr_id);
                else
                    vec_slot_states[it->second].load();
            }
    }

    load_miss_vecs(vec_miss);

    for (int i = 0; i < N; i++) {
        if (ids[i] < 0) continue;
        const Slot* slot = slots_out + i;
        if (need_vec) {
            auto it = cpu_cache_vec_dir.find(ids[i]);
            assert(it != cpu_cache_vec_dir.end());
            std::memcpy(vecs_out + (size_t)i * dim,
                        cpu_vec_slots.data() + (size_t)(it->second) * dim, sizeof(float) * dim);
            vec_slot_states[it->second].used();
        }
        if (need_nbr) {
            float* nbr_vecs_out_ = nbr_vecs_out + (size_t)i * max_degree * dim;
            for (int j = 0; j < slot->deg; j++) {
                const int nbr_id = slot->nbrs[j];
                auto it = cpu_cache_vec_dir.find(nbr_id);
                assert(it != cpu_cache_vec_dir.end());
                std::memcpy(nbr_vecs_out_ + (size_t)j * dim,
                            cpu_vec_slots.data() + (size_t)(it->second) * dim,
                            sizeof(float) * dim);
                vec_slot_states[it->second].used();
            }
        }
    }
}

void Graph::LoadRoarGraph(const char* filename) {
    // load graph to projection graph
    std::ifstream in(filename, std::ios::binary);
    uint32_t npts, projection_ep;
    in.read((char*)&projection_ep, sizeof(uint32_t));
    fo.print("Projection graph, ep: " + TOS(projection_ep));
    in.read((char*)&npts, sizeof(npts));
    std::vector<uint32_t> nbrs;
    float out_degree = 0.0;
    for (uint32_t i = 0; i < npts; i++) {
        uint32_t nbr_size;
        in.read((char*)&nbr_size, sizeof(nbr_size));
        nbrs.resize(nbr_size);
        in.read((char*)nbrs.data(), nbr_size * sizeof(uint32_t));
        out_degree += static_cast<float>(nbr_size);

        if (nbr_size > max_degree) {
            fo.print("Node " + TOS(i) + " has degree " + TOS(nbr_size) + " > max_degree");
            nbr_size = max_degree;
        }
        // fo.eprint("LoadRoarGraph nbr_size > max_degree: " + TOS(nbr_size));

        Slot& slot = cpu_slots[i];
        slot.node_id = i;
        slot.deg = nbr_size;
        for (int j = 0; j < nbr_size; j++) slot.nbrs[j] = (int)nbrs[j];
    }
    fo.print("Projection graph, avg_degree: " + TOS(out_degree / npts));
    in.close();

    cudaMemcpy(gpu_slots, &(cpu_slots[0]), npts * sizeof(Slot), cudaMemcpyHostToDevice);

    start_ids.clear();
    start_ids.push_back((int)projection_ep);
}

};  // namespace efanna2e

// __global__ void copy_last_data_kernel(const NBD* __restrict__ last_data,
//                                       NBD* __restrict__ load_data, int* __restrict__ ids,
//                                       const int* __restrict__ last_ids, int N, int max_degree,
//                                       int dim) {
//     const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
//     extern __shared__ int s_last_idx[];
//     int* s_deg = s_last_idx + 1;
//     for (int i = bid; i < N; i += gridDim.x) {
//         const int id = ids[i];
//         if (id < 0) continue;

//         if (!tid) *s_last_idx = -1;
//         __syncthreads();

//         for (int j = tid; j < N; j += tpb)
//             if (id == last_ids[j] || *s_last_idx >= 0) {
//                 *s_last_idx = j;
//                 break;
//             }
//         __syncthreads();

//         if (*s_last_idx < 0) continue;

//         if (!tid) {
//             *s_deg = last_data->degs[*s_last_idx];
//             ids[i] = -1;
//         }
//         __syncthreads();

//         const int in_offset = (*s_last_idx) * max_degree, out_offset = i * max_degree;
//         const int* nbrs_in = last_data->nbrs + in_offset;
//         const float *dists_in = last_data->dists + in_offset,
//                     *vecs_in = last_data->vecs + *s_last_idx * dim,
//                     *nbrs_vec_in = last_data->nbrs_vec + in_offset * dim;

//         int* nbrs_out = load_data->nbrs + out_offset;
//         float *dists_out = load_data->dists + out_offset, *vecs_out = load_data->vecs + i * dim,
//               *nbrs_vec_out = load_data->nbrs_vec + out_offset * dim;

//         for (int nbr_i = tid; nbr_i < *s_deg; nbr_i += tpb) {
//             nbrs_out[nbr_i] = nbrs_in[nbr_i];
//             dists_out[nbr_i] = dists_in[nbr_i];
//         }
//         for (int d = tid; d < dim; d += tpb) vecs_out[d] = vecs_in[d];

//         for (int nbr_d = tid; nbr_d < *s_deg * dim; nbr_d += tpb) {
//             nbrs_vec_out[nbr_d] = nbrs_vec_in[nbr_d];
//         }
//         __syncthreads();
//     }
// }

// kernel: apply updates to slots
// __global__ void update_data_kernel(Slot* __restrict__ slots, const int* __restrict__ slot_idxs,
//                                    const NBD* __restrict__ upd_data, int* delta_degree,
//                                    int* delta_noniso_num, int N, int max_degree) {
//     const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
//     extern __shared__ int s_deg[];
//     for (int slot_i = bid; slot_i < N; slot_i += gridDim.x) {
//         const int slot_idx = slot_idxs[slot_i];
//         if (slot_idx >= 0) {
//             // assert(slot_idx >= 0);
//             // assert(slot_idx < 5000000);
//             Slot* slot = slots + slot_idx;
//             const int* nbrs = upd_data->nbrs + slot_i * max_degree;
//             const float* dists = upd_data->dists + slot_i * max_degree;
//             if (tid == 0) {
//                 const int old_deg = slot->deg;
//                 slot->deg = *s_deg = upd_data->degs[slot_i];
//                 assert(*s_deg > 0);
//                 if (!old_deg) atomicAdd(delta_noniso_num, 1);
//                 atomicAdd(delta_degree, *s_deg - old_deg);
//             }
//             __syncthreads();

//             for (int nbr_i = tid; nbr_i < *s_deg; nbr_i += tpb) {
//                 assert(nbrs[nbr_i] >= 0);
//                 assert(nbr_i == 0 || nbrs[nbr_i] != nbrs[nbr_i - 1]);  // no duplicate nbrs
//                 slot->nbrs[nbr_i] = nbrs[nbr_i];
//                 slot->dists[nbr_i] = dists[nbr_i];
//             }
//             __syncthreads();
//         }
//     }
// }

// __global__ void load_data_kernel(const Slot* __restrict__ slots,
//                                  const float* __restrict__ vec_slots,
//                                  const int* __restrict__ slot_idxs,
//                                  const int* __restrict__ vec_slot_idxs,
//                                  const int* __restrict__ nbr_vec_slot_idxs,
//                                  NBD* __restrict__ load_data_out, int N, int dim, int
//                                  max_degree) {
//     const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
//     extern __shared__ int s_deg[];
//     for (int slot_i = bid; slot_i < N; slot_i += gridDim.x) {
//         const int slot_idx = slot_idxs[slot_i];
//         if (tid == 0)
//             load_data_out->degs[slot_i] = *s_deg = slot_idx < 0 ? -1 : slots[slot_idx].deg;
//         __syncthreads();

//         if (*s_deg >= 0) {
//             const Slot* slot = slots + slot_idx;
//             const float* vec = vec_slots + vec_slot_idxs[slot_i] * dim;
//             const int offset = slot_i * max_degree, *nbr_vec_slots = nbr_vec_slot_idxs + offset;
//             int* nbrs_out = load_data_out->nbrs + offset;
//             float *dists_out = load_data_out->dists + offset,
//                   *nbrs_vec_out = load_data_out->nbrs_vec + offset * dim;

//             for (int d = tid; d < dim; d += tpb) load_data_out->vecs[slot_i * dim + d] = vec[d];

//             for (int nbr_i = tid; nbr_i < *s_deg; nbr_i += tpb) {
//                 nbrs_out[nbr_i] = slot->nbrs[nbr_i];
//                 dists_out[nbr_i] = slot->dists[nbr_i];
//                 const float* nbr_vec = vec_slots + nbr_vec_slots[nbr_i] * dim;
//                 float* nbr_vec_out = nbrs_vec_out + nbr_i * dim;
//                 for (int d = 0; d < dim; d++) nbr_vec_out[d] = nbr_vec[d];
//             }
//         }
//     }
// }

// __global__ void load_data_kernel_v2(const Slot* __restrict__ slots,
//                                     const float* __restrict__ vec_slots,
//                                     const int* __restrict__ slot_idxs,
//                                     const int* __restrict__ vec_slot_idxs,
//                                     const int* __restrict__ nbr_vec_slot_idxs,
//                                     NBD* __restrict__ load_data_out, int N, int dim,
//                                     int max_degree) {
//     const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x, slot_i = bid * tpb + tid;
//     if (slot_i >= N) return;

//     const int slot_idx = slot_idxs[slot_i],
//               deg = load_data_out->degs[slot_i] = slot_idx < 0 ? -1 : slots[slot_idx].deg;
//     if (deg < 0) return;

//     const Slot* slot = slots + slot_idx;
//     // assert(vec_slot_idxs[slot_i] >= 0);
//     const float* vec = vec_slots + vec_slot_idxs[slot_i] * dim;
//     const int offset = slot_i * max_degree, *nbr_vec_slots = nbr_vec_slot_idxs + offset;
//     int* nbrs_out = load_data_out->nbrs + offset;
//     float *dists_out = load_data_out->dists + offset,
//           *nbrs_vec_out = load_data_out->nbrs_vec + offset * dim,
//           *vec_out = load_data_out->vecs + slot_i * dim;
//     for (int d = 0; d < dim; d++) vec_out[d] = vec[d];

//     for (int nbr_i = 0; nbr_i < deg; nbr_i++) {
//         nbrs_out[nbr_i] = slot->nbrs[nbr_i];
//         dists_out[nbr_i] = slot->dists[nbr_i];
//         assert(nbr_vec_slots[nbr_i] >= 0);
//         const float* nbr_vec = vec_slots + nbr_vec_slots[nbr_i] * dim;
//         float* nbr_vec_out = nbrs_vec_out + nbr_i * dim;
//         for (int d = 0; d < dim; d++) nbr_vec_out[d] = nbr_vec[d];
//     }
// }

// __global__ void gpu_load_data_kernel_v2(const Slot* __restrict__ slots,
//                                         const float* __restrict__ vec_slots,
//                                         const int* __restrict__ slot_idxs,
//                                         NBD* __restrict__ load_data_out, int N, int dim,
//                                         int max_degree) {
//     const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x, slot_i = bid * tpb + tid;
//     if (slot_i >= N) return;

//     const int slot_idx = slot_idxs[slot_i],
//               deg = load_data_out->degs[slot_i] = slot_idx < 0 ? -1 : slots[slot_idx].deg;
//     if (deg < 0) return;

//     const Slot* slot = slots + slot_idx;
//     const float* vec = vec_slots + slot_idx * dim;
//     const int offset = slot_i * max_degree;
//     int* nbrs_out = load_data_out->nbrs + offset;
//     float *dists_out = load_data_out->dists + offset,
//           *nbrs_vec_out = load_data_out->nbrs_vec + offset * dim,
//           *vec_out = load_data_out->vecs + slot_i * dim;
//     for (int d = 0; d < dim; d++) vec_out[d] = vec[d];

//     for (int nbr_i = 0; nbr_i < deg; nbr_i++) {
//         const int nbr_id = nbrs_out[nbr_i] = slot->nbrs[nbr_i];
//         assert(nbr_id >= 0);
//         dists_out[nbr_i] = slot->dists[nbr_i];
//         const float* nbr_vec = vec_slots + nbr_id * dim;
//         float* nbr_vec_out = nbrs_vec_out + nbr_i * dim;
//         for (int d = 0; d < dim; d++) nbr_vec_out[d] = nbr_vec[d];
//     }
// }

// void Graph::evict_random_slots(int need, cudaStream_t& stream) {
//     int step = 0, evict_num = 0;
//     std::mutex mtx;
//     std::condition_variable cv;
//     std::vector<int> evict_slots;
//     evict_slots.reserve(need);
//     while (evict_num < need) {
//         for (auto& kv : gpu_cache_dir) {
//             const int slot = kv.second, nid = kv.first;
//             if (gpu_slot_states[slot].get_wait_num() == 0) {
//                 if (gpu_slot_states[slot].is_dirty()) {
//                     const int cpu_slot = cpu_cache_dir[nid];
//                     CUDA_CHECK(cudaMemcpyAsync(&cpu_slots[cpu_slot], gpu_slots + slot,
//                                                sizeof(Slot), cudaMemcpyDeviceToHost, stream));
//                     cpu_slot_states[cpu_slot].mark_dirty();
//                 }

//                 gpu_slot_states[slot].free();
//                 // remove_from_bucket(gpu_slot_access_counts[slot].access_count, slot);
//                 // gpu_slot_access_counts[slot].access_count = 0;
//                 gpu_cache_dir.erase(nid);  // remove mapping now (avoid races)
//                 evict_slots.push_back(slot);
//                 if (++evict_num == need) break;
//             }
//         }

//         if (evict_num < need) {
//             fo.print("Step " + TOS(step++) + ": waiting for evicting slots(" + TOS(evict_num) +
//                      "/" + TOS(need) + ")");
//             std::unique_lock<std::mutex> lock(mtx);
//             cv.wait_for(lock, std::chrono::milliseconds(1000));
//         }
//     }

//     gpu_free_slots.batch_push(evict_slots);
// }

// int Graph::prefetch_slot(const int node_id, cudaStream_t stream) {
//     fo.eprint("TODO: prefetch_slot");
//     int slot;
//     if (cpu_cache_dir.find(node_id) != cpu_cache_dir.end())
//         slot = cpu_cache_dir[node_id];
//     else {
// #ifdef INFO_PRINT
//         fo.print("prefetch_slot node_id: " + TOS(node_id));
// #endif
//         if (cpu_free_slots.empty()) evict_random_cpu_slots(EVICT_BATCH, stream);

//         slot = cpu_free_slots.pop();
//         cpu_cache_dir[node_id] = slot;
//         DS.load_node_slot(node_id, cpu_slots[slot]);  // load node data from disk to CPU
//         cache
//     }
//     cpu_slot_states[slot].wait();
//     return slot;
// }

// void Graph::prefetch_slots(const std::vector<int>& node_ids, std::vector<int>& slots_res,
//                            cudaStream_t stream) {
//     fo.eprint("TODO: prefetch_slots");
//     const int N = node_ids.size();
//     slots_res.assign(N, -1);
//     std::vector<std::pair<int, int>> need_load_nids(N);  // <nid, idx in N>
//     std::atomic<int> need_num(0);
// #pragma omp parallel for
//     for (int i = 0; i < N; i++) {
//         const int node_id = node_ids[i];
//         if (node_id < 0) continue;
//         if (cpu_cache_dir.find(node_id) != cpu_cache_dir.end()) {
//             const int slot = slots_res[i] = cpu_cache_dir[node_id];
//             cpu_slot_states[slot].wait();
//         } else
//             need_load_nids[need_num++] = std::make_pair(node_id, i);
//     }

//     if (cpu_free_slots.size() < need_num) evict_random_cpu_slots(EVICT_BATCH, stream);
//     auto free_slots = cpu_free_slots.batch_pop(need_num);
// #pragma omp parallel for
//     for (int i = 0; i < need_num; i++) {
//         const int nid = need_load_nids[i].first,
//                   slot = slots_res[need_load_nids[i].second] = free_slots[i];
//         DS.load_node_slot(nid, cpu_slots[slot]);  // load node data from disk to CPU cache
//         cpu_slot_states[slot].wait();
//     }

//     for (int i = 0; i < need_num; i++) cpu_cache_dir[need_load_nids[i].first] =
//     free_slots[i];
// }

// void Graph::load_data_disk(int* d_slot_idxs, int* d_vec_slot_idxs, int* d_nbr_vec_slot_idxs,
//                            const std::vector<int>& node_ids, std::vector<int>& cpu_slots_res,
//                            std::vector<int>& gpu_slots_res, int N, cudaStream_t& stream,
//                            bool bucket_change) {
//     fo.eprint("TODO: load_data_disk");
//     std::vector<int> vec_slot_idxs, cpu_vec_slot_idxs, nbr_vec_slot_idxs(N * max_degree,
//     -1),
//         cpu_nbr_vec_slot_idxs;
//     gpu_slots_res.assign(N, -1);

//     // 1) check host-side directory for presence
//     prefetch_slots(node_ids, cpu_slots_res, stream);
//     std::vector<std::pair<int, int>> need_load_nids(N);  // <nid, idx in N>
//     std::vector<int> need_load_vec_nids, need_load_vec_nidxs,
//         need_load_nbr_vec_nids(N * max_degree), need_load_nbr_vec_nidxs(N * max_degree);
//     std::atomic<int> need_num(0), need_vec_num(0), need_nbr_vec_num(0);

//     if (d_vec_slot_idxs) {
//         vec_slot_idxs.resize(N, -1);
//         need_load_vec_nids.resize(N);
//         need_load_vec_nidxs.resize(N);
//     }

// #pragma omp parallel for
//     for (int i = 0; i < N; i++) {
//         const int node_id = node_ids[i];
//         if (node_id < 0) continue;
//         cudaStream_t stream;
//         cudaStreamCreate(&stream);
//         const int cpu_slot_i = cpu_slots_res[i];
//         if (gpu_cache_dir.find(node_id) != gpu_cache_dir.end()) {
//             const int slot = gpu_slots_res[i] = gpu_cache_dir[node_id];
//             if (gpu_slot_states[slot].is_dirty()) {
//                 CUDA_CHECK(cudaMemcpyAsync(&cpu_slots[cpu_slot_i], gpu_slots + slot,
//                 sizeof(Slot),
//                                            cudaMemcpyDeviceToHost, stream));
//                 gpu_slot_states[slot].clean();
//                 cpu_slot_states[cpu_slot_i].mark_dirty();
//             }
//             gpu_slot_states[slot].wait();
//         } else
//             need_load_nids[need_num++] = std::make_pair(node_id, i);

//         if (d_vec_slot_idxs) {
//             if (gpu_cache_vec_dir.find(node_id) != gpu_cache_vec_dir.end())
//                 vec_slot_idxs[i] = gpu_cache_vec_dir[node_id];
//             else {
//                 const int need_i = need_vec_num++;
//                 need_load_vec_nids[need_i] = node_id;
//                 need_load_vec_nidxs[need_i] = i;
//             }
//         }
//         cudaStreamSynchronize(stream);
//         cudaStreamDestroy(stream);
//         Slot* cpu_slot = &cpu_slots[cpu_slots_res[i]];
//         for (int j = 0; j < cpu_slot->deg; j++) {
//             const int nbrid = cpu_slot->nbrs[j], idx = i * max_degree + j;
//             if (gpu_cache_dir.find(nbrid) != gpu_cache_dir.end())
//                 nbr_vec_slot_idxs[idx] = gpu_cache_vec_dir[nbrid];
//             else {
//                 const int need_i = need_nbr_vec_num++;
//                 need_load_nbr_vec_nids[need_i] = nbrid;
//                 need_load_nbr_vec_nidxs[need_i] = idx;
//             }
//         }
//     }

//     int need_size = need_num - gpu_free_slots.size();
//     if (need_num > gpu_free_slots.size())
//         evict_random_slots(std::max(EVICT_BATCH, need_size), stream);
//     auto free_slots = gpu_free_slots.batch_pop(need_num);
//     // #pragma omp parallel for
//     for (int i = 0; i < need_num; i++) {
//         const int slot = free_slots[i];
//         CUDA_CHECK(cudaMemcpyAsync(gpu_slots + slot,
//                                    &cpu_slots[cpu_slots_res[need_load_nids[i].second]],
//                                    sizeof(Slot), cudaMemcpyHostToDevice, stream));
//         gpu_slot_states[slot].wait();
//         gpu_slots_res[need_load_nids[i].second] = slot;
//     }
//     for (int i = 0; i < need_num; i++) gpu_cache_dir[need_load_nids[i].first] =
//     free_slots[i];
//     // if (bucket_change)
//     //     for (int slot : gpu_slots_res) {
//     //         int old_count = gpu_slot_access_counts[slot].access_count++;
//     //         remove_from_bucket(old_count, slot);
//     //         insert_into_bucket(old_count + 1, slot);
//     //         change_bucket(slot);
//     //     }

//     prefetch_vecs(need_load_vec_nids, cpu_vec_slot_idxs);
//     prefetch_vecs(need_load_nbr_vec_nids, cpu_nbr_vec_slot_idxs);
//     need_size = need_vec_num + need_nbr_vec_num - gpu_free_vec_slots.size();
//     if (need_size > 0) evict_random_vec_slots(std::max(EVICT_BATCH, need_size));

//     auto free_vec_slots = gpu_free_vec_slots.batch_pop(need_vec_num);
//     // #pragma omp parallel for
//     for (int i = 0; i < need_vec_num; i++) {
//         const int nid = need_load_vec_nids[i], idx = need_load_vec_nidxs[i],
//                   slot = vec_slot_idxs[idx] = free_vec_slots[i];
//         CUDA_CHECK(cudaMemcpyAsync(gpu_vec_slots + slot * dim,
//                                    cpu_vec_slots.data() + cpu_vec_slot_idxs[idx] * dim,
//                                    sizeof(float) * dim, cudaMemcpyHostToDevice, stream));
//     }
//     for (int i = 0; i < need_vec_num; i++)
//         gpu_cache_vec_dir[need_load_vec_nids[i]] = free_vec_slots[i];

//     auto free_nbr_vec_slots = gpu_free_vec_slots.batch_pop(need_nbr_vec_num);
//     // #pragma omp parallel for
//     for (int i = 0; i < need_nbr_vec_num; i++) {
//         const int nid = need_load_nbr_vec_nids[i], idx = need_load_nbr_vec_nidxs[i],
//                   slot = nbr_vec_slot_idxs[idx] = free_nbr_vec_slots[i];
//         CUDA_CHECK(cudaMemcpyAsync(gpu_vec_slots + slot * dim,
//                                    cpu_vec_slots.data() + cpu_nbr_vec_slot_idxs[idx] * dim,
//                                    sizeof(float) * dim, cudaMemcpyHostToDevice, stream));
//     }
//     for (int i = 0; i < need_nbr_vec_num; i++)
//         gpu_cache_vec_dir[need_load_nbr_vec_nids[i]] = free_nbr_vec_slots[i];

//     CUDA_CHECK(cudaMemcpyAsync(d_slot_idxs, gpu_slots_res.data(), sizeof(int) * N,
//                                cudaMemcpyHostToDevice, stream));
//     if (d_vec_slot_idxs)
//         CUDA_CHECK(cudaMemcpyAsync(d_vec_slot_idxs, vec_slot_idxs.data(), sizeof(int) * N,
//                                    cudaMemcpyHostToDevice, stream));
//     CUDA_CHECK(cudaMemcpyAsync(d_nbr_vec_slot_idxs, nbr_vec_slot_idxs.data(),
//                                sizeof(int) * N * max_degree, cudaMemcpyHostToDevice,
//                                stream));
// }

// void Graph::load_data_cpu(int* d_slot_idxs, int* d_vec_slot_idxs, int* d_nbr_vec_slot_idxs,
//                           const std::vector<int>& node_ids, std::vector<int>& gpu_slots_res, int
//                           N, cudaStream_t& stream, bool bucket_change) {
//     std::vector<int> vec_slot_idxs, nbr_vec_slot_idxs(N * max_degree, -1);
//     std::vector<std::pair<int, int>> need_load_nids(N), need_load_vec_nids,
//         need_load_nbr_vec_nids(N * max_degree);  // <nid, idx in N>
//     gpu_slots_res.assign(N, -1);
//     if (d_vec_slot_idxs) {
//         vec_slot_idxs.resize(N, -1);
//         need_load_vec_nids.resize(N);
//     }
//     std::atomic<int> need_num(0), need_vec_num(0), need_nbr_vec_num(0);
// #pragma omp parallel for
//     for (int i = 0; i < N; i++) {
//         const int node_id = node_ids[i];
//         if (node_id < 0) continue;
//         cudaStream_t stream;
//         cudaStreamCreate(&stream);
//         if (gpu_cache_dir.find(node_id) != gpu_cache_dir.end()) {
//             const int slot = gpu_slots_res[i] = gpu_cache_dir[node_id];
//             if (gpu_slot_states[slot].is_dirty()) {
//                 CUDA_CHECK(cudaMemcpyAsync(&cpu_slots[node_id], gpu_slots + slot,
//                 sizeof(Slot),
//                                            cudaMemcpyDeviceToHost, stream));
//                 gpu_slot_states[slot].clean();
//                 cpu_slot_states[node_id].mark_dirty();
//             }
//             gpu_slot_states[slot].wait();
//         } else
//             need_load_nids[need_num++] = std::make_pair(node_id, i);
//         if (d_vec_slot_idxs) {
//             if (gpu_cache_vec_dir.find(node_id) != gpu_cache_vec_dir.end())
//                 vec_slot_idxs[i] = gpu_cache_vec_dir[node_id];
//             else
//                 need_load_vec_nids[need_vec_num++] = std::make_pair(node_id, i);
//         }
//         cudaStreamSynchronize(stream);
//         cudaStreamDestroy(stream);

//         Slot* cpu_slot = &cpu_slots[node_id];
//         for (int j = 0; j < cpu_slot->deg; j++) {
//             const int nbrid = cpu_slot->nbrs[j], idx = i * max_degree + j;
//             if (gpu_cache_dir.find(nbrid) != gpu_cache_dir.end())
//                 nbr_vec_slot_idxs[idx] = gpu_cache_vec_dir[nbrid];
//             else
//                 need_load_nbr_vec_nids[need_nbr_vec_num++] = std::make_pair(nbrid, idx);
//         }
//     }

//     int need_size = need_num - gpu_free_slots.size();
//     if (need_num > gpu_free_slots.size())
//         evict_random_slots(std::max(EVICT_BATCH, need_size), stream);
//     auto free_slots = gpu_free_slots.batch_pop(need_num);
//     for (int i = 0; i < need_num; i++) {
//         const int nid = need_load_nids[i].first, slot = gpu_cache_dir[nid] = free_slots[i];
//         CUDA_CHECK(cudaMemcpyAsync(gpu_slots + slot, &cpu_slots[nid], sizeof(Slot),
//                                    cudaMemcpyHostToDevice, stream));
//         gpu_slot_states[slot].wait();
//         gpu_slots_res[need_load_nids[i].second] = slot;
//     }
//     // if (bucket_change)
//     //     for (int slot : gpu_slots_res) {
//     // int old_count = gpu_slot_access_counts[slot].access_count++;
//     // remove_from_bucket(old_count, slot);
//     // insert_into_bucket(old_count + 1, slot);
//     //     change_bucket(slot);
//     // }

//     need_size = need_vec_num + need_nbr_vec_num - gpu_free_vec_slots.size();
//     if (need_size > 0) evict_random_vec_slots(std::max(EVICT_BATCH, need_size));
//     auto free_vec_slots = gpu_free_vec_slots.batch_pop(need_vec_num);
//     for (int i = 0; i < need_vec_num; i++) {
//         const int nid = need_load_vec_nids[i].first,
//                   slot = gpu_cache_vec_dir[nid] =
//                   vec_slot_idxs[need_load_vec_nids[i].second] =
//                       free_vec_slots[i];
//         CUDA_CHECK(cudaMemcpyAsync(gpu_vec_slots + slot * dim, cpu_vec_slots.data() + nid *
//         dim,
//                                    sizeof(float) * dim, cudaMemcpyHostToDevice, stream));
//     }

//     auto free_nbr_vec_slots = gpu_free_vec_slots.batch_pop(need_nbr_vec_num);
//     for (int i = 0; i < need_nbr_vec_num; i++) {
//         const int nid = need_load_nbr_vec_nids[i].first,
//                   slot = gpu_cache_vec_dir[nid] =
//                       nbr_vec_slot_idxs[need_load_nbr_vec_nids[i].second] =
//                       free_nbr_vec_slots[i];
//         CUDA_CHECK(cudaMemcpyAsync(gpu_vec_slots + slot * dim, cpu_vec_slots.data() + nid *
//         dim,
//                                    sizeof(float) * dim, cudaMemcpyHostToDevice, stream));
//     }

//     CUDA_CHECK(cudaMemcpyAsync(d_slot_idxs, gpu_slots_res.data(), sizeof(int) * N,
//                                cudaMemcpyHostToDevice, stream));
//     if (d_vec_slot_idxs)
//         CUDA_CHECK(cudaMemcpyAsync(d_vec_slot_idxs, vec_slot_idxs.data(), sizeof(int) * N,
//                                    cudaMemcpyHostToDevice, stream));
//     CUDA_CHECK(cudaMemcpyAsync(d_nbr_vec_slot_idxs, nbr_vec_slot_idxs.data(),
//                                sizeof(int) * N * max_degree, cudaMemcpyHostToDevice,
//                                stream));
// }

// void Graph::load_data_gpu(const std::vector<int>& node_ids, int N) {
// void Graph::load_data_gpu(int* d_nbr_vec_slot_idxs, const std::vector<int>& node_ids, int N,
//                           cudaStream_t& stream, bool bucket_change, std::vector<int>* degs)
//                           {
// std::vector<int> nbr_vec_slot_idxs(N * max_degree, -1);

// #pragma omp parallel for
//     for (int i = 0; i < N; i++) {
//         const int node_id = node_ids[i];
//         if (node_id < 0) continue;
//         // assert(node_id < total_node_size);
//         if (gpu_slot_states[node_id].is_dirty()) {
//             // cudaStream_t stream;
//             // CUDA_CHECK(cudaStreamCreate(&stream));
//             CUDA_CHECK(cudaMemcpyAsync(cpu_slots + node_id, gpu_slots + node_id,
//             sizeof(Slot),
//                                        cudaMemcpyDeviceToHost, stream));
//             CUDA_CHECK(cudaStreamSynchronize(stream));
//             // CUDA_CHECK(cudaStreamDestroy(stream));
//             gpu_slot_states[node_id].clean();
//             cpu_slot_states[node_id].mark_dirty();
//         }
//         Slot* cpu_slot = cpu_slots + node_id;
//         // memcpy(nbr_vec_slot_idxs.data() + i * max_degree, cpu_slot->nbrs,
//         //        sizeof(int) * cpu_slot->deg);
//         if (degs) (*degs)[i] = cpu_slot->deg;
//         for (int j = 0; j < cpu_slot->deg; j++) {
//             assert(cpu_slot->nbrs[j] >= 0);
//             nbr_vec_slot_idxs[i * max_degree + j] = cpu_slot->nbrs[j];
//         }
//     }
// if (bucket_change)
// for (int i = 0; i < N; i++) {
//     change_bucket(node_ids[i]);
// int old_count = gpu_slot_access_counts[slot].access_count++;
// remove_from_bucket(old_count, slot);
// insert_into_bucket(old_count + 1, slot);
// }

// CUDA_CHECK(cudaMemcpyAsync(d_nbr_vec_slot_idxs, nbr_vec_slot_idxs.data(),
//                            sizeof(int) * N * max_degree, cudaMemcpyHostToDevice, stream));
// }

// void Graph::load_data(const std::vector<int>& node_ids, std::vector<int>& cpu_slots_res,
//                       std::vector<int>& gpu_slots_res, NBD* d_load_data, int N,
//                       cudaStream_t& stream, bool bucket_change) {
//     auto e = gpu_record_time_start(stream);
//     thrust::device_vector<int> d_slot_idxs(N), d_vec_slot_idxs, d_nbr_vec_slot_idxs;
//     std::unique_lock<std::shared_mutex> lock(slot_mutex);
//     switch (gtype) {
//         case GraphType::GPU:
//             CUDA_CHECK(cudaMemcpyAsync(d_slot_idxs.data().get(), node_ids.data(), sizeof(int) *
//             N,
//                                        cudaMemcpyHostToDevice, stream));
//             // CUDA_CHECK(cudaMemcpyAsync(d_vec_slot_idxs.data().get(), node_ids.data(),
//             //                            sizeof(int) * N, cudaMemcpyHostToDevice, stream));
//             gpu_slots_res = node_ids;
//             load_data_gpu(node_ids, N);
//             // load_data_gpu(d_nbr_vec_slot_idxs.data().get(), node_ids, N, stream,
//             bucket_change); break;
//         case GraphType::CPU:
//             d_vec_slot_idxs.resize(N);
//             d_nbr_vec_slot_idxs.resize(N * max_degree);
//             load_data_cpu(d_slot_idxs.data().get(), d_vec_slot_idxs.data().get(),
//                           d_nbr_vec_slot_idxs.data().get(), node_ids, gpu_slots_res, N, stream,
//                           bucket_change);
//             break;
//         default:
//             d_vec_slot_idxs.resize(N);
//             d_nbr_vec_slot_idxs.resize(N * max_degree);
//             load_data_disk(d_slot_idxs.data().get(), d_vec_slot_idxs.data().get(),
//                            d_nbr_vec_slot_idxs.data().get(), node_ids, cpu_slots_res,
//                            gpu_slots_res, N, stream, bucket_change);
//             break;
//     }
//     float pre_time;
//     e = gpu_record_time_reset(e, stream, pre_time, "load_data_pre");

//     // int blocks = max_pow2_le(N);
//     // if (blocks <= 1024)
//     //     load_data_kernel<<<blocks, 32, sizeof(int), stream>>>(
//     //         gpu_slots, gpu_vec_slots, d_slot_idxs.data().get(), d_vec_slot_idxs.data().get(),
//     //         d_nbr_vec_slot_idxs.data().get(), d_load_data, N, dim, max_degree);
//     // else
//     if (gtype == GraphType::GPU)
//         gpu_load_data_kernel_v2<<<(N + 256 - 1) / 256, 256, 0, stream>>>(
//             gpu_slots, gpu_vec_slots, d_slot_idxs.data().get(), d_load_data, N, dim,
//             max_degree);
//     else
//         load_data_kernel_v2<<<(N + 256 - 1) / 256, 256, 0, stream>>>(
//             gpu_slots, gpu_vec_slots, d_slot_idxs.data().get(), d_vec_slot_idxs.data().get(),
//             d_nbr_vec_slot_idxs.data().get(), d_load_data, N, dim, max_degree);

//     // CUDA_CHECK(cudaStreamSynchronize(stream));
//     float load_time = gpu_record_time_stop(e, stream, "load_data_kernel");
// #ifdef INFO_PRINT
//     // fo.print("load data gpu(" + TOS(pre_time) + ", " + TOS(load_time) + "s): N-" + TOS(N));
// #endif
// }

// void Graph::update_graph_data(const std::vector<int>& gpu_slot_idxs, const NBD* upd_data, int N,
//                               cudaStream_t& stream) {
//     fo.eprint("No use: update_graph_data");
//     thrust::device_vector<int> d_slot_idxs(N), d_delta_degree(1, 0), d_delta_noniso_num(1,
//     0); int delta_degree, delta_noniso_num;

//     // assert(gpu_slot_idxs.size() >= N);
//     CUDA_CHECK(cudaMemcpyAsync(d_slot_idxs.data().get(), gpu_slot_idxs.data(), sizeof(int) *
//     N,
//                                cudaMemcpyHostToDevice, stream));

//     std::unique_lock<std::shared_mutex> lock(slot_mutex);

//     update_data_kernel<<<1024, 32, sizeof(int), stream>>>(
//         gpu_slots, d_slot_idxs.data().get(), upd_data, d_delta_degree.data().get(),
//         d_delta_noniso_num.data().get(), N, max_degree);

//     CUDA_CHECK(cudaMemcpyAsync(&delta_degree, d_delta_degree.data().get(), sizeof(int),
//                                cudaMemcpyDeviceToHost, stream));
//     CUDA_CHECK(cudaMemcpyAsync(&delta_noniso_num, d_delta_noniso_num.data().get(),
//     sizeof(int),
//                                cudaMemcpyDeviceToHost, stream));
//     CUDA_CHECK(cudaStreamSynchronize(stream));

//     for (int slot : gpu_slot_idxs) gpu_slot_states[slot].mark_dirty();

//     total_degree.fetch_add(delta_degree);
//     noniso_num.fetch_add(delta_noniso_num);

// #ifdef INFO_PRINT
//     // static int count = 0, interval = 1;
//     // if (++count == interval) {
//     //     count = 0;
//     // fo.print("Update graph: avg degree-" + TOS(1.0 * total_degree.load() /
//     total_node_size)
//     // +
//     //          ", noniso_num-" + TOS(noniso_num.load()));
//     // fo.print("Update graph: delta_degree-" + TOS(delta_degree) + ", delta_noniso_num-" +
//     //          TOS(delta_noniso_num) + ", avg degree-" +
//     //          TOS(1.0 * total_degree.load() / total_node_size) + ", noniso_num-" +
//     //          TOS(noniso_num.load()));
//     // }
// #endif
// }

// Node_Batch_Data::Node_Batch_Data(int n, int max_degree, int dim, cudaStream_t& stream) : n(n) {
//     CUDA_CHECK(cudaMallocAsync(&nbrs, n * max_degree * sizeof(int), stream));
//     CUDA_CHECK(cudaMallocAsync(&degs, n * sizeof(int), stream));
//     CUDA_CHECK(cudaMallocAsync(&dists, n * max_degree * sizeof(float), stream));
//     CUDA_CHECK(cudaMallocAsync(&nbrs_vec, n * max_degree * dim * sizeof(float), stream));
//     CUDA_CHECK(cudaMallocAsync(&vecs, n * dim * sizeof(float), stream));

//     CUDA_CHECK(cudaMemsetAsync(nbrs, -1, n * max_degree * sizeof(int), stream));
//     CUDA_CHECK(cudaMemsetAsync(degs, 0, n * sizeof(int), stream));
// }

// Node_Batch_Data::Node_Batch_Data(int n, int max_degree, cudaStream_t& stream) : n(n) {
//     CUDA_CHECK(cudaMallocAsync(&nbrs, n * max_degree * sizeof(int), stream));
//     CUDA_CHECK(cudaMallocAsync(&degs, n * sizeof(int), stream));
//     CUDA_CHECK(cudaMallocAsync(&dists_to_query, n * max_degree * sizeof(float), stream));
//     // CUDA_CHECK(cudaMallocAsync(&vecs, n * dim * sizeof(float), stream));

//     CUDA_CHECK(cudaMemsetAsync(nbrs, -1, n * max_degree * sizeof(int), stream));
//     CUDA_CHECK(cudaMemsetAsync(degs, 0, n * sizeof(int), stream));
// }

// void Node_Batch_Data::copy(int n, Node_Batch_Data* other, int max_degree, int dim,
//                            cudaStream_t& stream) {
//     CUDA_CHECK(cudaMemcpyAsync(nbrs, other->nbrs, n * max_degree * sizeof(int),
//                                cudaMemcpyDeviceToDevice, stream));
//     CUDA_CHECK(
//         cudaMemcpyAsync(degs, other->degs, n * sizeof(int), cudaMemcpyDeviceToDevice, stream));
//     CUDA_CHECK(cudaMemcpyAsync(dists, other->dists, n * max_degree * sizeof(float),
//                                cudaMemcpyDeviceToDevice, stream));
//     CUDA_CHECK(cudaMemcpyAsync(nbrs_vec, other->nbrs_vec, n * max_degree * dim * sizeof(float),
//                                cudaMemcpyDeviceToDevice, stream));
// }

// Node_Batch_Data::~Node_Batch_Data() {
//     CUDA_CHECK(cudaFree(nbrs));
//     CUDA_CHECK(cudaFree(degs));
//     if (dists) CUDA_CHECK(cudaFree(dists));
//     if (dists_to_query) CUDA_CHECK(cudaFree(dists_to_query));
//     if (nbrs_vec) CUDA_CHECK(cudaFree(nbrs_vec));
//     if (vecs) CUDA_CHECK(cudaFree(vecs));
// }