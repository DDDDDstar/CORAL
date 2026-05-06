
#include <omp.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "efanna2e/distance.h"
#include "fileout.h"
#include "gpufuncs.cuh"
#include "graph.cuh"
#include "utils.cuh"

namespace efanna2e {

Test_Result GPUFuncs::search_in_cpu(const float* d_queries, int* ids_res, float* dists_res, int Q,
                                    int k, std::vector<int>& start_node_ids) {
    if (k <= 0) k = bp.k;
    const int M = PC.search_start_nodes_num;

    // 先将 queries 从 GPU 拷回 CPU
    std::vector<float> queries((size_t)Q * bp.dim);
    cudaMemcpy(queries.data(), d_queries, (size_t)Q * bp.dim * sizeof(float),
               cudaMemcpyDeviceToHost);

    std::vector<int> ids((size_t)Q * k, -1);
    std::vector<float> dists;
    if (dists_res != nullptr) dists.assign((size_t)Q * k, std::numeric_limits<float>::infinity());

    // 只读共享数据
    const auto& slots = graph->get_cpu_slots();

    // 每个线程一份 visited
    const int num_threads = std::max(1, omp_get_max_threads());

    // visited_buffers[tid][node_id]
    std::vector<std::vector<uint8_t>> visited_buffers(num_threads,
                                                      std::vector<uint8_t>(bp.base_n, 0));

    // 每个线程一个独立 tag
    std::vector<uint8_t> visited_tags(num_threads, 1);

    size_t total_hop = 0;
    float time;
    Distance* distance_ = new DistanceL2();

#pragma omp parallel reduction(+ : total_hop)
    {
        const int tid = omp_get_thread_num();
        std::vector<uint8_t>& visited = visited_buffers[tid];
        uint8_t& visited_tag = visited_tags[tid];

        auto s = now_time();

#pragma omp for schedule(dynamic, 1)
        for (int qi = 0; qi < Q; qi++) {
            NeighborPriorityQueue search_queue(bp.beam_capacity);
            const float* query = queries.data() + (size_t)qi * bp.dim;

            // tag 递增；若溢出则清空当前线程自己的 visited
            if (++visited_tag == 0) {
                std::memset(visited.data(), 0, (size_t)bp.base_n);
                visited_tag = 1;
            }

            // 初始化起点
            for (int i = 0; i < M; i++) {
                const int id = start_node_ids[i];
                const float dist = distance_->compare(h_base + (size_t)id * bp.dim, query, bp.dim);
                search_queue.insert({id, dist, false});
                visited[id] = visited_tag;
            }

            uint32_t hops = M;

            while (search_queue.has_unexpanded_node()) {
                auto cur = search_queue.closest_unexpanded();
                const int cur_id = cur.id;
                const Slot& slot = slots[cur_id];

                ++hops;

                for (int i = 0; i < slot.deg; i++) {
                    const int nbr_id = slot.nbrs[i];
                    if (visited[nbr_id] == visited_tag) continue;

                    const float dist =
                        distance_->compare(h_base + (size_t)nbr_id * bp.dim, query, bp.dim);
                    search_queue.insert({nbr_id, dist, false});
                    visited[nbr_id] = visited_tag;
                }
            }

            total_hop += hops;

            const int topk = std::min<int>(k, search_queue.size());
            for (int i = 0; i < topk; i++) {
                ids[(size_t)qi * k + i] = search_queue[i].id;
                if (!dists.empty()) dists[(size_t)qi * k + i] = search_queue[i].distance;
            }
        }

        time = time_diff(s);
    }

    cudaMemcpy(ids_res, ids.data(), (size_t)Q * k * sizeof(int), cudaMemcpyHostToDevice);
    if (dists_res != nullptr)
        cudaMemcpy(dists_res, dists.data(), (size_t)Q * k * sizeof(float), cudaMemcpyHostToDevice);

    delete distance_;
    return Test_Result(-1, total_hop, time);
}

Test_Result GPUFuncs::search_in_cpu_for_disk(const float* d_queries, int* ids_res,
                                             float* dists_res, int Q, int k,
                                             std::vector<int>& start_node_ids) {
    if (k <= 0) k = bp.k;
    const int M = PC.search_start_nodes_num;

    // 先将 queries 从 GPU 拷回 CPU
    std::vector<float> queries((size_t)Q * bp.dim);
    cudaMemcpy(queries.data(), d_queries, (size_t)Q * bp.dim * sizeof(float),
               cudaMemcpyDeviceToHost);

    std::vector<int> ids((size_t)Q * k, -1);
    std::vector<float> dists;
    if (dists_res != nullptr) dists.assign((size_t)Q * k, std::numeric_limits<float>::infinity());

    // 只读共享数据
    std::vector<float> start_vecs(M * bp.dim);
    graph->load_vecs(start_node_ids.data(), M, start_vecs.data());

    // 每个线程一份 visited
    const int num_threads = std::max(1, omp_get_max_threads());

    // visited_buffers[tid][node_id]
    std::vector<std::vector<uint8_t>> visited_buffers(num_threads,
                                                      std::vector<uint8_t>(bp.base_n, 0));

    // 每个线程一个独立 tag
    std::vector<uint8_t> visited_tags(num_threads, 1);

    size_t total_hop = 0;
    Distance* distance_ = new DistanceL2();
    auto s = now_time();

#pragma omp parallel reduction(+ : total_hop)
    {
        const int tid = omp_get_thread_num();
        std::vector<uint8_t>& visited = visited_buffers[tid];
        uint8_t& visited_tag = visited_tags[tid];

#pragma omp for schedule(dynamic, 1)
        for (int qi = 0; qi < Q; qi++) {
            NeighborPriorityQueue search_queue(bp.beam_capacity);
            const float* query = queries.data() + (size_t)qi * bp.dim;

            // tag 递增；若溢出则清空当前线程自己的 visited
            if (++visited_tag == 0) {
                std::memset(visited.data(), 0, (size_t)bp.base_n);
                visited_tag = 1;
            }

            // 初始化起点
            for (int i = 0; i < M; i++) {
                const int id = start_node_ids[i];
                const float dist =
                    distance_->compare(start_vecs.data() + (size_t)i * bp.dim, query, bp.dim);
                search_queue.insert({id, dist, false});
                visited[id] = visited_tag;
            }

            uint32_t hops = M;

            while (search_queue.has_unexpanded_node()) {
                auto cur = search_queue.closest_unexpanded();
                const int cur_id = cur.id;
                const Slot& slot = graph->load_data_in_cpu(cur_id);

                ++hops;

                for (int i = 0; i < slot.deg; i++) {
                    const int nbr_id = slot.nbrs[i];
                    if (visited[nbr_id] == visited_tag) continue;

                    const float dist =
                        distance_->compare(graph->load_vec_in_cpu(nbr_id), query, bp.dim);
                    graph->vec_unlock(nbr_id);
                    search_queue.insert({nbr_id, dist, false});
                    visited[nbr_id] = visited_tag;
                }
                graph->slot_unlock(cur_id);
            }

            total_hop += hops;

            const int topk = std::min<int>(k, search_queue.size());
            for (int i = 0; i < topk; i++) {
                ids[(size_t)qi * k + i] = search_queue[i].id;
                if (!dists.empty()) dists[(size_t)qi * k + i] = search_queue[i].distance;
            }
        }
    }

    cudaMemcpy(ids_res, ids.data(), (size_t)Q * k * sizeof(int), cudaMemcpyHostToDevice);
    if (dists_res != nullptr)
        cudaMemcpy(dists_res, dists.data(), (size_t)Q * k * sizeof(float), cudaMemcpyHostToDevice);

    const float time = time_diff(s);
    delete distance_;
    return Test_Result(-1, total_hop, time);
}
}  // namespace efanna2e
