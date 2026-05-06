#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "fileout.h"
#include "graph.cuh"
#include "utils.cuh"

namespace efanna2e {

// 原子 CAS: 若 *addr == expected，则写入 desired，返回 true
static inline bool cas_int(int* addr, int expected, int desired) {
#if defined(__GNUG__) || defined(__clang__)
    return __sync_bool_compare_and_swap(addr, expected, desired);
#else
    // 非 GNU/Clang 环境可切换到 atomic_ref<int> (C++20)
    if (*addr == expected) {
        *addr = desired;
        return true;
    }
    return false;
#endif
}

// 对 visit_tag 做“未访问 -> 当前 round”的原子抢占
static inline bool try_visit(int* visit_tag_addr, int round) {
#if defined(__GNUG__) || defined(__clang__)
    return __sync_bool_compare_and_swap(visit_tag_addr, 0, round) ||
           __sync_bool_compare_and_swap(visit_tag_addr, round - 1, round);  // 不会用到，仅保守占位
#else
    if (*visit_tag_addr == 0) {
        *visit_tag_addr = round;
        return true;
    }
    return false;
#endif
}

// 更严格版本：只允许从 0 -> round
static inline bool try_visit_zero_to_round(int* visit_tag_addr, int round) {
#if defined(__GNUG__) || defined(__clang__)
    return __sync_bool_compare_and_swap(visit_tag_addr, 0, round);
#else
    if (*visit_tag_addr == 0) {
        *visit_tag_addr = round;
        return true;
    }
    return false;
#endif
}

void Graph::bfs_cpu(int start) {
    const int N = total_node_size;
    if (ws->N != N) ws->reset(N);

    const int round = ws->next_round();

    // 初始化本轮 BFS 的起点
    ws->visit_tag[start] = round;
    ws->hops[start] = 0;

    int frontier_size = 1;
    ws->frontier[0] = start;
    int level = 1;

    while (frontier_size > 0) {
        const int thread_count = omp_get_max_threads();

#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            auto& local = ws->local_bufs[tid];
            local.clear();

            // 经验 reserve，避免反复扩容
            if (local.capacity() < 2048) local.reserve(2048);

#pragma omp for schedule(dynamic, 64)
            for (int fi = 0; fi < frontier_size; ++fi) {
                const int u = ws->frontier[fi];
                const Slot& slot = cpu_slots[u];

                for (int j = 0; j < slot.deg; ++j) {
                    const int v = slot.nbrs[j];
                    if (v < 0) continue;

                    // 仅当本轮还未访问过时，抢占成功
                    if (try_visit_zero_to_round(&ws->visit_tag[v], round)) {
                        ws->hops[v] = level;
                        local.push_back(v);
                    }
                }
            }

            ws->local_sizes[tid] = static_cast<int>(local.size());
        }

        // prefix sum
        int next_size = 0;
        for (int t = 0; t < thread_count; ++t) {
            ws->local_offsets[t] = next_size;
            next_size += ws->local_sizes[t];
        }

#pragma omp parallel for schedule(static)
        for (int t = 0; t < thread_count; ++t) {
            const auto& local = ws->local_bufs[t];
            std::copy(local.begin(), local.end(),
                      ws->next_frontier.begin() + ws->local_offsets[t]);
        }

        frontier_size = next_size;
        if (frontier_size == 0) break;

        std::swap(ws->frontier, ws->next_frontier);
        ++level;
    }
}

void Graph::update_min_hops_cpu() {
    const int N = total_node_size;
    const int round = ws->bfs_round;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        if (ws->visit_tag[i] == round) {
            const int h = ws->hops[i];
            int& mh = ws->min_hops[i];
            if (mh < 0 || h < mh) mh = h;
        } else {
            if (ws->min_hops[i] == INT_MAX) ws->min_hops[i] = -1;
        }
    }
}

int Graph::argmax_min_hops_cpu() {
    const int N = total_node_size;

    int global_idx = -1;
    int global_val = INT_MIN;

#pragma omp parallel
    {
        int local_idx = -1;
        int local_val = INT_MIN;

#pragma omp for nowait schedule(static)
        for (int i = 0; i < N; ++i) {
            if (ws->chosen[i]) continue;

            const int v = ws->min_hops[i];
            if (v > local_val || (v == local_val && (local_idx < 0 || i < local_idx))) {
                local_val = v;
                local_idx = i;
            }
        }

#pragma omp critical
        {
            if (local_idx >= 0 &&
                (local_val > global_val ||
                 (local_val == global_val && (global_idx < 0 || local_idx < global_idx)))) {
                global_val = local_val;
                global_idx = local_idx;
            }
        }
    }

    return global_idx;
}

void Graph::FPS_cpu(std::vector<int>& selected, int M) {
    auto ts = std::chrono::high_resolution_clock::now();

    const int N = total_node_size;
    if (N <= 0 || M <= 0) return;
    if ((int)selected.size() >= M) return;

    // 建议你把它做成 Graph 成员缓存，这里先写成本地 static/局部也行
    if (ws == nullptr) ws = new FPSWorkspaceCPU(N);
    ws->clear_fps_state();

    int start = -1;

    // 如果传入的 selected 已有内容，就先标记 chosen，并重建 min_hops
    if (!selected.empty()) {
        for (int x : selected) {
            if (x >= 0 && x < N) ws->chosen[x] = 1;
        }

        // 重新根据已有 selected 建 min_hops
        for (int x : selected) {
            if (x < 0 || x >= N) continue;
            bfs_cpu(x);
            update_min_hops_cpu();
        }

        start = argmax_min_hops_cpu();
        if (start < 0) return;
    } else {
        fo.print("Generate new start nodes (CPU best)...");

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(0, N - 1);

        while (true) {
            start = dis(gen);
            if (cpu_slots[start].deg > 0) break;
        }

        selected.reserve(M);
        selected.push_back(start);
        ws->chosen[start] = 1;

        bfs_cpu(start);
        update_min_hops_cpu();
    }

    while ((int)selected.size() < M) {
        start = argmax_min_hops_cpu();
        if (start < 0) break;

        selected.push_back(start);
        ws->chosen[start] = 1;

        bfs_cpu(start);
        update_min_hops_cpu();
    }

    const float time = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::high_resolution_clock::now() - ts)
                           .count() /
                       1000.0f;

    fo.print("FPS_cpu(" + TOS(time) + "s): M=" + TOS(M));
}
}  // namespace efanna2e