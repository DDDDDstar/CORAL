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

__device__ __forceinline__ void warpTopL(int& lane_id, float& lane_d, int* top_ids,
                                         float* top_dists, int lane, int L, BP bp) {
    bool conti = true;
#pragma unroll
    for (int i = 0; i < L; ++i) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float d2 = __shfl_down_sync(0xffffffff, lane_d, off);
            const int id2 = __shfl_down_sync(0xffffffff, lane_id, off);
            if (id2 >= 0 && compare_dist(lane_d, d2, bp)) {
                lane_d = d2;
                lane_id = id2;
            }
        }
        if (lane == 0) {
            conti = false;
            if (lane_id >= 0) {
                if (compare_dist(top_dists[0], lane_d, bp)) {
                    conti = true;
                    top_ids[0] = lane_id;
                    top_dists[0] = lane_d;
                    heap_sift_down(top_ids, top_dists, 0, L, bp);
                }
            }
        }
        if (!__shfl_sync(0xffffffff, conti, 0)) break;
        if (lane_id == __shfl_sync(0xffffffff, lane_id, 0)) lane_id = -1;
    }
}

__device__ __forceinline__ int beam_insert(int* hash_beam, int key, size_t cap) {
    assert(key >= 0);
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

int hash_exists(int* hash_beam, int key, size_t cap) {
    for (int i = 0; i < cap; ++i) {  // linear probing
        const int val = hash_beam[(key + i) % cap];
        if (val == -1) return 0;   // 不存在
        if (val == key) return 1;  // 存在
    }
    return 0;
}

__global__ void beam_search_kernel(const int* __restrict__ start_node_ids,
                                   const float* __restrict__ queries,
                                   const Slot* __restrict__ slots,
                                   const float* __restrict__ vec_slots, BCD* __restrict__ beams,
                                   int* __restrict__ hash_visited, int* __restrict__ ids_res,
                                   float* __restrict__ dists_res, int* __restrict__ hops,
                                   int* __restrict__ unprocessed_qid, int k, int Q, int M, BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x, wpb = tpb / warpSize;
    const int lane = tid % warpSize;  // warp 内线程号
    const int warp = tid / warpSize;  // block 内 warp 号
    const float farthest = farthest_dist_d(bp), closest = closest_dist_d(bp);
    const int end_step = 1;

    extern __shared__ float sh_query[];                          // dim
    float* sh_dists = sh_query + bp.dim;                         // tpb
    int* sh_filter_idxs = (int*)(sh_dists + tpb);                // tpb
    int* sh_idxs = sh_filter_idxs + tpb;                         // tpb
    BCD* sh_beams_a = (BCD*)(sh_idxs + tpb);                     // beam_capacity
    BCD* sh_beams_b = sh_beams_a + bp.beam_capacity;             // beam_capacity
    bool* sh_expanded = (bool*)(sh_beams_b + bp.beam_capacity);  // beam_capacity

    __shared__ int s_qid, s_new_cand_idx, s_beam_num, s_hop;
    __shared__ float s_farthest_dist;  // 当前 beam 中最远距离
    __shared__ int s_flag, s_step, s_beam_select;

    if (tid == 0) {
        s_qid = -1;
        s_step = 0;
    }

    int* beam_visited = hash_visited + bid * bp.hash_n;
    BCD* beam = beams + bid * bp.beam_size;
    while (true) {
        __syncthreads();
        if (s_qid < 0) {
            if (tid == 0) {
                s_qid = atomicAdd(unprocessed_qid, 1);  // 获取新的 qid
                if (s_qid >= Q)                         // 所有 qid 都处理完了
                    s_qid = -1;
                else
                    s_hop = 0, s_beam_num = M, s_farthest_dist = farthest, s_beam_select = 0;
            }
            __syncthreads();
            if (s_qid < 0) break;
            // 初始化：
            for (int i = tid; i < bp.hash_n; i += tpb) beam_visited[i] = -1;
            for (int d = tid; d < bp.dim; d += tpb) sh_query[d] = queries[s_qid * bp.dim + d];
            for (int i = tid; i < bp.beam_capacity; i += tpb) sh_expanded[i] = false;
            __syncthreads();
            for (int start_i = warp; start_i < M; start_i += wpb) {
                int start_id = -1;
                if (lane == 0) {
                    start_id = start_node_ids[start_i];
                    assert(start_id >= 0);
                    const int success = beam_insert(beam_visited, start_id, bp.hash_n);
                    // if (!success) printf("ERROR: %d is visited", start_id);
                }
                start_id = __shfl_sync(0xffffffff, start_id, 0);
                float dist =
                    warp_calc_dist(vec_slots + (size_t)start_id * bp.dim, sh_query, lane, bp);
                if (lane == 0) {
                    sh_beams_a[start_i] = {start_id, dist};
                    // if (slots[start_id].deg <= 0)
                    //     printf("ERROR: slots[%d].deg = %d", start_id, slots[start_id].deg);
                }
            }
            for (int i = M + tid; i < bp.beam_capacity; i += tpb) sh_beams_a[i] = {-1, farthest};
            for (int i = tid; i < bp.beam_capacity; i += tpb) sh_beams_b[i] = {-1, farthest};
        }

        if (!tid) {
            s_new_cand_idx = 0, s_flag = 1;
            s_step++;
        }
        __syncthreads();

        BCD* sh_beams = !s_beam_select ? sh_beams_a : sh_beams_b;
        for (int beam_i = warp; beam_i < s_beam_num; beam_i += wpb) {
            int id = sh_beams[beam_i].id, deg = -1;
            if (lane == 0 && !sh_expanded[beam_i]) {
                assert(id >= 0);
                deg = slots[id].deg;
                sh_expanded[beam_i] = true;
            }
            deg = __shfl_sync(0xffffffff, deg, 0);
            if (deg <= 0) continue;

            const int* nbrs = slots[id].nbrs;

            for (int nbr_i = lane; nbr_i < deg; nbr_i += warpSize) {  // warp 内遍历邻居
                // bool valid = false;
                const int nbrid = nbrs[nbr_i];
                // float dist;
                if (beam_insert(beam_visited, nbrid, bp.hash_n) == 1) {
                    const float dist =
                        calc_distance(vec_slots + (size_t)nbrid * bp.dim, sh_query, bp);
                    // valid = compare_dist(s_farthest_dist, dist, bp);
                    if (compare_dist(s_farthest_dist, dist, bp))
                        beam[atomicAdd(&s_new_cand_idx, 1)] = {nbrid, dist};
                }

                // warp 压缩写入 beam
                // const unsigned active = __activemask(), mask = __ballot_sync(active, valid);
                // const int cnt = __popc(mask);
                // if (cnt > 0) {
                //     int base_idx = -1;
                //     if (lane == 0) base_idx = atomicAdd(&s_new_cand_idx, cnt);
                //     base_idx = __shfl_sync(active, base_idx, 0);
                //     if (valid) beam[base_idx + __popc(mask & ((1u << lane) - 1))] = {nbrid,
                //     dist};
                // }
            }
        }
        __syncthreads();
        // if (s_step == end_step) return;

        const int total = s_beam_num + s_new_cand_idx;
        if (s_new_cand_idx == 0) {  // 收敛, 筛选 top-k 的节点
            // if (s_beam_num < bp.beam_capacity) printf("ERROR: s_beam_num < bp.beam_capacity");
            assert(s_beam_num == bp.beam_capacity);
            int* ids_out = ids_res + s_qid * k;
            float* dists_out = dists_res ? dists_res + s_qid * k : nullptr;
            for (int i = 0; i < k; i++) {
                int local_idx = -1;
                float local_best = farthest;
                for (int j = tid; j < bp.beam_capacity; j += tpb) {
                    if (sh_beams[j].id < 0) continue;
                    const float dist = sh_beams[j].dist;
                    if (compare_dist(local_best, dist, bp)) local_best = dist, local_idx = j;
                }
                sh_dists[tid] = local_best, sh_idxs[tid] = local_idx;
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
                    ids_out[i] = sh_beams[ith_idx].id;
                    if (dists_out) dists_out[i] = sh_dists[0];
                    sh_beams[ith_idx].id = -1;
                }
                __syncthreads();
            }
            if (tid == 0) {
                hops[s_qid] = s_hop;
                s_qid = -1;
            }
        } else if (total < bp.beam_capacity) {
            for (int i = tid; i < s_new_cand_idx; i += tpb) sh_beams[s_beam_num + i] = beam[i];
            __syncthreads();
            if (tid == 0) {
                atomicAdd(&s_hop, s_new_cand_idx);
                s_beam_num = total;
            }
        } else {  // 筛选 top-beam_capacity 的节点
            float farthest_d;
            if (total - bp.beam_capacity > bp.beam_capacity) {
                BCD* sh_new_beams = s_beam_select ? sh_beams_a : sh_beams_b;
                for (int i = 0; i < bp.beam_capacity; i++) {
                    int local_idx = -1;
                    float local_best = farthest;
                    for (int j = tid; j < total; j += tpb) {
                        const float dist =
                            j < s_beam_num ? sh_beams[j].dist : beam[j - s_beam_num].dist;
                        if (compare_dist(local_best, dist, bp)) local_best = dist, local_idx = j;
                    }
                    sh_dists[tid] = local_best, sh_idxs[tid] = local_idx;
                    __syncthreads();

                    for (int offset = tpb >> 1; offset > 0; offset >>= 1) {
                        if (tid < offset &&
                            compare_dist(sh_dists[tid], sh_dists[tid + offset], bp)) {
                            sh_dists[tid] = sh_dists[tid + offset];
                            sh_idxs[tid] = sh_idxs[tid + offset];
                        }
                        __syncthreads();
                    }
                    if (tid == 0) {
                        const int ith_idx = sh_idxs[0];
                        if (ith_idx < s_beam_num) {
                            sh_new_beams[i] = sh_beams[ith_idx];
                            sh_beams[ith_idx] = {-1, farthest};
                            sh_expanded[i] = sh_expanded[ith_idx];
                        } else {
                            sh_new_beams[i] = beam[ith_idx - s_beam_num];
                            beam[ith_idx - s_beam_num] = {-1, farthest};
                            sh_expanded[i] = false;
                        }
                    }
                    __syncthreads();
                }
                if (!tid) {
                    farthest_d = sh_new_beams[bp.beam_capacity - 1].dist;
                    s_beam_select = 1 - s_beam_select;
                }
            } else {
                const int sh_free_num = bp.beam_capacity - s_beam_num;
                for (int i = tid; i < sh_free_num; i += tpb) sh_beams[s_beam_num + i] = beam[i];
                sh_filter_idxs[tid] = sh_free_num + tid;
                __syncthreads();
                while (s_flag) {
                    if (tid == 0) s_flag = 0;
                    __syncthreads();
                    int local_idx = -1;
                    float local_worst = closest;
                    for (int j = tid; j < bp.beam_capacity; j += tpb) {
                        const float dist = sh_beams[j].dist;
                        if (compare_dist(dist, local_worst, bp)) local_worst = dist, local_idx = j;
                    }
                    sh_dists[tid] = local_worst;
                    sh_idxs[tid] = local_idx;
                    __syncthreads();
                    for (int offset = tpb >> 1; offset > 0; offset >>= 1) {
                        if (tid < offset &&
                            compare_dist(sh_dists[tid + offset], sh_dists[tid], bp)) {
                            sh_dists[tid] = sh_dists[tid + offset];
                            sh_idxs[tid] = sh_idxs[tid + offset];
                        }
                        __syncthreads();
                    }

                    farthest_d = sh_dists[0];
                    int i = sh_filter_idxs[tid];
                    while (!s_flag && i < s_new_cand_idx) {
                        const BCD& cand = beam[i];
                        if (compare_dist(cand.dist, farthest_d, bp))
                            i += tpb;
                        else {
                            if (!atomicCAS(&s_flag, 0, 1)) {
                                const int worst_idx = sh_idxs[0];
                                assert(cand.id >= 0);
                                sh_beams[worst_idx] = cand;
                                sh_expanded[worst_idx] = false;
                                i += tpb;
                            }
                            break;
                        }
                    }
                    sh_filter_idxs[tid] = i;
                    __syncthreads();
                }
            }
            if (tid == 0) {
                s_farthest_dist = farthest_d;
                atomicAdd(&s_hop, s_new_cand_idx);
                s_beam_num = bp.beam_capacity;
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__ int atomicTestAndSetBit(uint32_t* bitset, int bit_index) {
    assert(bit_index >= 0);
    const uint32_t mask = 1u << (bit_index & 31);
    uint32_t* addr = bitset + (bit_index >> 5);
    const uint32_t old = atomicOr(addr, mask);
    return (old & mask) != 0;
}

__device__ __forceinline__ int bitTest(uint32_t* bitset, int bit_index) {
    return (bitset[bit_index >> 5] & (1u << (bit_index & 31))) != 0;
}

__global__ void beam_search_kernel(const int* __restrict__ start_node_ids,
                                   const float* __restrict__ queries,
                                   const Slot* __restrict__ slots,
                                   const float* __restrict__ vec_slots, BCD* __restrict__ beams,
                                   uint32_t* __restrict__ visited, int* __restrict__ ids_res,
                                   float* __restrict__ dists_res, int* __restrict__ hops,
                                   int* __restrict__ unprocessed_qid, int k, int Q, int M, BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x, wpb = tpb / warpSize;
    const int lane = tid % warpSize;  // warp 内线程号
    const int warp = tid / warpSize;  // block 内 warp 号
    const float farthest = farthest_dist_d(bp), closest = closest_dist_d(bp);

    extern __shared__ float sh_query[];                          // dim
    float* sh_dists = sh_query + bp.dim;                         // tpb
    int* sh_filter_idxs = (int*)(sh_dists + tpb);                // tpb
    int* sh_idxs = sh_filter_idxs + tpb;                         // tpb
    BCD* sh_beams_a = (BCD*)(sh_idxs + tpb);                     // beam_capacity
    BCD* sh_beams_b = sh_beams_a + bp.beam_capacity;             // beam_capacity
    bool* sh_expanded = (bool*)(sh_beams_b + bp.beam_capacity);  // beam_capacity

    __shared__ int s_qid, s_new_cand_idx, s_beam_num, s_hop;
    __shared__ int s_flag, s_beam_select;
    __shared__ float s_farthest_dist;  // 当前 beam 中最远距离

    if (tid == 0) s_qid = -1;

    uint32_t* beam_visited = visited + bid * bp.visit_n;
    BCD* beam = beams + bid * bp.beam_size;
    while (true) {
        __syncthreads();
        if (s_qid < 0) {
            if (tid == 0) {
                s_qid = atomicAdd(unprocessed_qid, 1);  // 获取新的 qid
                if (s_qid >= Q)                         // 所有 qid 都处理完了
                    s_qid = -1;
                else
                    s_hop = 0, s_beam_num = M, s_beam_select = 0, s_farthest_dist = farthest;
            }
            __syncthreads();
            if (s_qid < 0) break;
            // 初始化：
            for (int i = tid; i < bp.visit_n; i += tpb) beam_visited[i] = 0;
            for (int d = tid; d < bp.dim; d += tpb) sh_query[d] = queries[s_qid * bp.dim + d];
            for (int i = tid; i < bp.beam_capacity; i += tpb) sh_expanded[i] = false;
            __syncthreads();
            for (int start_i = warp; start_i < M; start_i += wpb) {
                int start_id = -1;
                if (lane == 0) {
                    start_id = start_node_ids[start_i];
                    const int vis = atomicTestAndSetBit(beam_visited, start_id);
                    // if (vis) printf("ERROR: %d is visited", start_id);
                }
                start_id = __shfl_sync(0xffffffff, start_id, 0);
                float dist =
                    warp_calc_dist(vec_slots + (size_t)start_id * bp.dim, sh_query, lane, bp);
                if (lane == 0) {
                    sh_beams_a[start_i] = {start_id, dist};
                    // if (slots[start_id].deg <= 0)
                    //     printf("ERROR: slots[%d].deg = %d", start_id, slots[start_id].deg);
                }
            }
            for (int i = M + tid; i < bp.beam_capacity; i += tpb) sh_beams_a[i] = {-1, farthest};
            for (int i = tid; i < bp.beam_capacity; i += tpb) sh_beams_b[i] = {-1, farthest};
        }

        if (!tid) s_new_cand_idx = 0, s_flag = 1;
        __syncthreads();

        BCD* sh_beams = !s_beam_select ? sh_beams_a : sh_beams_b;
        for (int beam_i = warp; beam_i < s_beam_num; beam_i += wpb) {
            int id = sh_beams[beam_i].id, deg = -1;
            if (lane == 0 && !sh_expanded[beam_i]) {
                assert(id >= 0);
                deg = slots[id].deg;
                sh_expanded[beam_i] = true;
            }
            deg = __shfl_sync(0xffffffff, deg, 0);
            if (deg <= 0) continue;

            const int* nbrs = slots[id].nbrs;
            for (int nbr_i = lane; nbr_i < deg; nbr_i += warpSize) {  // warp 内遍历邻居
                const int nbrid = nbrs[nbr_i];
                float dist;
                if (atomicTestAndSetBit(beam_visited, nbrid) == 0) {
                    dist = calc_distance(vec_slots + (size_t)nbrid * bp.dim, sh_query, bp);
                    if (compare_dist(s_farthest_dist, dist, bp)) {
                        beam[atomicAdd(&s_new_cand_idx, 1)] = {nbrid, dist};
                    }
                    // valid = compare_dist(s_farthest_dist, dist, bp);
                }
            }
        }
        __syncthreads();

        const int total = s_beam_num + min(s_new_cand_idx, bp.beam_capacity);
        if (s_new_cand_idx == 0) {  // 收敛, 筛选 top-k 的节点
            // if (s_beam_num < bp.beam_capacity) printf("ERROR: s_beam_num < bp.beam_capacity");
            assert(s_beam_num == bp.beam_capacity);
            int* ids_out = ids_res + s_qid * k;
            float* dists_out = dists_res ? dists_res + s_qid * k : nullptr;
            for (int i = 0; i < k; i++) {
                int local_idx = -1;
                float local_best = farthest;
                for (int j = tid; j < bp.beam_capacity; j += tpb) {
                    if (sh_beams[j].id < 0) continue;
                    const float dist = sh_beams[j].dist;
                    if (compare_dist(local_best, dist, bp)) local_best = dist, local_idx = j;
                }
                sh_dists[tid] = local_best, sh_idxs[tid] = local_idx;
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
                    ids_out[i] = sh_beams[ith_idx].id;
                    if (dists_out) dists_out[i] = sh_dists[0];
                    sh_beams[ith_idx].id = -1;
                }
                __syncthreads();
            }
            if (tid == 0) {
                hops[s_qid] = s_hop;
                s_qid = -1;
            }
        } else if (total <= bp.beam_capacity) {
            for (int i = tid; i < s_new_cand_idx; i += tpb) sh_beams[s_beam_num + i] = beam[i];
            __syncthreads();
            if (tid == 0) {
                atomicAdd(&s_hop, s_new_cand_idx);
                s_beam_num = total;
            }
        } else {  // 筛选 top-beam_capacity 的节点
            float farthest_d;
            if (total > 2 * bp.beam_capacity) {
                BCD* sh_new_beams = s_beam_select ? sh_beams_a : sh_beams_b;
                for (int i = 0; i < bp.beam_capacity; i++) {
                    int local_idx = -1;
                    float local_best = farthest;
                    for (int j = tid; j < total; j += tpb) {
                        const float dist =
                            j < s_beam_num ? sh_beams[j].dist : beam[j - s_beam_num].dist;
                        if (compare_dist(local_best, dist, bp)) local_best = dist, local_idx = j;
                    }
                    sh_dists[tid] = local_best, sh_idxs[tid] = local_idx;
                    __syncthreads();

                    for (int offset = tpb >> 1; offset > 0; offset >>= 1) {
                        if (tid < offset &&
                            compare_dist(sh_dists[tid], sh_dists[tid + offset], bp)) {
                            sh_dists[tid] = sh_dists[tid + offset];
                            sh_idxs[tid] = sh_idxs[tid + offset];
                        }
                        __syncthreads();
                    }
                    if (tid == 0) {
                        const int ith_idx = sh_idxs[0];
                        if (ith_idx < s_beam_num) {
                            sh_new_beams[i] = sh_beams[ith_idx];
                            sh_beams[ith_idx] = {-1, farthest};
                            sh_expanded[i] = sh_expanded[ith_idx];
                        } else {
                            sh_new_beams[i] = beam[ith_idx - s_beam_num];
                            beam[ith_idx - s_beam_num] = {-1, farthest};
                            sh_expanded[i] = false;
                        }
                    }
                    __syncthreads();
                }
                if (!tid) {
                    farthest_d = sh_new_beams[bp.beam_capacity - 1].dist;
                    s_beam_select = 1 - s_beam_select;
                }
            } else {
                const int sh_free_num = bp.beam_capacity - s_beam_num;
                for (int i = tid; i < sh_free_num; i += tpb) sh_beams[s_beam_num + i] = beam[i];
                sh_filter_idxs[tid] = sh_free_num + tid;
                __syncthreads();
                while (s_flag) {
                    if (tid == 0) s_flag = 0;
                    __syncthreads();
                    int local_idx = -1;
                    float local_worst = closest;
                    for (int j = tid; j < bp.beam_capacity; j += tpb) {
                        const float dist = sh_beams[j].dist;
                        if (compare_dist(dist, local_worst, bp)) local_worst = dist, local_idx = j;
                    }
                    sh_dists[tid] = local_worst;
                    sh_idxs[tid] = local_idx;
                    __syncthreads();
                    for (int offset = tpb >> 1; offset > 0; offset >>= 1) {
                        if (tid < offset &&
                            compare_dist(sh_dists[tid + offset], sh_dists[tid], bp)) {
                            sh_dists[tid] = sh_dists[tid + offset];
                            sh_idxs[tid] = sh_idxs[tid + offset];
                        }
                        __syncthreads();
                    }

                    farthest_d = sh_dists[0];
                    int i = sh_filter_idxs[tid];
                    while (!s_flag && i < s_new_cand_idx) {
                        const BCD& cand = beam[i];
                        if (compare_dist(cand.dist, farthest_d, bp))
                            i += tpb;
                        else {
                            if (!atomicCAS(&s_flag, 0, 1)) {
                                const int worst_idx = sh_idxs[0];
                                assert(cand.id >= 0);
                                sh_beams[worst_idx] = cand;
                                sh_expanded[worst_idx] = false;
                                i += tpb;
                            }
                            break;
                        }
                    }
                    sh_filter_idxs[tid] = i;
                    __syncthreads();
                }
            }
            if (tid == 0) {
                s_farthest_dist = farthest_d;
                atomicAdd(&s_hop, s_new_cand_idx);
                s_beam_num = bp.beam_capacity;
            }
        }
        __syncthreads();
    }
}

__global__ void beam_search_kernel(const int* __restrict__ start_node_ids,
                                   const float* __restrict__ queries,
                                   const Slot* __restrict__ slots,
                                   const float* __restrict__ vec_slots, BCD* __restrict__ beams,
                                   uint32_t* __restrict__ visited, uint32_t* __restrict__ deleted,
                                   int* __restrict__ ids_res, float* __restrict__ dists_res,
                                   int* __restrict__ hops, int* __restrict__ unprocessed_qid,
                                   int k, int Q, int M, BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x, wpb = tpb / warpSize;
    const int lane = tid % warpSize;  // warp 内线程号
    const int warp = tid / warpSize;  // block 内 warp 号
    const float farthest = farthest_dist_d(bp), closest = closest_dist_d(bp);

    extern __shared__ float sh_query[];                          // dim
    float* sh_dists = sh_query + bp.dim;                         // tpb
    int* sh_filter_idxs = (int*)(sh_dists + tpb);                // tpb
    int* sh_idxs = sh_filter_idxs + tpb;                         // tpb
    BCD* sh_beams_a = (BCD*)(sh_idxs + tpb);                     // beam_capacity
    BCD* sh_beams_b = sh_beams_a + bp.beam_capacity;             // beam_capacity
    bool* sh_expanded = (bool*)(sh_beams_b + bp.beam_capacity);  // beam_capacity

    __shared__ int s_qid, s_new_cand_idx, s_beam_num, s_hop, s_i;
    __shared__ int s_flag, s_beam_select;
    __shared__ float s_farthest_dist;  // 当前 beam 中最远距离

    if (tid == 0) s_qid = -1;

    uint32_t* beam_visited = visited + bid * bp.visit_n;
    BCD* beam = beams + bid * bp.beam_size;
    while (true) {
        __syncthreads();
        if (s_qid < 0) {
            if (tid == 0) {
                s_qid = atomicAdd(unprocessed_qid, 1);  // 获取新的 qid
                if (s_qid >= Q)                         // 所有 qid 都处理完了
                    s_qid = -1;
                else
                    s_hop = 0, s_beam_num = M, s_beam_select = 0, s_farthest_dist = farthest,
                    s_i = 0;
            }
            __syncthreads();
            if (s_qid < 0) break;
            // 初始化：
            for (int i = tid; i < bp.visit_n; i += tpb) beam_visited[i] = 0;
            for (int d = tid; d < bp.dim; d += tpb) sh_query[d] = queries[s_qid * bp.dim + d];
            for (int i = tid; i < bp.beam_capacity; i += tpb) sh_expanded[i] = false;
            __syncthreads();
            for (int start_i = warp; start_i < M; start_i += wpb) {
                int start_id = -1;
                if (lane == 0) {
                    start_id = start_node_ids[start_i];
                    const int vis = atomicTestAndSetBit(beam_visited, start_id);
                    // if (vis) printf("ERROR: %d is visited", start_id);
                }
                start_id = __shfl_sync(0xffffffff, start_id, 0);
                float dist =
                    warp_calc_dist(vec_slots + (size_t)start_id * bp.dim, sh_query, lane, bp);
                if (lane == 0) {
                    sh_beams_a[start_i] = {start_id, dist};
                    // if (slots[start_id].deg <= 0)
                    //     printf("ERROR: slots[%d].deg = %d", start_id, slots[start_id].deg);
                }
            }
            for (int i = M + tid; i < bp.beam_capacity; i += tpb) sh_beams_a[i] = {-1, farthest};
            for (int i = tid; i < bp.beam_capacity; i += tpb) sh_beams_b[i] = {-1, farthest};
        }

        if (!tid) s_new_cand_idx = 0, s_flag = 1;
        __syncthreads();

        BCD* sh_beams = !s_beam_select ? sh_beams_a : sh_beams_b;
        for (int beam_i = warp; beam_i < s_beam_num; beam_i += wpb) {
            int id = sh_beams[beam_i].id, deg = -1;
            if (lane == 0 && !sh_expanded[beam_i]) {
                assert(id >= 0);
                deg = slots[id].deg;
                sh_expanded[beam_i] = true;
            }
            deg = __shfl_sync(0xffffffff, deg, 0);
            if (deg <= 0) continue;

            const int* nbrs = slots[id].nbrs;
            for (int nbr_i = lane; nbr_i < deg; nbr_i += warpSize) {  // warp 内遍历邻居
                bool valid = false;
                const int nbrid = nbrs[nbr_i];
                float dist;
                if (atomicTestAndSetBit(beam_visited, nbrid) == 0) {
                    dist = calc_distance(vec_slots + (size_t)nbrid * bp.dim, sh_query, bp);
                    if (compare_dist(s_farthest_dist, dist, bp)) {
                        beam[atomicAdd(&s_new_cand_idx, 1)] = {nbrid, dist};
                    }
                    // valid = compare_dist(s_farthest_dist, dist, bp);
                }
                // warp 压缩写入 beam
                // const unsigned active = __activemask(), mask = __ballot_sync(active, valid);
                // const int cnt = __popc(mask);
                // if (cnt > 0) {
                //     int base_idx = -1;
                //     if (lane == 0) base_idx = atomicAdd(&s_new_cand_idx, cnt);
                //     base_idx = __shfl_sync(active, base_idx, 0);
                //     if (valid) beam[base_idx + __popc(mask & ((1u << lane) - 1))] = {nbrid,
                //     dist};
                // }
            }
        }
        __syncthreads();

        const int total = s_beam_num + s_new_cand_idx;
        if (s_new_cand_idx == 0) {  // 收敛, 筛选 top-k 的节点
            // if (s_beam_num < bp.beam_capacity) printf("ERROR: s_beam_num < bp.beam_capacity");
            assert(s_beam_num == bp.beam_capacity);
            int* ids_out = ids_res + s_qid * k;
            float* dists_out = dists_res ? dists_res + s_qid * k : nullptr;
            while (s_i < k) {
                int local_idx = -1;
                float local_best = farthest;
                for (int j = tid; j < bp.beam_capacity; j += tpb) {
                    if (sh_beams[j].id < 0) continue;
                    const float dist = sh_beams[j].dist;
                    if (compare_dist(local_best, dist, bp)) local_best = dist, local_idx = j;
                }
                sh_dists[tid] = local_best, sh_idxs[tid] = local_idx;
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
                    const int id = sh_beams[ith_idx].id;
                    if (id < 0 || bitTest(deleted, id) == 0) {
                        if (id < 0) printf("search id < 0!");
                        ids_out[s_i] = id;
                        if (dists_out) dists_out[s_i] = sh_dists[0];
                        s_i++;
                    }
                    sh_beams[ith_idx].id = -1;
                }
                __syncthreads();
            }
            if (tid == 0) {
                hops[s_qid] = s_hop;
                s_qid = -1;
            }
        } else if (total <= bp.beam_capacity) {
            for (int i = tid; i < s_new_cand_idx; i += tpb) sh_beams[s_beam_num + i] = beam[i];
            __syncthreads();
            if (tid == 0) {
                atomicAdd(&s_hop, s_new_cand_idx);
                s_beam_num = total;
            }
        } else {  // 筛选 top-beam_capacity 的节点
            float farthest_d;
            if (total > 2 * bp.beam_capacity) {
                BCD* sh_new_beams = s_beam_select ? sh_beams_a : sh_beams_b;
                for (int i = 0; i < bp.beam_capacity; i++) {
                    int local_idx = -1;
                    float local_best = farthest;
                    for (int j = tid; j < total; j += tpb) {
                        const float dist =
                            j < s_beam_num ? sh_beams[j].dist : beam[j - s_beam_num].dist;
                        if (compare_dist(local_best, dist, bp)) local_best = dist, local_idx = j;
                    }
                    sh_dists[tid] = local_best, sh_idxs[tid] = local_idx;
                    __syncthreads();

                    for (int offset = tpb >> 1; offset > 0; offset >>= 1) {
                        if (tid < offset &&
                            compare_dist(sh_dists[tid], sh_dists[tid + offset], bp)) {
                            sh_dists[tid] = sh_dists[tid + offset];
                            sh_idxs[tid] = sh_idxs[tid + offset];
                        }
                        __syncthreads();
                    }
                    if (tid == 0) {
                        const int ith_idx = sh_idxs[0];
                        if (ith_idx < s_beam_num) {
                            sh_new_beams[i] = sh_beams[ith_idx];
                            sh_beams[ith_idx] = {-1, farthest};
                            sh_expanded[i] = sh_expanded[ith_idx];
                        } else {
                            sh_new_beams[i] = beam[ith_idx - s_beam_num];
                            beam[ith_idx - s_beam_num] = {-1, farthest};
                            sh_expanded[i] = false;
                        }
                    }
                    __syncthreads();
                }
                if (!tid) {
                    farthest_d = sh_new_beams[bp.beam_capacity - 1].dist;
                    s_beam_select = 1 - s_beam_select;
                }
            } else {
                const int sh_free_num = bp.beam_capacity - s_beam_num;
                for (int i = tid; i < sh_free_num; i += tpb) sh_beams[s_beam_num + i] = beam[i];
                sh_filter_idxs[tid] = sh_free_num + tid;
                __syncthreads();
                while (s_flag) {
                    if (tid == 0) s_flag = 0;
                    __syncthreads();
                    int local_idx = -1;
                    float local_worst = closest;
                    for (int j = tid; j < bp.beam_capacity; j += tpb) {
                        const float dist = sh_beams[j].dist;
                        if (compare_dist(dist, local_worst, bp)) local_worst = dist, local_idx = j;
                    }
                    sh_dists[tid] = local_worst;
                    sh_idxs[tid] = local_idx;
                    __syncthreads();
                    for (int offset = tpb >> 1; offset > 0; offset >>= 1) {
                        if (tid < offset &&
                            compare_dist(sh_dists[tid + offset], sh_dists[tid], bp)) {
                            sh_dists[tid] = sh_dists[tid + offset];
                            sh_idxs[tid] = sh_idxs[tid + offset];
                        }
                        __syncthreads();
                    }

                    farthest_d = sh_dists[0];
                    int i = sh_filter_idxs[tid];
                    while (!s_flag && i < s_new_cand_idx) {
                        const BCD& cand = beam[i];
                        if (compare_dist(cand.dist, farthest_d, bp))
                            i += tpb;
                        else {
                            if (!atomicCAS(&s_flag, 0, 1)) {
                                const int worst_idx = sh_idxs[0];
                                assert(cand.id >= 0);
                                sh_beams[worst_idx] = cand;
                                sh_expanded[worst_idx] = false;
                                i += tpb;
                            }
                            break;
                        }
                    }
                    sh_filter_idxs[tid] = i;
                    __syncthreads();
                }
            }
            if (tid == 0) {
                s_farthest_dist = farthest_d;
                atomicAdd(&s_hop, s_new_cand_idx);
                s_beam_num = bp.beam_capacity;
            }
        }
        __syncthreads();
    }
}

Test_Result GPUFuncs::search_gpu(const float* queries, int* ids_res, float* dists_res, int Q,
                                 int k, std::vector<int>& start_node_ids, bool have_delete,
                                 cudaStream_t stream) {
    if (k <= 0) k = bp.k;
    if (stream == nullptr) stream = streams["search"].stream;

    // auto& start_node_ids = graph->get_search_start_nodes_fps(M, reset);
    // assert(start_node_ids.size() >= M);
    const int M = start_node_ids.size();
    // for (auto id : start_node_ids) assert(id >= 0 && id < bp.base_n);
    // std::string str;
    // for (int i = 0; i < M; i++) {
    //     const int id = start_node_ids[i];
    //     str += TOS(id) + " " + TOS(graph->get_cpu_slots()[id].deg) + "; ";
    // }
    // fo.print("start nodes: " + str);
    int *hops, *d_unprocessed_qid, *d_start_node_ids, *d_hops;
    CUDA_CHECK(cudaMallocHost(&hops, Q * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_unprocessed_qid, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_start_node_ids, M * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_hops, Q * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_unprocessed_qid, 0, sizeof(int)));
    CUDA_CHECK(cudaMemcpyAsync(d_start_node_ids, start_node_ids.data(), M * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    int tpb = 512, shared_size = (bp.dim + tpb) * sizeof(float) + 2 * tpb * sizeof(int) +
                                 bp.beam_capacity * (sizeof(BCD) * 2 + sizeof(bool));
    auto e = gpu_record_time_start(stream);

    auto data = search_data[0];
    if (!have_delete)
        beam_search_kernel<<<data->batch_, tpb, shared_size, stream>>>(
            d_start_node_ids, queries, graph->get_gpu_slots(), graph->get_gpu_vec_slots(),
            data->d_beams_, data->d_visited_, ids_res, dists_res, d_hops, d_unprocessed_qid, k, Q,
            M, bp);
    else
        beam_search_kernel<<<data->batch_, tpb, shared_size, stream>>>(
            d_start_node_ids, queries, graph->get_gpu_slots(), graph->get_gpu_vec_slots(),
            data->d_beams_, data->d_visited_, d_deleted.data().get(), ids_res, dists_res, d_hops,
            d_unprocessed_qid, k, Q, M, bp);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(hops, d_hops, Q * sizeof(int), cudaMemcpyDeviceToHost, stream));
    const float time = gpu_record_time_stop(e, stream, "beam search");
    size_t total_hop = 0;
    for (int i = 0; i < Q; ++i) total_hop += hops[i];
    cudaFreeHost(hops);
    cudaFree(d_unprocessed_qid);
    cudaFree(d_start_node_ids);
    cudaFree(d_hops);

    return Test_Result(-1, total_hop, time);
}

__global__ void ids_verify_kernel(const int* __restrict__ ids_res, const int* __restrict__ ids_gt,
                                  int* hit, int Q, int k) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    __shared__ int s_hit;
    for (int qi = bid; qi < Q; qi += gridDim.x) {
        if (tid == 0) s_hit = 0;
        __syncthreads();

        for (int i = tid; i < k; i += tpb) {
            const int id = ids_res[qi * k + i], *gts = ids_gt + qi * k;
            for (int j = 0; j < k; j++)
                if (id == gts[j]) {
                    atomicAdd(&s_hit, 1);
                    break;
                }
        }
        __syncthreads();
        if (tid == 0) atomicAdd(hit, s_hit);
    }
}

void GPUFuncs::search_verify(const int* ids_res, const int* ids_gt, Test_Result& tr, int Q, int k,
                             cudaStream_t stream) {
    if (stream == nullptr) stream = streams["search"].stream;
    thrust::device_vector<int> hit(1, 0);
    ids_verify_kernel<<<1024, 64, 0, stream>>>(ids_res, ids_gt, hit.data().get(), Q, k);
    CUDA_CHECK(cudaMemcpyAsync(&(tr.hits), hit.data().get(), sizeof(int), cudaMemcpyDeviceToHost,
                               stream));
    // std::vector<int> res(bp.k);
    // cudaMemcpyAsync(res.data(), ids_res, bp.k * sizeof(int), cudaMemcpyDeviceToHost, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // std::string str;
    // for (int id : res) str += std::to_string(id) + " ";
    // fo.print(str);
}

}  // namespace efanna2e

// __device__ __forceinline__ void blockTopKInsert(BCD* heap, int id, float dist, BP bp, int K,
//                                                 float* s_worst, int* s_lock, int* s_any_insert)
//                                                 {
//     if (compare_dist(dist, *s_worst, bp)) return;  // quick reject

//     while (atomicCAS(s_lock, 0, 1) != 0);  // lock

//     if (compare_dist(heap[0].dist, dist, bp)) {
//         heap[0] = {id, dist, false};
//         heap_sift_down(heap, 0, K, bp);
//         *s_worst = heap[0].dist;
//         if (*s_any_insert == 0) atomicExch(s_any_insert, 1);
//     }

//     __threadfence_block();
//     atomicExch(s_lock, 0);
// }

// __device__ __forceinline__ void blockSelectTopK(BCD* cand, int total, BCD* out, int K, BP bp) {
//     extern __shared__ float sh_d[];
//     int* sh_i = (int*)(sh_d + blockDim.x);

//     const float worst = farthest_dist_d(bp);
//     for (int k = 0; k < K; ++k) {
//         float best = worst;
//         int best_i = -1;
//         for (int i = threadIdx.x; i < total; i += blockDim.x)
//             if (cand[i].id >= 0 && compare_dist(best, cand[i].dist, bp)) {
//                 best = cand[i].dist;
//                 best_i = i;
//             }
//         sh_d[threadIdx.x] = best;
//         sh_i[threadIdx.x] = best_i;
//         __syncthreads();

//         for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
//             if (threadIdx.x < s && compare_dist(sh_d[threadIdx.x], sh_d[threadIdx.x + s], bp)) {
//                 sh_d[threadIdx.x] = sh_d[threadIdx.x + s];
//                 sh_i[threadIdx.x] = sh_i[threadIdx.x + s];
//             }
//             __syncthreads();
//         }
//         if (threadIdx.x == 0) {
//             const int kth_i = sh_i[0];
//             out[k] = cand[kth_i];
//             cand[kth_i].id = -1;
//         }
//         __syncthreads();
//     }
// }

// void id_res_extract_from_BCD(const BCD* beam, int* ids_out, cudaStream_t stream, int k) {
//     auto policy = thrust::cuda::par.on(stream);
//     thrust::device_ptr<const BCD> beam_ptr(beam);
//     thrust::device_ptr<int> ids_ptr(ids_out);

//     thrust::transform(policy, beam_ptr, beam_ptr + k, ids_ptr,
//                       [] __host__ __device__(const BCD& bc) { return bc.id; });
// }

// constexpr int WARP_L = 32;    // warp local buffer
// constexpr int BLOCK_L = 256;  // block merge buffer

// __global__ void cagra_beam_search_nolock_kernel(
//     const int* __restrict__ start_ids, const float* __restrict__ queries,
//     const Slot* __restrict__ slots, const float* __restrict__ vecs, BCD* __restrict__ beams,
//     uint32_t* __restrict__ visited, int* __restrict__ out_ids, float* __restrict__ out_dists,
//     int* __restrict__ unprocessed_qid, int Q, int k, int M, BP bp) {
//     const int bid = blockIdx.x, tid = threadIdx.x, lane = tid % warpSize, warp = tid / warpSize,
//               tpb = blockDim.x, wpb = tpb / warpSize;

//     extern __shared__ char smem[];
//     float* sh_query = (float*)smem;
//     int* warp_ids = (int*)(sh_query + bp.dim);
//     float* warp_dist = (float*)(warp_ids + wpb * WARP_L);
//     BCD* cand = (BCD*)(warp_dist + wpb * WARP_L);

//     __shared__ int s_qid;
//     if (tid == 0) s_qid = -1;

//     uint32_t* vis = visited + (size_t)bid * bp.visit_n;
//     BCD* beam = beams + (size_t)bid * bp.beam_capacity;
//     const float farthest = farthest_dist_d(bp);

//     while (true) {
//         __syncthreads();
//         if (s_qid < 0) {
//             if (tid == 0) {
//                 const int q = atomicAdd(unprocessed_qid, 1);
//                 s_qid = (q < Q) ? q : -1;
//             }
//             __syncthreads();
//             if (s_qid < 0) return;

//             for (int i = tid; i < bp.visit_n; i += tpb) vis[i] = 0;
//             for (int d = tid; d < bp.dim; d += tpb) sh_query[d] = queries[s_qid * bp.dim + d];
//             __syncthreads();

//             for (int start_i = warp; start_i < bp.beam_capacity; start_i += wpb) {
//                 if (start_i < M) {
//                     int start_id = -1;
//                     if (lane == 0) {
//                         start_id = start_ids[start_i];
//                         if (atomicTestAndSetBit(vis, start_id)) {
//                             start_id = -1;
//                             beam[start_i] = {-1, farthest, true};
//                         }
//                     }
//                     start_id = __shfl_sync(0xffffffff, start_id, 0);
//                     if (start_id < 0) continue;

//                     float dist = 0.0f;
//                     const float* base = vecs + (size_t)start_id * bp.dim;
//                     // lane-per-dim 计算距离
//                     for (int d = lane; d < bp.dim; d += 32) {
//                         const float diff = sh_query[d] - base[d];
//                         dist +=
//                             (bp.metric == DIST_METRIC::L2_) ? diff * diff : sh_query[d] *
//                             base[d];
//                     }
//                     for (int offset = 16; offset > 0; offset >>= 1)  // warp reduce sum
//                         dist += __shfl_down_sync(0xffffffff, dist, offset);
//                     if (lane == 0) beam[start_i] = {start_id, dist, false};
//                 } else if (lane == 0)
//                     beam[start_i] = {-1, farthest, true};
//             }
//         }
//         __syncthreads();

//         int local_id = -1;
//         float local_dist = farthest;
//         for (int bi = warp; bi < bp.beam_capacity; bi += wpb) {
//             if (beam[bi].id < 0 || beam[bi].expanded) continue;
//             if (lane == 0) beam[bi].expanded = true;
//             const int id = beam[bi].id, deg = slots[id].deg, *nbrs = slots[id].nbrs;
//             for (int ni = lane; ni < deg; ni += warpSize) {
//                 const int nid = nbrs[ni];
//                 if (!atomicTestAndSetBit(vis, nid)) {
//                     float d = calc_distance(vecs + (size_t)nid * bp.dim, sh_query, bp);
//                     if (compare_dist(local_dist, d, bp)) {
//                         local_dist = d;
//                         local_id = nid;
//                     }
//                 }
//             }
//         }
//         warpTopL(local_id, local_dist, warp_ids + warp * WARP_L, warp_dist + warp * WARP_L,
//         WARP_L,
//                  bp);
//         __syncthreads();

//         int total = wpb * WARP_L;
//         for (int i = tid; i < total; i += tpb) cand[i] = {warp_ids[i], warp_dist[i], false};
//         __syncthreads();

//         blockSelectTopK(cand, total, beam, bp.beam_capacity, bp);
//         __syncthreads();

//         if (beam[0].id < 0) {
//             if (tid == 0) s_qid = -1;
//             continue;
//         }

//         bool any_unexpanded = false;
//         for (int i = tid; i < bp.beam_capacity; i += tpb)
//             if (beam[i].id >= 0 && !beam[i].expanded) any_unexpanded = true;

//         if (!__syncthreads_or(any_unexpanded)) {
//             int* oid = out_ids + s_qid * k;
//             float* od = out_dists ? out_dists + s_qid * k : nullptr;

//             blockSelectTopK(beam, bp.beam_capacity, beam, k, bp);

//             if (tid < k) {
//                 oid[tid] = beam[tid].id;
//                 if (od) od[tid] = beam[tid].dist;
//             }

//             if (tid == 0) s_qid = -1;
//         }
//     }
// }

// Test_Result GPUFuncs::search_gpu(const std::vector<int>& start_node_ids, const float* queries,
//                                  int* ids_res, float* dists_res, int Q, int k, int batch,
//                                  cudaStream_t stream) {
//     if (k <= 0) k = bp.k;
//     if (stream == nullptr) stream = streams["search"].stream;
//     const int M = start_node_ids.size();
//     thrust::device_vector<int> d_start_node_ids(M), d_beam_qids(batch), d_init_beam_idxs(batch);
//     thrust::device_vector<bool> d_converged(batch);
//     cudaMemcpyAsync(d_start_node_ids.data().get(), start_node_ids.data(), M * sizeof(int),
//                     cudaMemcpyHostToDevice, stream);

//     int *beam_qids, *init_beam_idxs, *hops;
//     bool* converged;
//     cudaMallocHost(&beam_qids, batch * sizeof(int));
//     cudaMallocHost(&init_beam_idxs, batch * sizeof(int));
//     cudaMallocHost(&hops, sizeof(int));
//     cudaMallocHost(&converged, batch * sizeof(bool));
//     // 每个 Beam List 负责的查询 id（0 ~ Q-1）:
//     std::iota(beam_qids, beam_qids + batch, 0);
//     std::iota(init_beam_idxs, init_beam_idxs + batch, 0);

//     auto policy = thrust::cuda::par.on(stream);
//     std::atomic<int> unprocessed_qid{batch},  // 第一个未处理的查询 id
//         beam_init_num{batch};
//     const uint32_t blocks = std::min(batch, 2048), tpb = 512;
//     thrust::fill(policy, d_beam_visited.begin(), d_beam_visited.end(), 0);
//     auto e = gpu_record_time_start(stream);
//     float expand_time, total_time = 0;
//     size_t total_hops = 0;

//     while (true) {
//         if (beam_init_num > 0) {
//             cudaMemcpyAsync(d_beam_qids.data().get(), beam_qids, batch * sizeof(int),
//                             cudaMemcpyHostToDevice, stream);
//             cudaMemcpyAsync(d_init_beam_idxs.data().get(), init_beam_idxs,
//                             beam_init_num * sizeof(int), cudaMemcpyHostToDevice, stream);
//             beam_init_kernel<<<(beam_init_num.load() + tpb - 1) / tpb, tpb, bp.dim *
//             sizeof(float),
//                                stream>>>(d_start_node_ids.data().get(),
//                                          d_init_beam_idxs.data().get(),
//                                          d_beam_qids.data().get(), graph->get_gpu_vec_slots(),
//                                          queries, d_beams.data().get(),
//                                          d_beam_ends.data().get(), d_beam_visited.data().get(),
//                                          M, beam_init_num, bp);
//         }

//         thrust::device_vector<int> d_hops(1, 0);
//         beam_expand_kernel_opt<<<blocks, tpb, bp.dim * sizeof(float), stream>>>(
//             queries, d_beam_qids.data().get(), graph->get_gpu_slots(),
//             graph->get_gpu_vec_slots(), d_beams.data().get(), d_beam_ends.data().get(),
//             d_beam_visited.data().get(), d_converged.data().get(), d_hops.data().get(), batch,
//             bp);
//         // warp_beam_expand_kernel<<<blocks, tpb, bp.dim * sizeof(float), stream>>>(
//         //     queries, d_beam_qids.data().get(), graph->get_gpu_slots(),
//         //     graph->get_gpu_vec_slots(), d_beams.data().get(), d_beam_ends.data().get(),
//         //     d_beam_visited.data().get(), d_converged.data().get(), ids_res, dists_res,
//         //     d_hops.data().get(), batch, bp);
//         // beam_expand_kernel<<<blocks, tpb, bp.dim * sizeof(float), stream>>>(
//         //     queries, d_beam_qids.data().get(), graph->get_gpu_slots(),
//         //     graph->get_gpu_vec_slots(), d_beams.data().get(), d_beam_ends.data().get(),
//         //     d_beam_visited.data().get(), d_converged.data().get(), d_hops.data().get(),
//         batch,
//             //     bp);
//             cudaMemcpyAsync(converged, d_converged.data().get(), batch * sizeof(bool),
//                             cudaMemcpyDeviceToHost, stream);
//         CUDA_CHECK(cudaMemcpyAsync(hops, d_hops.data().get(), sizeof(int),
//         cudaMemcpyDeviceToHost,
//                                    stream));
//         // cudaStreamSynchronize(stream);
//         e = gpu_record_time_reset(e, stream, expand_time, "search expand");
//         total_hops += *hops;
//         total_time += expand_time;

//         beam_init_num = 0;
//         std::atomic<bool> converge{true};
// #pragma omp parallel for
//         for (int i = 0; i < batch; i++) {
//             const int qid = beam_qids[i];
//             if (qid < 0) {
//                 assert(unprocessed_qid >= Q);
//                 continue;
//             }
//             if (converged[i]) {
//                 id_res_extract_from_BCD(d_beams.data().get() + (size_t)i * bp.beam_size,
//                                         ids_res + qid * k, stream, k);
//                 if (dists_res)
//                     cudaMemcpyAsync(dists_res + qid * k,
//                                     d_beam_dists.data().get() + i * bp.beam_size,
//                                     k * sizeof(float), cudaMemcpyDeviceToDevice, stream);

//                 const int new_qid = unprocessed_qid.fetch_add(1);
//                 if (new_qid < Q) {
//                     if (new_qid % 10000 == 0) fo.print(TOS(100.0 * new_qid / Q) + "%
//                     finished."); beam_qids[i] = new_qid; init_beam_idxs[beam_init_num++] = i;
//                     thrust::fill(policy, d_beam_visited.begin() + (size_t)i * bp.visit_n,
//                                  d_beam_visited.begin() + (size_t)(i + 1) * bp.visit_n, 0);
//                 } else {
//                     beam_qids[i] = -1;
//                 }
//             } else
//                 converge = false;
//         }
//         if (converge && unprocessed_qid >= Q && beam_init_num == 0) break;

//         dist_extract_from_BCD(d_beams.data().get(), d_beam_dists.data().get(), stream,
//                               (size_t)batch * bp.beam_size, bp);

//         segmented_sort_pairs(d_beams.data().get(), d_beam_dists.data().get(),
//                              d_beams_sort.data().get(), d_beam_dists_sort.data().get(),
//                              d_beam_starts.data().get(), d_beam_ends.data().get(), stream,
//                              batch, bp.beam_size);
//     }

//     total_time += gpu_record_time_stop(e, stream);
//     cudaFreeHost(beam_qids);
//     cudaFreeHost(init_beam_idxs);
//     cudaFreeHost(hops);
//     cudaFreeHost(converged);
//     // CUDA_CHECK(cudaStreamSynchronize(stream));
//     return Test_Result(0, total_hops, total_time);
// }

// __global__ void beam_expand_kernel(const float* __restrict__ queries,
//                                    const int* __restrict__ beam_qids,
//                                    const Slot* __restrict__ slots,
//                                    const VecSlot* __restrict__ vec_slots, BCD* __restrict__
//                                    beams, uint32_t* __restrict__ visited, int* __restrict__
//                                    expand_nums, int* hops, int batch, BP bp) {
//     const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
//     extern __shared__ float sh_query[];
//     __shared__ int s_expand_num;
//     __shared__ int s_new_cand_num;
//     __shared__ float s_farthest_dist;
//     // uint32_t* s_bloom = (uint32_t*)(s_expand_num + 1);
//     for (int batch_i = bid; batch_i < batch; batch_i += gridDim.x) {
//         const int qid = beam_qids[batch_i];
//         if (qid < 0) continue;

//         BCD* beam = beams + (size_t)batch_i * bp.beam_size;
//         uint32_t* beam_visited = visited + (size_t)batch_i * bp.visit_n;
//         const float farthest = farthest_dist_d(bp);
//         for (int d = tid; d < bp.dim; d += tpb) sh_query[d] = queries[qid * bp.dim + d];
//         if (tid == 0) {
//             s_expand_num = 0;
//             s_new_cand_num = bp.beam_capacity;
//             s_farthest_dist =  // 找到当前 beam 中的最远距离
//                 beam[bp.beam_capacity - 1].id >= 0 ? beam[bp.beam_capacity - 1].dist : farthest;
//         }
//         __syncthreads();

//         for (int beam_i = tid; beam_i < bp.beam_capacity; beam_i += tpb) {
//             const int id = beam[beam_i].id;
//             const int deg = id < 0 ? -1 : slots[id].deg;
//             BCD* beam_new = beam + bp.beam_capacity + beam_i * bp.max_degree;
//             if (deg < 0 || beam[beam_i].expanded) {
//                 for (int i = 0; i < bp.max_degree; i++) {
//                     beam_new[i].id = -1;
//                     beam_new[i].dist = farthest;
//                 }
//                 continue;
//             }
//             assert(deg > 0);
//             beam[beam_i].expanded = true;

//             int new_node_num = 0;
//             const int* nbrs = slots[id].nbrs;
//             int new_ids[MAX_DEGREE];
//             float new_dists[MAX_DEGREE];
//             for (int nbr_i = 0; nbr_i < deg; nbr_i++) {
//                 const int nbrid = nbrs[nbr_i];
//                 assert(nbrid >= 0);
//                 // 若之前已经 visit 过，或者其到 query 的距离远于当前 beam 中的最远距离，则跳过
//                 if (atomicTestAndSetBit(beam_visited, nbrid)) continue;
//                 // if (atomicBloomTestAndSet3(beam_visited, bp.bloom_u32, nbrid)) continue;
//                 const float dist = calc_distance(sh_query, vec_slots[nbrid].vec, bp);
//                 if (compare_dist(dist, s_farthest_dist, bp)) continue;
//                 beam_new[new_node_num].id = nbrid;
//                 beam_new[new_node_num].dist = dist;
//                 beam_new[new_node_num++].expanded = false;
//             }
//             if (new_node_num > 0) {
//                 atomicAdd(hops, new_node_num);
//                 atomicAdd(&s_expand_num, 1);
//             }
//             for (; new_node_num < bp.max_degree; new_node_num++) {
//                 beam_new[new_node_num].id = -1;
//                 beam_new[new_node_num].dist = farthest;
//             }
//         }
//         __syncthreads();

//         if (tid == 0) {
//             expand_nums[batch_i] = s_expand_num;
//             // if (s_expand_num == 0) assert(beam[bp.beam_capacity - 1].id >= 0);
//         }
//         __syncthreads();
//     }
// }

// __global__ void beam_init_kernel(const int* __restrict__ start_node_ids,
//                                  const int* __restrict__ init_beam_idxs,
//                                  const int* __restrict__ beam_qids,
//                                  const VecSlot* __restrict__ vec_slots,
//                                  const float* __restrict__ queries, BCD* __restrict__ beams,
//                                  uint32_t* __restrict__ visited, int M, int init_num, BP bp) {
//     const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
//     extern __shared__ float s_query[];
//     for (int beam_i = bid; beam_i < init_num; beam_i += gridDim.x) {
//         const int beam_idx = init_beam_idxs[beam_i], qid = beam_qids[beam_idx];
//         assert(beam_idx >= 0);
//         assert(qid >= 0);
//         BCD* init_beams = beams + (size_t)beam_idx * bp.beam_size;
//         uint32_t* beam_visited = visited + (size_t)beam_idx * bp.visit_n;
//         for (int d = tid; d < bp.dim; d += tpb) s_query[d] = queries[qid * bp.dim + d];
//         __syncthreads();

//         for (int i = tid; i < bp.beam_capacity; i += tpb) {
//             BCD* beam = init_beams + i;
//             if (i >= M) {
//                 beam->id = -1;
//                 beam->dist = farthest_dist_d(bp);
//             } else {
//                 const int id = beam->id = start_node_ids[i];
//                 assert(id >= 0);
//                 assert(id < bp.base_n);
//                 atomicTestAndSetBit(beam_visited, id);
//                 // atomicBloomTestAndSet3(beam_visited, bp.bloom_u32, id);
//                 beam->dist = calc_distance(s_query, vec_slots[id].vec, bp);
//                 beam->expanded = false;  // 初始化为未扩展
//             }
//         }
//     }
// }

// __global__ void beam_init_kernel(const int* __restrict__ start_node_ids,
//                                  const int* __restrict__ init_beam_idxs,
//                                  const int* __restrict__ beam_qids,
//                                  const float* __restrict__ vec_slots,
//                                  const float* __restrict__ queries, BCD* __restrict__ beams,
//                                  size_t* __restrict__ beam_ends, uint32_t* __restrict__ visited,
//                                  int M, int init_num, BP bp) {
//     const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
//     for (int beam_i = bid * tpb + tid; beam_i < init_num; beam_i += gridDim.x * tpb) {
//         const int beam_idx = init_beam_idxs[beam_i], qid = beam_qids[beam_idx];
//         BCD* init_beams = beams + (size_t)beam_idx * bp.beam_size;
//         uint32_t* beam_visited = visited + (size_t)beam_idx * bp.visit_n;
//         float query[DIM];
//         for (int d = 0; d < bp.dim; d++) query[d] = queries[qid * bp.dim + d];
//         beam_ends[beam_idx] = (size_t)beam_idx * bp.beam_size + M;

//         for (int i = 0; i < M; i++) {
//             BCD* beam = init_beams + i;
//             const int id = beam->id = start_node_ids[i];
//             assert(id >= 0);
//             atomicTestAndSetBit(beam_visited, id);
//             beam->dist = calc_distance(query, vec_slots + id * bp.dim, bp);
//             beam->expanded = false;  // 初始化为未扩展
//         }
//     }
// }

// __global__ void beam_expand_kernel(const float* __restrict__ queries,
//                                    const int* __restrict__ beam_qids,
//                                    const Slot* __restrict__ slots,
//                                    const float* __restrict__ vec_slots, BCD* __restrict__ beams,
//                                    size_t* __restrict__ beam_ends, uint32_t* __restrict__
//                                    visited, bool* __restrict__ batch_converged, int*
//                                    __restrict__ hops, int batch, BP bp) {
//     const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
//     extern __shared__ float sh_query[];
//     __shared__ int s_new_cand_idx;
//     __shared__ int s_beam_num;
//     __shared__ float s_farthest_dist;
//     // uint32_t* s_bloom = (uint32_t*)(s_expand_num + 1);
//     for (int batch_i = bid; batch_i < batch; batch_i += gridDim.x) {
//         const int qid = beam_qids[batch_i];
//         if (qid < 0) continue;

//         const size_t offset = (size_t)batch_i * bp.beam_size;
//         BCD* beam = beams + offset;
//         uint32_t* beam_visited = visited + (size_t)batch_i * bp.visit_n;
//         const float farthest = farthest_dist_d(bp);
//         for (int d = tid; d < bp.dim; d += tpb) sh_query[d] = queries[qid * bp.dim + d];
//         if (tid == 0) {
//             s_beam_num = beam_ends[batch_i] - offset;
//             if (s_beam_num > bp.beam_capacity) {
//                 s_beam_num = bp.beam_capacity;
//                 s_farthest_dist = beam[bp.beam_capacity - 1].dist;  // 找到当前 beam
//                 中的最远距离
//             } else
//                 s_farthest_dist = farthest;

//             s_new_cand_idx = s_beam_num;
//         }
//         __syncthreads();

//         for (int beam_i = tid; beam_i < s_beam_num; beam_i += tpb) {
//             const int id = beam[beam_i].id;
//             const int deg = id < 0 ? -1 : slots[id].deg;
//             if (!deg)
//                 printf("beam_expand_kernel ERROR: beam_i: %d, id: %d, deg: %d", beam_i, id,
//                 deg);
//             if (deg <= 0 || beam[beam_i].expanded) continue;
//             // assert(deg > 0);
//             beam[beam_i].expanded = true;

//             const int* nbrs = slots[id].nbrs;
//             int new_ids[MAX_DEGREE], new_node_num = 0;
//             float new_dists[MAX_DEGREE];
//             for (int nbr_i = 0; nbr_i < deg; nbr_i++) {
//                 const int nbrid = nbrs[nbr_i];
//                 // 若之前已经 visit 过，或者其到 query 的距离远于当前 beam 中的最远距离，则跳过
//                 if (atomicTestAndSetBit(beam_visited, nbrid)) continue;
//                 const float dist = calc_distance(sh_query, vec_slots + nbrid * bp.dim, bp);
//                 if (compare_dist(dist, s_farthest_dist, bp)) continue;
//                 new_ids[new_node_num] = nbrid;
//                 new_dists[new_node_num++] = dist;
//             }
//             if (new_node_num > 0) {
//                 // atomicAdd(&s_expand_num, 1);
//                 const int start = atomicAdd(&s_new_cand_idx, new_node_num);
//                 assert(start + new_node_num <= bp.beam_size);
//                 BCD* beam_new = beam + start;
//                 for (int i = 0; i < new_node_num; i++) {
//                     beam_new[i].id = new_ids[i];
//                     beam_new[i].dist = new_dists[i];
//                     beam_new[i].expanded = false;
//                 }
//             }
//         }
//         __syncthreads();

//         if (tid == 0) {
//             // assert(s_new_cand_idx <= bp.beam_size);
//             const int new_cand_num = s_new_cand_idx - s_beam_num;
//             if (new_cand_num > 0) {
//                 atomicAdd(hops, new_cand_num);
//                 batch_converged[batch_i] = false;
//                 beam_ends[batch_i] = (size_t)offset + s_new_cand_idx;
//             } else {
//                 batch_converged[batch_i] = true;
//                 beam_ends[batch_i] = (size_t)offset;
//             }
//         }
//         __syncthreads();
//     }
// }

// __global__ void warp_beam_expand_kernel(
//     const float* __restrict__ queries, const int* __restrict__ beam_qids,
//     const Slot* __restrict__ slots, const float* __restrict__ vec_slots, BCD* __restrict__
//     beams, size_t* __restrict__ beam_ends, uint32_t* __restrict__ visited, bool* __restrict__
//     batch_converged, int* __restrict__ ids_res, float* __restrict__ dists_res, int* __restrict__
//     hops, int batch, int k, BP bp) {
//     // ===== warp / thread info =====
//     const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x, wpb = blockDim.x >> 5;
//     const int lane = tid & 31;  // warp 内线程 id
//     const int warp = tid >> 5;  // block 内 warp id

//     extern __shared__ float sh_query[];  // 大小 >= bp.dim
//     __shared__ int s_new_cand_idx;       // 新候选 beam 写入起点
//     __shared__ int s_beam_num;      // 原 beam 数
//     __shared__ float s_farthest_dist;    // 当前 beam 中最远距离

//     for (int batch_i = bid; batch_i < batch; batch_i += gridDim.x) {
//         const int qid = beam_qids[batch_i];
//         if (qid < 0) continue;

//         const size_t offset = (size_t)batch_i * bp.beam_size;
//         BCD* beam = beams + offset;
//         uint32_t* beam_visited = visited + (size_t)batch_i * bp.visit_n;
//         for (int d = tid; d < bp.dim; d += tpb) sh_query[d] = queries[qid * bp.dim + d];

//         if (tid == 0) {  // 初始化 warp 共享统计变量
//             s_beam_num = (int)(beam_ends[batch_i] - offset);
//             if (s_beam_num > bp.beam_capacity) {
//                 assert(beam[bp.beam_capacity - 1].id >= 0);
//                 s_beam_num = bp.beam_capacity;
//                 s_farthest_dist = beam[bp.beam_capacity - 1].dist;
//             } else
//                 s_farthest_dist = farthest_dist_d(bp);

//             s_new_cand_idx = s_beam_num;
//         }
//         __syncthreads();

//         // warp-per-beam-node: warp 并行处理 beam 中每个节点
//         for (int beam_i = warp; beam_i < s_beam_num; beam_i += wpb) {
//             // lane0 获取节点 id 和 degree，并广播给 warp
//             int id = -1, deg = -1;
//             if (lane == 0)
//                 if (beam[beam_i].id >= 0 && !beam[beam_i].expanded) {
//                     id = beam[beam_i].id;
//                     deg = slots[id].deg;
//                     if (deg == 0) printf("ERROR: beam_i: %d, id: %d, deg: %d\n", beam_i, id,
//                     deg); beam[beam_i].expanded = true;  // warp 独占，无需原子
//                 }
//             id = __shfl_sync(0xffffffff, id, 0);
//             deg = __shfl_sync(0xffffffff, deg, 0);
//             if (deg <= 0) continue;

//             const Slot& slot = slots[id];
//             // 串行处理每个 nbr，warp 内 32 线程并行计算每个 nbr 的距离
//             for (int nbr_i = 0; nbr_i < deg; ++nbr_i) {
//                 const int nbrid = slot.nbrs[nbr_i];
//                 bool was_visited = false;
//                 if (lane == 0) was_visited = atomicTestAndSetBit(beam_visited, nbrid);
//                 was_visited = __shfl_sync(0xffffffff, was_visited, 0);
//                 if (was_visited) continue;

//                 float dist = 0.0f;
//                 const float* base = vec_slots + (size_t)nbrid * bp.dim;
//                 // lane-per-dim 计算距离
//                 for (int d = lane; d < bp.dim; d += 32) {
//                     const float diff = sh_query[d] - base[d];
//                     dist += (bp.metric == DIST_METRIC::L2_) ? diff * diff : sh_query[d] *
//                     base[d];
//                 }
//                 // warp reduce sum
//                 for (int offset = 16; offset > 0; offset >>= 1)
//                     dist += __shfl_down_sync(0xffffffff, dist, offset);

//                 if (lane == 0 && compare_dist(s_farthest_dist, dist, bp)) {
//                     int out_idx = atomicAdd(&s_new_cand_idx, 1);
//                     assert(out_idx < bp.beam_size);
//                     beam[out_idx].id = nbrid;
//                     beam[out_idx].dist = dist;
//                     beam[out_idx].expanded = false;
//                 }
//             }
//         }
//         __syncthreads();

//         if (s_new_cand_idx == s_beam_num) {
//             int* ids_out = ids_res + qid * k;
//             for (int i = tid; i < k; i += tpb) ids_out[i] = beam[i].id;
//             if (dists_res != nullptr) {
//                 float* dists_out = dists_res + qid * k;
//                 for (int i = tid; i < k; i += tpb) dists_out[i] = beam[i].dist;
//             }
//             if (tid == 0) {
//                 batch_converged[batch_i] = true;
//                 beam_ends[batch_i] = offset;
//             }
//         } else if (tid == 0) {
//             atomicAdd(hops, s_new_cand_idx - s_beam_num);
//             batch_converged[batch_i] = false;
//             beam_ends[batch_i] = offset + s_new_cand_idx;
//         }
//         __syncthreads();
//     }
// }

// __global__ void beam_expand_kernel_opt(
//     const float* __restrict__ queries, const int* __restrict__ beam_qids,
//     const Slot* __restrict__ slots, const float* __restrict__ vec_slots, BCD* __restrict__
//     beams, size_t* __restrict__ beam_ends, uint32_t* __restrict__ visited, bool* __restrict__
//     batch_converged, int* __restrict__ hops, int batch, BP bp) { const int bid = blockIdx.x, tid
//     = threadIdx.x, warps_per_block = blockDim.x >> 5; const int lane = tid & 31;  // warp
//     内线程号 const int warp = tid >> 5;  // block 内 warp 号

//     extern __shared__ float sh_query[];
//     __shared__ int s_new_cand_idx;     // beam 中新候选写入起点
//     __shared__ int s_beam_num;    // 原 beam 数
//     __shared__ float s_farthest_dist;  // 当前 beam 中最远距离

//     // block 轮询 batch
//     for (int batch_i = bid; batch_i < batch; batch_i += gridDim.x) {
//         const int qid = beam_qids[batch_i];
//         if (qid < 0) continue;

//         const size_t offset = (size_t)batch_i * bp.beam_size;
//         const float farthest = farthest_dist_d(bp);
//         BCD* beam = beams + offset;
//         uint32_t* beam_visited = visited + (size_t)batch_i * bp.visit_n;

//         for (int d = tid; d < bp.dim; d += blockDim.x) sh_query[d] = queries[qid * bp.dim + d];
//         if (tid == 0) {
//             s_beam_num = (int)(beam_ends[batch_i] - offset);
//             if (s_beam_num > bp.beam_capacity) {
//                 s_beam_num = bp.beam_capacity;
//                 s_farthest_dist = beam[bp.beam_capacity - 1].dist;
//             } else
//                 s_farthest_dist = farthest;

//             s_new_cand_idx = s_beam_num;
//         }
//         __syncthreads();

//         // warp-per-beam-node 扩展
//         for (int beam_i = warp; beam_i < s_beam_num; beam_i += warps_per_block) {
//             int id = -1, deg = -1;
//             if (lane == 0)
//                 if (beam[beam_i].id >= 0 && !beam[beam_i].expanded) {
//                     id = beam[beam_i].id;
//                     deg = slots[id].deg;
//                     beam[beam_i].expanded = true;  // 本 warp 独占，无需原子
//                 }
//             id = __shfl_sync(0xffffffff, id, 0);
//             deg = __shfl_sync(0xffffffff, deg, 0);
//             if (deg <= 0) continue;

//             const Slot& slot = slots[id];
//             for (int nbr_i = lane; nbr_i < deg; nbr_i += 32) {  // warp 内遍历邻居
//                 bool valid = false;
//                 int nbrid = slot.nbrs[nbr_i], cnt;
//                 float dist;
//                 if (atomicTestAndSetBit(beam_visited, nbrid) == 0) {
//                     dist = calc_distance(vec_slots + (size_t)nbrid * bp.dim, sh_query, bp);
//                     valid = compare_dist(s_farthest_dist, dist, bp);
//                 }

//                 // warp 压缩写入 beam
//                 const unsigned active = __activemask(), mask = __ballot_sync(active, valid);
//                 if (lane == 0) cnt = __popc(mask);
//                 cnt = __shfl_sync(active, cnt, 0);
//                 if (cnt > 0) {
//                     int base_idx;
//                     if (lane == 0) base_idx = atomicAdd(&s_new_cand_idx, cnt);
//                     base_idx = __shfl_sync(active, base_idx, 0);
//                     if (valid) {
//                         const int out_idx = base_idx + __popc(mask & ((1u << lane) - 1));
//                         assert(out_idx < bp.beam_size);
//                         beam[out_idx].id = nbrid;
//                         beam[out_idx].dist = dist;
//                         beam[out_idx].expanded = false;
//                     }
//                 }
//             }
//         }
//         __syncthreads();

//         if (tid == 0) {
//             const int new_cand_num = s_new_cand_idx - s_beam_num;
//             if (new_cand_num > 0) {
//                 atomicAdd(hops, new_cand_num);
//                 batch_converged[batch_i] = false;
//                 beam_ends[batch_i] = offset + s_new_cand_idx;
//             } else {
//                 batch_converged[batch_i] = true;
//                 beam_ends[batch_i] = offset;
//             }
//         }
//         __syncthreads();
//     }
// }

// __global__ void heap_beam_search_kernel(
//     const int* __restrict__ start_node_ids, const float* __restrict__ queries,
//     const Slot* __restrict__ slots, const float* __restrict__ vec_slots, BCD* __restrict__
//     beams, uint32_t* __restrict__ visited, int* __restrict__ ids_res, float* __restrict__
//     dists_res, int* __restrict__ hops, int* __restrict__ unprocessed_qid, int k, int Q, int
//     warp_l, BP bp) { const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x, wpb = tpb
//     >> 5; const int lane = tid & 31, warp = tid >> 5;  // warp 内线程号和 block 内 warp 号 const
//     float farthest = farthest_dist_d(bp); const int pq = 0;

//     extern __shared__ float sh_query[];
//     float *sh_dists = sh_query + bp.dim, *sh_warp_dists = sh_dists + tpb;
//     int *sh_idxs = (int*)(sh_warp_dists + wpb * warp_l), *sh_warp_ids = sh_idxs + tpb;
//     __shared__ int s_qid, s_hop, s_any_insert;

//     if (tid == 0) s_qid = -1;

//     uint32_t* beam_visited = visited + (size_t)bid * bp.visit_n;
//     BCD* beam = beams + (size_t)bid * bp.beam_capacity;
//     while (true) {
//         __syncthreads();
//         if (s_qid < 0) {
//             if (tid == 0) {  // 获取新的 qid
//                 s_qid = atomicAdd(unprocessed_qid, 1);
//                 if (s_qid >= Q)
//                     s_qid = -1;  // 所有 qid 都处理完了
//                 else
//                     s_hop = 0;
//             }
//             __syncthreads();
//             if (s_qid < 0) return;
//             // 初始化：
//             for (int i = tid; i < bp.visit_n; i += tpb) beam_visited[i] = 0;
//             for (int d = tid; d < bp.dim; d += tpb) sh_query[d] = queries[s_qid * bp.dim + d];
//             __syncthreads();

//             for (int start_i = warp; start_i < bp.beam_capacity; start_i += wpb) {
//                 int start_id = -1;
//                 if (lane == 0) {
//                     start_id = start_node_ids[start_i];
//                     if (atomicTestAndSetBit(beam_visited, start_id)) {
//                         start_id = -1;
//                         beam[start_i] = {-1, farthest, true};
//                     }
//                 }
//                 start_id = __shfl_sync(0xffffffff, start_id, 0);
//                 if (start_id < 0) continue;

//                 float dist =
//                     warp_calc_dist(vec_slots + (size_t)start_id * bp.dim, sh_query, lane, bp);
//                 if (lane == 0) beam[start_i] = {start_id, dist, false};
//             }
//             __syncthreads();
//             if (tid == 0)
//                 for (int i = bp.beam_capacity / 2 - 1; i >= 0; i--)
//                     heap_sift_down(beam, i, bp.beam_capacity, bp);
//         }
//         __syncthreads();

//         for (int i = tid; i < wpb * warp_l; i += tpb)
//             sh_warp_ids[i] = -1, sh_warp_dists[i] = farthest;
//         __syncthreads();
//         for (int beam_i = warp; beam_i < bp.beam_capacity; beam_i += wpb) {
//             int id = -1, deg = -1;
//             if (lane == 0 && !beam[beam_i].expanded) {
//                 id = beam[beam_i].id;
//                 deg = slots[id].deg;
//                 beam[beam_i].expanded = true;
//                 // if (!s_qid) printf("(%d %d) ", id, deg);
//             }
//             id = __shfl_sync(0xffffffff, id, 0);
//             deg = __shfl_sync(0xffffffff, deg, 0);
//             if (deg <= 0) continue;

//             const int* nbrs = slots[id].nbrs;
//             int local_id = -1;
//             float local_dist = farthest;
//             for (int nbr_i = 0; nbr_i < deg; nbr_i++) {  // warp 内遍历邻居
//                 int nbrid = -1;
//                 if (lane == 0) {
//                     nbrid = nbrs[nbr_i];
//                     // if (!s_qid) printf("(%d %d) ", nbr_i, nbrid);
//                     if (atomicTestAndSetBit(beam_visited, nbrid)) nbrid = -1;
//                 }
//                 nbrid = __shfl_sync(0xffffffff, nbrid, 0);
//                 if (nbrid < 0) continue;
//                 const float dist =
//                     warp_calc_dist(vec_slots + (size_t)nbrid * bp.dim, sh_query, lane, bp);
//                 // if (!s_qid && !lane) printf("(%d %f) ", nbrid, dist);
//                 if (lane == nbr_i % warpSize && compare_dist(local_dist, dist, bp)) {
//                     local_id = nbrid;
//                     local_dist = dist;
//                 }
//             }
//             warpTopL(local_id, local_dist, sh_warp_ids + warp * warp_l,
//                      sh_warp_dists + warp * warp_l, lane, warp_l, bp);
//         }
//         __syncthreads();
//         if (tid == 0) {
//             s_any_insert = 0;
//             for (int i = 0; i < wpb * warp_l; i++) {
//                 const int id = sh_warp_ids[i];
//                 // if (!s_qid) printf("(%d %f) ", id, sh_warp_dists[i]);
//                 if (id < 0) continue;
//                 const float dist = sh_warp_dists[i];
//                 if (compare_dist(beam[0].dist, dist, bp)) {
//                     beam[0] = {id, dist, false};
//                     heap_sift_down(beam, 0, bp.beam_capacity, bp);
//                     s_any_insert = 1;
//                     s_hop++;
//                 }
//             }
//             if (!s_qid) printf("\nhop: %d\n", s_hop);
//         }
//         __syncthreads();

//         if (s_any_insert == 0) {
//             const int total = bp.beam_capacity;
//             int* ids_out = ids_res + s_qid * k;
//             float* dists_out = dists_res ? dists_res + s_qid * k : nullptr;
//             for (int i = 0; i < k; i++) {
//                 int local_idx = -1;
//                 float local_best = farthest;
//                 for (int j = tid; j < total; j += tpb) {
//                     if (beam[j].id < 0) continue;
//                     const float dist = beam[j].dist;
//                     if (compare_dist(local_best, dist, bp)) {
//                         local_best = dist;
//                         local_idx = j;
//                     }
//                 }
//                 sh_dists[tid] = local_best;
//                 sh_idxs[tid] = local_idx;
//                 __syncthreads();

//                 for (int offset = tpb >> 1; offset > 0; offset >>= 1) {
//                     if (tid < offset && compare_dist(sh_dists[tid], sh_dists[tid + offset], bp))
//                     {
//                         sh_dists[tid] = sh_dists[tid + offset];
//                         sh_idxs[tid] = sh_idxs[tid + offset];
//                     }
//                     __syncthreads();
//                 }
//                 if (tid == 0) {
//                     const int ith_idx = sh_idxs[0];
//                     ids_out[i] = beam[ith_idx].id;
//                     if (dists_out) dists_out[i] = sh_dists[0];
//                     beam[ith_idx].id = -1;
//                 }
//                 __syncthreads();
//             }
//             if (tid == 0) {
//                 hops[s_qid] = s_hop;
//                 s_qid = -1;
//             }
//         }
//     }
// }
