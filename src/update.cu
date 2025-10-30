#include <cfloat>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>

#include "fileout.h"
#include "gpufuncs.cuh"
#include "utils.cuh"

namespace efanna2e {
__global__ void find_closest_paired_query_kernel(const int *__restrict__ knns,
                                                 const int *__restrict__ base_query_ids,
                                                 const float *__restrict__ queries,
                                                 const float *__restrict__ insert_vectors,
                                                 int *__restrict__ paired_query_ids_res, int batch,
                                                 BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    extern __shared__ unsigned char smem[];
    int *sh_paired_qid = (int *)smem;
    float *sh_paired_query_dist = (float *)(sh_paired_qid + 1);
    for (int ins_idx = bid; ins_idx < batch; ins_idx += gridDim.x) {
        if (!tid) *sh_paired_query_dist = farthest_dist_d(bp);
        __syncthreads();

        for (int i = tid; i < bp.k; i += tpb) {
            const int qid = base_query_ids[knns[ins_idx * bp.k + i]];
            const float *query = queries + qid * bp.dim;
            const float dist = calc_distance(query, insert_vectors + ins_idx * bp.dim,
                                             bp);  // 插入与查询向量的距离
            atomicClosestFloat(sh_paired_query_dist, dist, bp);
            __threadfence();  // 刷写全局/共享内存，使其他 block 可见
            if (*sh_paired_query_dist - dist < EPS) sh_paired_qid[i] = qid;
        }
        __syncthreads();

        if (!tid) paired_query_ids_res[ins_idx] = *sh_paired_qid;
    }
}

__global__ void base_insert_kernel(const float *__restrict__ insert_vectors,
                                   float *__restrict__ base, int batch, int base_n, BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    for (int ins_idx = bid; ins_idx < batch; ins_idx += gridDim.x)
        for (int d = tid; d < bp.dim; d += tpb)
            base[(base_n + ins_idx) * bp.dim + d] = insert_vectors[ins_idx * bp.dim + d];
}

__global__ void k_plus_ones_build_kernel(const int *__restrict__ query_knns,
                                         const int *__restrict__ paired_query_ids,
                                         int *__restrict__ k_plus_ones_res, int batch, int base_n,
                                         BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    for (int ins_idx = bid; ins_idx < batch; ins_idx += gridDim.x) {
        for (int i = tid; i < bp.k + 1; i += tpb) {
            k_plus_ones_res[ins_idx * (bp.k + 1) + i] =
                i < bp.k ? query_knns[paired_query_ids[ins_idx] * bp.k + i] : base_n + ins_idx;
        }
    }
}

__global__ void get_from_edge_num_kernel(const int *__restrict__ graph,
                                         const int *__restrict__ graph_deg,
                                         int *__restrict__ from_edge_nums_res, int base_n, BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    for (int base_id = bid; base_id < base_n; base_id += gridDim.x)
        for (int nbr_i = tid; nbr_i < graph_deg[base_id]; nbr_i += tpb)
            atomicAdd(from_edge_nums_res + graph[base_id * bp.max_degree + nbr_i], 1);
}

__global__ void get_from_vectors_kernel(const int *__restrict__ graph,
                                        const int *__restrict__ graph_deg,
                                        const int *__restrict__ del_ids,
                                        int *__restrict__ from_ids_res,
                                        int *__restrict__ from_nums_res, int batch, int base_n,
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
    const float *__restrict__ base, const int *__restrict__ graph,
    const float *__restrict__ graph_dist, const int *__restrict__ graph_deg,
    const int *__restrict__ del_ids, CN *__restrict__ cand_nbrs, int *__restrict__ pivot_ids,
    int *pivot_idx, int batch, int base_n, BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x, cand_size = bp.max_degree * 2;
    extern __shared__ unsigned char smem[];
    int *sh_to_vec_ids = (int *)smem, *sh_to_vec_num = sh_to_vec_ids + bp.max_degree;

    for (int del_idx = bid; del_idx < batch; del_idx += gridDim.x) {
        const int del_id = del_ids[del_idx];
        for (int i = tid; i < graph_deg[del_id]; i += tpb)
            sh_to_vec_ids[i] = graph[del_id * bp.max_degree + i];
        if (!tid) *sh_to_vec_num = graph_deg[del_id];
        __syncthreads();

        for (int base_id = tid; base_id < base_n; base_id += tpb) {
            bool is_from = false;
            int graph_idx = base_id * bp.max_degree;
            for (int nbr_i = 0; nbr_i < graph_deg[base_id]; nbr_i++) {
                if (graph[graph_idx + nbr_i] == del_id) {
                    is_from = true;
                    break;
                }
            }
            if (is_from) {
                const int idx = atomicAdd(pivot_idx, 1), cand_idx = idx * cand_size,
                          old_nbr_num = graph_deg[base_id],
                          total_nbr_num = old_nbr_num + *sh_to_vec_num;
                pivot_ids[idx] = base_id;
                for (int nbr_i = 0; nbr_i < old_nbr_num; nbr_i++) {
                    cand_nbrs[cand_idx + nbr_i].idx = nbr_i;
                    cand_nbrs[cand_idx + nbr_i].id = graph[graph_idx + nbr_i];
                    cand_nbrs[cand_idx + nbr_i].status = Initial;
                    cand_nbrs[cand_idx + nbr_i].dist = graph_dist[graph_idx + nbr_i];
                }
                for (int nbr_i = graph_deg[base_id]; nbr_i < total_nbr_num; nbr_i++) {
                    cand_nbrs[cand_idx + nbr_i].idx = nbr_i;
                    cand_nbrs[cand_idx + nbr_i].id = sh_to_vec_ids[nbr_i];
                    cand_nbrs[cand_idx + nbr_i].status = Initial;
                    cand_nbrs[cand_idx + nbr_i].dist =
                        calc_distance(base, base_id, sh_to_vec_ids[nbr_i], bp);
                }
                for (int nbr_i = total_nbr_num; nbr_i < cand_size; nbr_i++) {
                    cand_nbrs[cand_idx + nbr_i].idx = nbr_i;
                    cand_nbrs[cand_idx + nbr_i].id = -1;
                    cand_nbrs[cand_idx + nbr_i].status = Discarded;
                    cand_nbrs[cand_idx + nbr_i].dist = farthest_dist_d(bp);
                }
            }
        }
    }
}

__global__ void candidate_ignore_kernel(const float *__restrict__ base, CN *__restrict__ cand_nbrs,
                                        const int *__restrict__ sort_idxs,
                                        int *__restrict__ nbr_nums, int new_nbr_i, int pivot_num,
                                        BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x, cand_size = bp.max_degree * 2;
    for (int pivot_i = bid; pivot_i < pivot_num; pivot_i += gridDim.x) {
        CN *new_nbr = get_CN(cand_nbrs, sort_idxs, new_nbr_i, 0, pivot_i, 0, cand_size);
        // 无效节点或邻居已满，直接淘汰
        if (new_nbr->id == -1 || nbr_nums[pivot_i] >= bp.max_degree)
            new_nbr->status = Discarded;
        else if (new_nbr->status == Initial)  // 待定节点，保留且用于淘汰后续节点
        {
            new_nbr->status = Retained;
            nbr_nums[pivot_i]++;
            for (int delta = tid + 1; delta < cand_size - new_nbr_i; delta += tpb) {
                CN *wti_cand_nbr = get_CN(cand_nbrs, sort_idxs, new_nbr_i + delta, 0, pivot_i, 0,
                                          cand_size);  // 待淘汰候选邻居
                if (wti_cand_nbr->status == Initial) {
                    // 如果新邻居和待淘汰候选邻居是同一个节点，则直接淘汰
                    if (new_nbr->id == wti_cand_nbr->id) wti_cand_nbr->status = Repeated;
                    // 否则比较 pivot 到待淘汰候选邻居的距离和新邻居到待淘汰候选邻居距离
                    else if (compare_dist(wti_cand_nbr->dist,
                                          calc_distance(base, wti_cand_nbr->id, new_nbr->id, bp),
                                          bp))
                        wti_cand_nbr->status = Discarded;  // 淘汰
                }
            }
        }
    }
}

__global__ void update_new_nbrs_kernel(const int *__restrict__ vec_ids,
                                       const int *__restrict__ sort_idxs, int *__restrict__ graph,
                                       int *__restrict__ graph_deg, float *__restrict__ graph_dist,
                                       CN *__restrict__ cand_nbrs,
                                       int *__restrict__ from_edge_nums, int pivot_num, BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x, cand_size = 2 * bp.max_degree;
    extern __shared__ unsigned char sh_mem[];
    int *sh_nbr_num = (int *)sh_mem;
    for (int i = bid; i < pivot_num; i += gridDim.x) {
        if (!tid) *sh_nbr_num = 0;
        __syncthreads();

        const int base_id = vec_ids[i], idx_start = base_id * bp.max_degree;

        for (int old_nbr_i = tid; old_nbr_i < graph_deg[base_id]; old_nbr_i += tpb)
            atomicSub(from_edge_nums + graph[idx_start + old_nbr_i], 1);

        for (int j = tid; j < cand_size && *sh_nbr_num < bp.max_degree; j += tpb) {
            const CN *cand_nbr = get_CN(cand_nbrs, sort_idxs, j, 0, i, 0, cand_size);
            if (cand_nbr->status == Retained) {
                const int nbr_i = atomicAdd(sh_nbr_num, 1);
                if (nbr_i >= bp.max_degree) break;

                const int idx = idx_start + nbr_i;
                graph[idx] = cand_nbr->id;  // assert(cand_nbr.id != -1);
                graph_dist[idx] = cand_nbr->dist;

                atomicAdd(from_edge_nums + cand_nbr->id, 1);
            }
        }
        __syncthreads();

        for (int j = tid; j < cand_size && *sh_nbr_num < bp.max_degree; j += tpb) {
            const CN *cand_nbr = get_CN(cand_nbrs, sort_idxs, j, 0, i, 0, cand_size);
            if (cand_nbr->id >= 0 && cand_nbr->status == Discarded) {
                const int nbr_i = atomicAdd(sh_nbr_num, 1);
                if (nbr_i >= bp.max_degree) break;

                const int idx = idx_start + nbr_i;
                graph[idx] = cand_nbr->id;
                graph_dist[idx] = cand_nbr->dist;
                atomicAdd(from_edge_nums + cand_nbr->id, 1);
            }
        }
        __syncthreads();

        if (!tid) graph_deg[base_id] = *sh_nbr_num < bp.max_degree ? *sh_nbr_num : bp.max_degree;
    }
}

void GPUFuncs::vector_delete(int *del_ids, int batch) {
    const std::string name = "ins_del";
    const int base_n = base_n_upd.load();
    cudaStream_t &stream = streams[name].stream;

    event_record_time_start(name);

    thrust::device_vector<int> d_del_ids(batch);
    CUDA_CHECK(cudaMemcpyAsync(d_del_ids.data().get(), del_ids, batch * sizeof(int),
                               cudaMemcpyHostToDevice, stream));

    std::vector<int> h_from_edge_nums(base_n);
    CUDA_CHECK(cudaMemcpy(h_from_edge_nums.data(), from_edge_nums.data().get(),
                          base_n * sizeof(int), cudaMemcpyDeviceToHost));

    int total_from_vec_num = 0, cand_size = 2 * bp.max_degree;
    for (int i = 0; i < batch; i++) total_from_vec_num += h_from_edge_nums[del_ids[i]];

    const int total_cand_size = total_from_vec_num * cand_size;
    thrust::device_vector<int> from_ids(total_from_vec_num), cand_idxs(total_cand_size),
        cand_idxs_sort(total_cand_size), offsets(total_from_vec_num + 1),
        nbr_nums(total_from_vec_num, 0);
    thrust::device_vector<float> cand_dists(total_cand_size), cand_dists_sort(total_cand_size);
    thrust::device_vector<CN> cand_nbrs(total_cand_size);
    int *from_idx;
    CUDA_CHECK(cudaMallocAsync(&from_idx, sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(from_idx, 0, sizeof(int), stream));

    vector_deletion_patch_kernel<<<batch, 256, (bp.max_degree + 1) * sizeof(int), stream>>>(
        d_b.data().get(), graph.data().get(), graph_dist.data().get(), graph_deg.data().get(),
        d_del_ids.data().get(), cand_nbrs.data().get(), from_ids.data().get(), from_idx, batch,
        base_n, bp);

    CUDA_CHECK(cudaFreeAsync(from_idx, stream));

    idx_dist_extract(cand_nbrs.data().get(), cand_idxs.data().get(), cand_dists.data().get(),
                     stream, total_cand_size, bp);
    auto res = segmented_sort_pairs(cand_idxs.data().get(), cand_dists.data().get(),
                                    cand_idxs_sort.data().get(), cand_dists_sort.data().get(),
                                    offsets.data().get(), stream, total_cand_size, cand_size);
    for (int i = 0; i < cand_size; ++i)
        candidate_ignore_kernel<<<batch, 32, 0, stream>>>(d_b.data().get(), cand_nbrs.data().get(),
                                                          res.first, nbr_nums.data().get(), i,
                                                          total_from_vec_num, bp);
    {
        std::unique_lock<std::shared_mutex> lock(graph_mtx);
        update_new_nbrs_kernel<<<batch, 32, 0, stream>>>(  // 将新邻居更新到 Graph 中
            from_ids.data().get(), res.first, graph.data().get(), graph_deg.data().get(),
            graph_dist.data().get(), cand_nbrs.data().get(), from_edge_nums.data().get(),
            total_from_vec_num, bp);
    }
}

Test_Result GPUFuncs::beam_search_for_insert(float *insert_vectors, int *d_knn_res, int batch,
                                             cudaStream_t &stream) {
    std::unique_lock<std::shared_mutex> lock(beam_mtx);
    Test_Result res = beam_search(insert_vectors, batch);
    for (int i = 0; i < batch; i++)
        thrust::transform(thrust::cuda::par.on(stream), beam + i * bp.beam_size,
                          beam + i * bp.beam_size + bp.k, d_knn_res + i * bp.k,
                          [] __host__ __device__(const BCD &bc) { return bc.id; });
    return res;
}

void GPUFuncs::vector_insert(float *insert_vectors, int batch) {
    const std::string name = "ins_del";
    const int base_n = base_n_upd.load();
    cudaStream_t &stream = streams[name].stream;

    event_record_time_start(name);

    thrust::device_vector<int> knns(batch * bp.k), paired_query_ids(batch),
        k_plus_ones(batch * (bp.k + 1));
    thrust::device_vector<float> d_insert_vectors(batch * bp.dim);
    CUDA_CHECK(cudaMemcpyAsync(d_insert_vectors.data().get(), insert_vectors,
                               batch * bp.dim * sizeof(float), cudaMemcpyHostToDevice, stream));

    // 将插入向量作为查询搜索 knns：
    beam_search_for_insert(insert_vectors, knns.data().get(), batch, stream);

    // 找到 knns 中每个 base 向量的配对查询向量中
    // 离插入向量最近的查询向量作为插入向量的配对查询向量：
    find_closest_paired_query_kernel<<<batch, 64, sizeof(int) + sizeof(float), stream>>>(
        knns.data().get(), base_query_ids, d_q, d_insert_vectors.data().get(),
        paired_query_ids.data().get(), batch, bp);

    k_plus_ones_build_kernel<<<batch, 64, 0, stream>>>(
        query_knns.get(), paired_query_ids.data().get(), k_plus_ones.data().get(), batch, base_n,
        bp);

    // 将新向量插入到 base，同时扩充图存储数据结构：
    d_b.resize(base_n + batch);
    base_insert_kernel<<<batch, 64, 0, stream>>>(d_insert_vectors.data().get(), d_b.data().get(),
                                                 batch, base_n, bp);
    graph_resize(batch);

    neibor_aware(k_plus_ones.data().get(), batch, bp.k + 1, base_n, stream,
                 from_edge_nums.data().get());

    event_record_time_stop(name, "vector_insert");
}

void GPUFuncs::update_prepare() {
    base_n_upd.store(bp.base_n);
    from_edge_nums.resize(bp.base_n);

    const std::string name = "ins_del";
    cudaStream_t &stream = streams[name].stream;

    std::shared_lock<std::shared_mutex> lock(graph_mtx);

    get_from_edge_num_kernel<<<1024, 32, 0, stream>>>(graph.data().get(), graph_deg.data().get(),
                                                      from_edge_nums.data().get(), bp.base_n, bp);
}

void GPUFuncs::update_free() {}

void GPUFuncs::graph_resize(int new_size) {
    base_n_upd.fetch_add(new_size);

    int base_n = base_n_upd.load();
    graph.resize(base_n * bp.max_degree);  // 扩展 graph
    graph_dist.resize(base_n * bp.max_degree);
    graph_deg.resize(base_n);
    from_edge_nums.resize(base_n);

    fo.print("graph resize to " + TOS(base_n) + " with new " + TOS(new_size) + " vectors");
}
};  // namespace efanna2e