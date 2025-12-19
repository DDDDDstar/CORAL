#include <cfloat>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>

#include "fileout.h"
#include "gpufuncs.cuh"
#include "utils.cuh"

namespace efanna2e {
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

__global__ void k_plus_ones_build_kernel(const int* __restrict__ insert_ids,
                                         const int* __restrict__ query_knns,
                                         const int* __restrict__ paired_query_ids,
                                         int* __restrict__ k_plus_ones_res, int batch, BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    extern __shared__ int* s_out[];
    const int** s_knns = s_out + 1;
    for (int ins_idx = bid; ins_idx < batch; ins_idx += gridDim.x) {
        if (tid == 0) {
            *s_out = k_plus_ones_res + ins_idx * (bp.k + 1);
            *s_knns = query_knns + paired_query_ids[ins_idx] * bp.k;
        }
        __syncthreads();

        for (int i = tid; i < bp.k + 1; i += tpb)
            (*s_out)[i] = i < bp.k ? (*s_knns)[i] : insert_ids[ins_idx];
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
    const int* __restrict__ del_ids,
    const NBD* __restrict__ del_node_data,        // 删除节点数据
    const NBD* __restrict__ del_node_innbr_data,  // 删除节点的入邻居数据
    const int* __restrict__ del_idxs,             // 每个入邻居对应的删除节点索引
    int innbrs_num, CN* __restrict__ cand_nbrs, int batch, BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x,
              cand_size = bp.max_degree * 2 - 1;
    extern __shared__ int sh_outnbr_ids[];
    float* sh_outnbr_vecs = (float*)(sh_outnbr_ids + bp.max_degree);
    int *sh_outnbr_num = (int*)(sh_outnbr_vecs + bp.dim * bp.max_degree),
        *sh_del_id = sh_outnbr_num + 1;

    for (int del_idx = bid; del_idx < batch; del_idx += gridDim.x) {
        if (tid == 0) {
            *sh_outnbr_num = del_node_data->degs[del_idx];
            *sh_del_id = del_ids[del_idx];
        }
        __syncthreads();

        for (int i = tid; i < del_node_data->degs[del_idx]; i += tpb) {
            const int outnbr_id = sh_outnbr_ids[i] =
                del_node_data->nbrs[del_idx * bp.max_degree + i];
            for (int del_i = 0; del_i < batch; del_i++) {  // 检查出邻居中是否有被删除节点
                if (outnbr_id == del_ids[del_i]) {
                    sh_outnbr_ids[i] = -1;
                    break;
                }
            }
            const float* nbr_vec =
                del_node_data->nbrs_vec + del_idx * bp.max_degree * bp.dim + i * bp.dim;
            for (int d = 0; d < bp.dim; d++) sh_outnbr_vecs[i * bp.dim + d] = nbr_vec[d];
        }

        __syncthreads();

        bool finished = false;
        for (int innbr_idx = tid; innbr_idx < innbrs_num; innbr_idx += tpb) {
            if (del_idxs[innbr_idx] != del_idx) {
                if (finished) break;
                continue;
            }
            finished = true;

            const int cand_idx = innbr_idx * cand_size,
                      old_nbr_num = del_node_innbr_data->degs[innbr_idx];
            int nbr_i = 0;
            for (int old_nbr_i = 0; old_nbr_i < old_nbr_num; old_nbr_i++) {
                CN* cand_nbr = cand_nbrs + cand_idx + nbr_i;
                const int id = del_node_innbr_data->nbrs[innbr_idx * bp.max_degree + old_nbr_i];
                if (id == *sh_del_id)  // 跳过被删除节点!
                    continue;

                cand_nbr->idx = nbr_i++;
                cand_nbr->id = id;
                cand_nbr->status = Initial;
                cand_nbr->dist = del_node_innbr_data->dists[innbr_idx * bp.max_degree + old_nbr_i];

                const float* nbr_vec = del_node_innbr_data->nbrs_vec +
                                       innbr_idx * bp.max_degree * bp.dim + old_nbr_i * bp.dim;
                for (int d = 0; d < bp.dim; d++) cand_nbr->vec[d] = nbr_vec[d];
            }
            for (int outnbr_i = 0; outnbr_i < *sh_outnbr_num; outnbr_i++) {
                const int id = sh_outnbr_ids[outnbr_i];
                if (id == -1) continue;

                CN* cand_nbr = cand_nbrs + cand_idx + nbr_i;
                cand_nbr->idx = nbr_i++;
                cand_nbr->id = id;
                cand_nbr->status = Initial;

                const float* nbr_vec = sh_outnbr_vecs + outnbr_i * bp.dim;
                cand_nbr->dist =
                    calc_distance(del_node_innbr_data->vecs + innbr_idx * bp.dim, nbr_vec, bp);
                for (int d = 0; d < bp.dim; d++) cand_nbr->vec[d] = nbr_vec[d];
            }
            for (; nbr_i < cand_size; nbr_i++) {
                cand_nbrs[cand_idx + nbr_i].idx = nbr_i;
                cand_nbrs[cand_idx + nbr_i].id = -1;
                cand_nbrs[cand_idx + nbr_i].status = Discarded;
                cand_nbrs[cand_idx + nbr_i].dist = farthest_dist_d(bp);
            }
        }
    }
}

__global__ void candidate_ignore_kernel(const float* __restrict__ base, CN* __restrict__ cand_nbrs,
                                        const int* __restrict__ sort_idxs,
                                        int* __restrict__ nbr_nums, int new_nbr_i, int pivot_num,
                                        BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x, cand_size = bp.max_degree * 2;
    for (int pivot_i = bid; pivot_i < pivot_num; pivot_i += gridDim.x) {
        CN* new_nbr = get_CN(cand_nbrs, sort_idxs, new_nbr_i, 0, pivot_i, 0, cand_size);
        // 无效节点或邻居已满，直接淘汰
        if (new_nbr->id == -1 || nbr_nums[pivot_i] >= bp.max_degree)
            new_nbr->status = Discarded;
        else if (new_nbr->status == Initial)  // 待定节点，保留且用于淘汰后续节点
        {
            new_nbr->status = Retained;
            nbr_nums[pivot_i]++;
            for (int delta = tid + 1; delta < cand_size - new_nbr_i; delta += tpb) {
                CN* wti_cand_nbr = get_CN(cand_nbrs, sort_idxs, new_nbr_i + delta, 0, pivot_i, 0,
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

__global__ void update_new_nbrs_kernel(const int* __restrict__ vec_ids,
                                       const int* __restrict__ sort_idxs, int* __restrict__ graph,
                                       int* __restrict__ graph_deg, float* __restrict__ graph_dist,
                                       CN* __restrict__ cand_nbrs,
                                       int* __restrict__ from_edge_nums, int pivot_num, BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x, cand_size = 2 * bp.max_degree;
    extern __shared__ unsigned char sh_mem[];
    int* sh_nbr_num = (int*)sh_mem;
    for (int i = bid; i < pivot_num; i += gridDim.x) {
        if (!tid) *sh_nbr_num = 0;
        __syncthreads();

        const int base_id = vec_ids[i], idx_start = base_id * bp.max_degree;

        for (int old_nbr_i = tid; old_nbr_i < graph_deg[base_id]; old_nbr_i += tpb)
            atomicSub(from_edge_nums + graph[idx_start + old_nbr_i], 1);

        for (int j = tid; j < cand_size && *sh_nbr_num < bp.max_degree; j += tpb) {
            const CN* cand_nbr = get_CN(cand_nbrs, sort_idxs, j, 0, i, 0, cand_size);
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
            const CN* cand_nbr = get_CN(cand_nbrs, sort_idxs, j, 0, i, 0, cand_size);
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

void GPUFuncs::vector_delete(std::vector<int>& del_ids) {
    const int del_num = del_ids.size();
    cudaStream_t stream = streams["upd"].stream;

    auto e = gpu_record_time_start(stream);

    std::vector<int> cpu_slot_idxs, gpu_slot_idxs, nbr_cpu_slot_idxs, nbr_gpu_slot_idxs,
        h_del_innbrs, h_del_idxs;
    std::unordered_set<int> del_set(del_ids.begin(), del_ids.end());
    graph->get_innbrs(del_ids, h_del_innbrs, h_del_idxs);
    h_del_innbrs.erase(std::remove_if(h_del_innbrs.begin(), h_del_innbrs.end(),
                                      [&](int id) { return del_set.count(id); }),
                       h_del_innbrs.end());  // 如果删除节点的入邻居中有删除节点，则移除

    const int innbrs_num = h_del_innbrs.size();
    NBD del_innbr_data(innbrs_num, bp.max_degree, bp.dim, stream), *d_del_innbr_data_buffer;
    CUDA_CHECK(cudaMallocAsync(&d_del_innbr_data_buffer, sizeof(NBD), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_del_innbr_data_buffer, &del_innbr_data, sizeof(NBD),
                               cudaMemcpyHostToDevice, stream));
    graph->load_data(h_del_innbrs, nbr_cpu_slot_idxs, nbr_gpu_slot_idxs, d_del_innbr_data_buffer,
                     innbrs_num, stream);

    graph->load_data(del_ids, cpu_slot_idxs, gpu_slot_idxs, d_del_node_data_buffer, del_num,
                     stream);
    graph->mark_used_graph_data(cpu_slot_idxs, gpu_slot_idxs);
    graph->delete_nodes(del_ids);

    thrust::device_vector<int> d_del_idxs(innbrs_num), d_del_ids(del_num);
    CUDA_CHECK(cudaMemcpyAsync(d_del_ids.data().get(), del_ids.data(), sizeof(int) * del_num,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_del_idxs.data().get(), h_del_idxs.data(), del_num * sizeof(int),
                               cudaMemcpyHostToDevice, stream));

    const int cand_size = 2 * bp.max_degree - 1, total_cand_size = innbrs_num * cand_size;
    thrust::device_vector<int> cand_idxs(total_cand_size), cand_idxs_sort(total_cand_size),
        offsets(innbrs_num + 1), nbr_nums(innbrs_num, 0);
    thrust::device_vector<float> cand_dists(total_cand_size), cand_dists_sort(total_cand_size);
    thrust::device_vector<CN> cand_nbrs(total_cand_size);

    vector_deletion_patch_kernel<<<del_num, 256, (bp.max_degree + 1) * sizeof(int), stream>>>(
        d_del_ids.data().get(), d_del_node_data_buffer, d_del_innbr_data_buffer,
        d_del_idxs.data().get(), innbrs_num, cand_nbrs.data().get(), del_num, bp);

    idx_dist_extract(cand_nbrs.data().get(), cand_idxs.data().get(), cand_dists.data().get(),
                     stream, total_cand_size, bp);
    auto res = segmented_sort_pairs(cand_idxs.data().get(), cand_dists.data().get(),
                                    cand_idxs_sort.data().get(), cand_dists_sort.data().get(),
                                    offsets.data().get(), stream, total_cand_size, cand_size);
    const int threads = 256, blocks = (innbrs_num + threads - 1) / threads;
    for (int i = 0; i < cand_size; ++i)
        candidate_ignore_kernel<<<blocks, threads, 0, stream>>>(cand_nbrs.data().get(), res.first,
                                                                nbr_nums.data().get(), i,
                                                                innbrs_num, cand_size, bp);

    update_new_nbrs_kernel<<<blocks, threads, 0, stream>>>(res.first, d_del_innbr_data_buffer,
                                                           cand_nbrs.data().get(), innbrs_num,
                                                           cand_size, bp);  // 更新新邻居

    graph->update_graph_data(nbr_gpu_slot_idxs, d_del_innbr_data_buffer, innbrs_num, stream);

    const float time = gpu_record_time_stop(e, stream);
}

void GPUFuncs::vector_insert(float* h_insert_vectors, int batch) {
    cudaStream_t stream = streams["upd"].stream;

    auto e = gpu_record_time_start(stream);

    graph->insert_nodes(h_insert_vectors, h_ins_ids, batch);

    CUDA_CHECK(cudaMemcpyAsync(d_ins_ids.data().get(), h_ins_ids.data(), batch * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_insert_vectors.data().get(), h_insert_vectors,
                               batch * bp.dim * sizeof(float), cudaMemcpyHostToDevice, stream));

    query_index->Top1_Search(h_insert_vectors, d_paired_query_ids.data().get(), batch, stream);

    // 将插入向量作为查询搜索 knns：
    // beam_search_for_insert(insert_vectors, knns.data().get(), batch, stream);

    // 找到 knns 中每个 base 向量的配对查询向量中
    // 离插入向量最近的查询向量作为插入向量的配对查询向量：
    // find_closest_paired_query_kernel<<<batch, 64, sizeof(int) + sizeof(float), stream>>>(
    //     knns.data().get(), base_query_ids, d_q, d_insert_vectors.data().get(),
    //     paired_query_ids.data().get(), batch, bp);

    k_plus_ones_build_kernel<<<batch, 64, 0, stream>>>(
        d_ins_ids.data().get(), query_index->get_knns().get(), d_paired_query_ids.data().get(),
        d_k_plus_ones.data().get(), batch, bp);
    CUDA_CHECK(cudaMemcpyAsync(h_k_plus_ones.data(), d_k_plus_ones.data().get(),
                               batch * (bp.k + 1) * sizeof(int), cudaMemcpyDeviceToHost, stream));

    // 将新向量插入到 base，同时扩充图存储数据结构：
    // base_insert_kernel<<<batch, 64, 0, stream>>>(d_insert_vectors.data().get(),
    // d_b.data().get(),
    //                                              batch, base_n, bp);
    // graph_resize(batch);

    std::vector<int> gpu_slot_idxs, cpu_slot_idxs;
    graph->load_data(h_k_plus_ones, cpu_slot_idxs, gpu_slot_idxs, d_k_plus_one_node_data,
                     batch * (bp.k + 1), stream);
    neibor_aware(d_k_plus_ones.data().get(), d_k_plus_one_node_data, batch, bp.k + 1, stream);
    graph->update_graph_data(gpu_slot_idxs, d_k_plus_one_node_data, batch * (bp.k + 1), stream);

    gpu_record_time_stop(e, stream);
}

void GPUFuncs::update_prepare() {
    const std::string name = "upd";
    cudaStream_t& stream = streams[name].stream;

    h_del_node_data = new NBD(BATCH, bp.max_degree, bp.dim, stream);
    CUDA_CHECK(cudaMallocAsync(&d_del_node_data_buffer, sizeof(NBD), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_del_node_data_buffer, h_del_node_data, sizeof(NBD),
                               cudaMemcpyHostToDevice, stream));

    d_ins_ids.resize(BATCH);
    d_paired_query_ids.resize(BATCH);
    d_k_plus_ones.resize(BATCH * (bp.k + 1));
    h_k_plus_ones.resize(BATCH * (bp.k + 1));
    d_insert_vectors.resize(BATCH * bp.dim);

    h_k_plus_one_node_data = new NBD(BATCH * (bp.k + 1), bp.max_degree, bp.dim, stream);
    CUDA_CHECK(cudaMallocAsync(&d_k_plus_one_node_data, sizeof(NBD), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_k_plus_one_node_data, h_k_plus_one_node_data, sizeof(NBD),
                               cudaMemcpyHostToDevice, stream));
}

void GPUFuncs::update_free() {
    const std::string name = "upd";
    cudaStream_t& stream = streams[name].stream;
    CUDA_CHECK(cudaFreeAsync(d_del_node_data_buffer, stream));
    delete h_del_node_data;
    delete h_k_plus_one_node_data;
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