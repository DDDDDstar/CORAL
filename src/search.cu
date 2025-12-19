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

__global__ void beam_init_kernel(const float* __restrict__ queries,
                                 const NBD* __restrict__ start_node_data,
                                 const int* __restrict__ start_node_ids, BCD* __restrict__ beams,
                                 int M, BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    extern __shared__ float s_query[];
    for (int d = tid; d < bp.dim; d++) s_query[d] = queries[bid * bp.dim + d];
    __syncthreads();

    BCD* beam = beams + bid * bp.beam_size;
    for (int i = tid; i < bp.beam_size; i += tpb) {
        if (i < M) {
            beam[i].id = start_node_ids[i];
            beam[i].dist = calc_distance(s_query, start_node_data->vecs, i, bp);
            beam[i].expanded = false;  // 初始化为未扩展
        } else {
            beam[i].id = -1;
            beam[i].dist = farthest_dist_d(bp);
            beam[i].expanded = false;  // 初始化为未扩展
        }
    }
}

// __global__ void beam_expand_kernel(const float* __restrict__ query,
//                                    const NBD* __restrict__ beam_node_data,
//                                    int* __restrict__ visited, BCD* __restrict__ beam,
//                                    int* expand_num, int* hops, BP bp) {
//     const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
//     extern __shared__ float sh_query[];
//     int *sh_expand = (int*)(sh_query + bp.dim), *sh_new_node_num = sh_expand + 1;
//     for (int beam_i = bid; beam_i < bp.beam_capacity; beam_i += gridDim.x) {
//         if (tid == 0) {
//             if (beam[beam_i].id == -1 || beam[beam_i].expanded ||
//                 atomicAdd(expand_num, 1) >= bp.beam_expand_num)
//                 *sh_expand = 0;
//             else {
//                 *sh_expand = 1;
//                 *sh_new_node_num = 0;
//                 beam[beam_i].expanded = true;
//             }
//         }
//         __syncthreads();

//         if (*sh_expand == 1) {
//             for (int j = tid; j < bp.dim; j += tpb) sh_query[j] = query[j];
//             __syncthreads();

//             const float* nbr_vecs = beam_node_data->nbrs_vec + beam_i * bp.max_degree * bp.dim;
//             BCD* beam_new = beam + bp.beam_capacity + beam_i * bp.max_degree;
//             const float farthest = farthest_dist_d(bp);
//             for (int nbr_i = tid; nbr_i < bp.max_degree; nbr_i += tpb) {
//                 if (nbr_i >= beam_node_data->degs[beam_i]) {
//                     beam_new[nbr_i].id = -1;
//                     beam_new[nbr_i].dist = farthest;
//                     continue;
//                 }
//                 const int nbr_id = beam_node_data->nbrs[beam_i * bp.max_degree + nbr_i];
//                 // assert(nbr_id >= 0);
//                 // assert(nbr_id < bp.base_n);
//                 if (atomicTestAndSetBit(visited, nbr_id) == 1) {  //
//                 原子比较更新，跳过已访问的节点
//                     beam_new[nbr_i].id = -1;
//                     beam_new[nbr_i].dist = farthest;
//                     continue;
//                 }
//                 beam_new[nbr_i].id = nbr_id;
//                 beam_new[nbr_i].dist = calc_distance(sh_query, nbr_vecs, nbr_i, bp);
//                 beam_new[nbr_i].expanded = false;  // 初始化为未扩展
//                 atomicAdd(sh_new_node_num, 1);
//             }
//             __syncthreads();

//             if (tid == 0)
//                 if (*sh_new_node_num == 0) {  // 如果所有节点都拓展过，被拓展的 beam node 数量
//                 -1
//                     atomicSub(expand_num, 1);
//                 } else
//                     atomicAdd(hops, *sh_new_node_num);
//         }
//         __syncthreads();
//     }
// }

__global__ void beam_expand_kernel(const float* __restrict__ queries,
                                   const NBD* __restrict__ beam_node_data, BCD* __restrict__ beams,
                                   int* expand_nums, int* hops, int batch, BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    extern __shared__ float sh_query[];
    int* s_expand_num = (int*)(sh_query + bp.dim);
    for (int batch_i = bid; batch_i < batch; batch_i += gridDim.x) {
        if (expand_nums[batch_i] < 0) return;

        for (int d = tid; d < bp.dim; d += tpb) sh_query[d] = queries[batch_i * bp.dim + d];
        if (tid == 0) *s_expand_num = 0;
        __syncthreads();

        BCD* beam = beams + batch_i * bp.beam_size;
        const int offset = batch_i * bp.beam_capacity, *degs = beam_node_data->degs + offset;
        const float *beam_nbr_vecs = beam_node_data->nbrs_vec + offset * bp.max_degree * bp.dim,
                    farthest = farthest_dist_d(bp);
        for (int beam_i = tid; beam_i < bp.beam_capacity - 1;
             beam_i += tpb)  // 防止相同节点被重复拓展
            if (beam[beam_i].id == beam[beam_i + 1].id) {
                beam[beam_i + 1].id = -1;
                beam[beam_i].dist = farthest;
            }
        __syncthreads();

        for (int beam_i = tid; beam_i < bp.beam_capacity; beam_i += tpb) {
            const int offset = beam_i * bp.max_degree;
            BCD* beam_new = beam + bp.beam_capacity + offset;
            if (beam[beam_i].id == -1 || beam[beam_i].expanded) {
                for (int i = 0; i < bp.max_degree; i++) {
                    beam_new[i].id = -1;
                    beam_new[i].dist = farthest;
                }
                continue;
            }

            beam[beam_i].expanded = true;

            int new_node_num = 0;
            const int* nbrs = beam_node_data->nbrs + offset;
            const float *nbr_vecs = beam_nbr_vecs + offset * bp.dim, dist = beam[beam_i].dist;
            int nbr_i = 0;
            for (; nbr_i < degs[beam_i]; nbr_i++) {
                const float new_dist = calc_distance(sh_query, nbr_vecs, nbr_i, bp);
                // 如果新节点到查询点的距离比被拓展节点距离还远的话，就不用拓展：
                if (compare_dist(new_dist, dist, bp)) {
                    beam_new[nbr_i].id = nbrs[nbr_i];
                    beam_new[nbr_i].dist = new_dist;
                    beam_new[nbr_i].expanded = false;
                    new_node_num++;
                } else {
                    beam_new[nbr_i].id = -1;
                    beam_new[nbr_i].dist = farthest;
                }
            }
            for (; nbr_i < bp.max_degree; nbr_i++) {
                beam_new[nbr_i].id = -1;
                beam_new[nbr_i].dist = farthest;
            }
            if (new_node_num > 0) {
                atomicAdd(hops, new_node_num);
                atomicAdd(s_expand_num, 1);
            }
        }
        __syncthreads();

        if (tid == 0) expand_nums[batch_i] = *s_expand_num;
        __syncthreads();
    }
}

// void id_dist_extract_from_BCD(const BCD* beam, int* ids_out, float* dists_out,
//                               cudaStream_t& stream, int total_num, BP bp) {
//     thrust::device_ptr<const BCD> beam_ptr(beam);
//     thrust::device_ptr<int> ids_ptr(ids_out);
//     thrust::device_ptr<float> dists_ptr(dists_out);
//     auto out_begin = thrust::make_zip_iterator(thrust::make_tuple(ids_ptr, dists_ptr));

//     // 提取 BCD 结构体数组 beam 中的 dist 字段，对于 IP
//     // 距离，取负实现从大到小排序
//     if (bp.metric == DIST_METRIC::L2)
//         thrust::transform(
//             thrust::cuda::par.on(stream), beam_ptr, beam_ptr + total_num, out_begin,
//             [] __host__ __device__(const BCD& bc) { return thrust::make_tuple(bc.id, bc.dist);
//             });
//     else
//         thrust::transform(
//             thrust::cuda::par.on(stream), beam_ptr, beam_ptr + total_num, out_begin,
//             [] __host__ __device__(const BCD& bc) { return thrust::make_tuple(bc.id, -bc.dist);
//             });
// }

void dist_extract_from_BCD(const BCD* beam, float* dists_out, cudaStream_t& stream, int total_num,
                           BP bp) {
    thrust::device_ptr<const BCD> beam_ptr(beam);
    thrust::device_ptr<float> dists_ptr(dists_out);

    // 提取 BCD 结构体数组 beam 中的 dist 字段，对于 IP
    // 距离，取负实现从大到小排序
    if (bp.metric == DIST_METRIC::L2)
        thrust::transform(thrust::cuda::par.on(stream), beam_ptr, beam_ptr + total_num, dists_ptr,
                          [] __host__ __device__(const BCD& bc) { return bc.dist; });
    else
        thrust::transform(thrust::cuda::par.on(stream), beam_ptr, beam_ptr + total_num, dists_ptr,
                          [] __host__ __device__(const BCD& bc) { return -bc.dist; });
}

void id_extract_from_BCD(const BCD* beam, int* ids_out, cudaStream_t& stream, int batch_n,
                         int batch, int extract_n) {
    // thrust::device_ptr<const BCD> beam_ptr(beam);
    // thrust::device_ptr<int> ids_ptr(ids_out);
    // for (int i = 0; i < batch; i++)
    //     thrust::transform(thrust::cuda::par.on(stream), beam_ptr + i * batch_n,
    //                       beam_ptr + i * batch_n + extract_n, ids_ptr + i * extract_n,
    //                       [] __host__ __device__(const BCD& bc) { return bc.id; });
    thrust::counting_iterator<int> idx(0);
    thrust::transform(thrust::cuda::par.on(stream), idx, idx + batch * extract_n,
                      thrust::device_pointer_cast(ids_out),
                      [beam, batch_n, extract_n] __device__(int t) {
                          return beam[(t / extract_n) * batch_n + (t % extract_n)].id;
                      });
}

__global__ void beam_test_knn_kernel(const BCD* __restrict__ beams,
                                     const int* __restrict__ gt_knns, int* __restrict__ hits_out,
                                     BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    extern __shared__ int s_hit[];
    int* s_knns = s_hit + 1;

    for (int i = tid; i < bp.k; i++) s_knns[i] = gt_knns[bid * bp.k + i];
    if (tid == 0) *s_hit = 0;
    __syncthreads();

    const BCD* beam = beams + bid * bp.beam_size;
    for (int beam_i = tid; beam_i < bp.k; beam_i += tpb) {
        const int id = beam[beam_i].id;
        for (int gt_i = 0; gt_i < bp.k; gt_i++)
            if (id == s_knns[gt_i]) {
                atomicAdd(s_hit, 1);
                break;
            }
    }
    __syncthreads();

    if (tid == 0) atomicAdd(hits_out, *s_hit);
}

int GPUFuncs::beam_search(const std::vector<int>& start_node_ids, const float* h_queries,
                          int batch, cudaStream_t& stream) {
    const int M = start_node_ids.size();
    std::string ids, degs;
    for (int id : start_node_ids) ids += TOS(id) + " ";
    // fo.print("beam_search start ids(" + TOS(M) + "): " + ids);
    int step = 0, total_hops;
    std::vector<int> gpu_slot_idxs, cpu_slot_idxs, expand_nums(batch, 0), hops(batch, 0);
    thrust::device_vector<int> d_start_node_ids(M), d_hops(1, 0), d_expand_nums(batch);

    CUDA_CHECK(cudaMemcpyAsync(d_queries.data().get(), h_queries, BATCH * bp.dim * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_start_node_ids.data().get(), start_node_ids.data(),
                               M * sizeof(int), cudaMemcpyHostToDevice, stream));

    graph->load_data(start_node_ids, cpu_slot_idxs, gpu_slot_idxs, d_beam_node_data, M, stream);
    for (int i = 0; i < M; i++) {
        const HostSlot* slot = graph->get_cpu_slot(cpu_slot_idxs[i]);
        degs += TOS(slot->deg) + " ";
    }
    fo.print("beam_search start degs(" + TOS(M) + "): " + degs);
    beam_node_data->self_copy(batch, bp.beam_capacity, M, bp.max_degree, bp.dim, stream);
    graph->mark_used_graph_data(cpu_slot_idxs, gpu_slot_idxs);

    // 初始化 beam，将起始点放入 beam
    beam_init_kernel<<<batch, 256, bp.dim * sizeof(float), stream>>>(
        d_queries.data().get(), d_beam_node_data, d_start_node_ids.data().get(),
        d_beams.data().get(), M, bp);

#ifdef GPU_TEST
    cudaStreamSynchronize(stream);
    std::vector<BCD> h_beam(batch * bp.beam_size);
    int ep;
    std::string res;
    CUDA_CHECK(cudaMemcpy(h_beam.data(), beam, batch * bp.beam_size * sizeof(BCD),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&ep, graph_ep, sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < bp.k; j++) {
            const int id = h_beam[i * bp.beam_size + j].id;
            res += TOS(id) + "/" + TOS(h_beam[i * bp.beam_size + j].dist) + " ";
        }
        res += "\n";
    }
    // fo.iprint(TOS(ep) + "\n" + res);
#endif

    int not_converge_num = batch;
    while (true) {
        CUDA_CHECK(cudaMemcpyAsync(d_expand_nums.data().get(), expand_nums.data(),
                                   batch * sizeof(int), cudaMemcpyHostToDevice, stream));
        beam_expand_kernel<<<not_converge_num, 256, bp.dim * sizeof(float), stream>>>(
            d_queries.data().get(), d_beam_node_data, d_beams.data().get(),
            d_expand_nums.data().get(), d_hops.data().get(), batch, bp);

        CUDA_CHECK(cudaMemcpyAsync(expand_nums.data(), d_expand_nums.data().get(),
                                   batch * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(&total_hops, d_hops.data().get(), sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        not_converge_num = 0;
        for (int i = 0; i < batch; i++) {
            if (expand_nums[i] > 0) {
                not_converge_num++;
                expand_nums[i] = 0;
            } else
                expand_nums[i] = -1;
        }
        fo.print("beam search: step-" + TOS(++step) + ", not_converge_num-" +
                 TOS(not_converge_num) + ", total_hops-" + TOS(total_hops));
        if (!not_converge_num) break;

        dist_extract_from_BCD(d_beams.data().get(), d_beam_dists.data().get(), stream,
                              batch * bp.beam_size, bp);

        segmented_sort_pairs(d_beams.data().get(), d_beam_dists.data().get(),
                             d_beams_sort.data().get(), d_beam_dists_sort.data().get(),
                             d_beam_offsets.data().get(), stream, batch, bp.beam_size);

        id_extract_from_BCD(d_beams.data().get(), d_beam_ids.data().get(), stream, bp.beam_size,
                            batch, bp.beam_capacity);

        CUDA_CHECK(cudaMemcpyAsync(beam_ids.data(), d_beam_ids.data().get(),
                                   sizeof(int) * bp.beam_capacity * batch, cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        graph->load_data(beam_ids, cpu_slot_idxs, gpu_slot_idxs, d_beam_node_data,
                         batch * bp.beam_capacity, stream);
        graph->mark_used_graph_data(cpu_slot_idxs, gpu_slot_idxs);
    }
#ifdef GPU_TEST
    CUDA_CHECK(cudaMemcpy(h_beam.data(), beam, batch * bp.beam_size * sizeof(BCD),
                          cudaMemcpyDeviceToHost));
    res = "";
    bool flag = false;
    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < bp.k; j++) {
            const int id = h_beam[i * bp.beam_size + j].id;
            res += TOS(id) + "/" + TOS(h_beam[i * bp.beam_size + j].dist) + " ";
            if (id < 0) flag = true;
        }
        res += "\n";
    }
    // fo.iprint(res);
    assert(!flag);
#endif
    return total_hops;
}

int GPUFuncs::beam_search_verify(const int* h_gt_knns, int batch, cudaStream_t& stream) {
    int hits;
    thrust::device_vector<int> d_gt_knns(bp.k * batch), d_hits(1, 0);

    CUDA_CHECK(cudaMemcpyAsync(d_gt_knns.data().get(), h_gt_knns, batch * bp.k * sizeof(int),
                               cudaMemcpyHostToDevice, stream));

    beam_test_knn_kernel<<<batch, 64, sizeof(int) * (bp.k + 1), stream>>>(
        d_beams.data().get(), d_gt_knns.data().get(), d_hits.data().get(), bp);

    CUDA_CHECK(
        cudaMemcpyAsync(&hits, d_hits.data().get(), sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    return hits;
}

// Test_Result GPUFuncs::beam_search_for_insert(float* insert_vectors, int* d_knn_res, int
// batch,
//                                              const std::string& stream_name = "upd") {
//     cudaStream_t& stream = streams[stream_name].stream;
//     std::unique_lock<std::shared_mutex> lock(search_mtx);
//     Test_Result tr;
//     std::vector<int> start_node_ids;
//     const int M = graph->get_top_M_access_nodes(start_node_ids);
//     start_node_ids.resize(M);
//     for (int i = 0; i < batch; i++) {
//         tr.accumulate(beam_search(start_node_ids, insert_vectors + i * bp.dim,
//         stream_name)); thrust::transform(thrust::cuda::par.on(stream), beam, beam + bp.k,
//         d_knn_res + i * bp.k,
//                           [] __host__ __device__(const BCD& bc) { return bc.id; });
//     }
//     return tr;
// }

Test_Result GPUFuncs::test_search_and_verify(int test_query_idx, int batch,
                                             const std::vector<int>& start_node_ids) {
    cudaStream_t stream = streams["search"].stream;
    auto e = gpu_record_time_start(stream);

    const int hops = beam_search(start_node_ids, test_query_data + test_query_idx * bp.dim, batch,
                                 stream),
              hits = beam_search_verify(test_query_knns + test_query_idx * bp.k, batch, stream);

    const float time = gpu_record_time_stop(e, stream);

    return Test_Result(hits, hops, time);
}

void GPUFuncs::test_query_data_prepare(const float* data, const int* knns) {
    test_query_data = data;
    test_query_knns = knns;
}

void GPUFuncs::search_prepare() {  // 图结构数据内存分配：
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // CUDA_CHECK(cudaMallocAsync(&d_start_node_ids, sizeof(int) * SEARCH_START_NODES_NUM,
    // stream));

    // h_beam_node_data = new NBD(bp.beam_capacity, bp.max_degree, bp.dim);
    // CUDA_CHECK(cudaMallocAsync(&d_beam_node_data, sizeof(NBD), stream));
    // CUDA_CHECK(cudaMemcpyAsync(d_beam_node_data, h_beam_node_data, sizeof(NBD),
    //                            cudaMemcpyHostToDevice, stream));

    // 每一个拓展节点需要的用于存储邻居的内存大小:
    bp.beam_expand_num =
        (GB(BEAM_MAX_MEM_SIZE) / 2 / (sizeof(BCD) + sizeof(float)) - bp.beam_capacity) /
        bp.max_degree / BATCH;  // 内存限制下最大可拓展节点数量
    bp.beam_expand_num = std::min(bp.beam_capacity, bp.beam_expand_num);
    bp.beam_size = bp.beam_capacity + bp.beam_expand_num * bp.max_degree;
    d_beams.resize(BATCH * bp.beam_size);
    d_beam_dists.resize(BATCH * bp.beam_size);
    d_beams_sort.resize(BATCH * bp.beam_size);
    d_beam_dists_sort.resize(BATCH * bp.beam_size);
    d_beam_offsets.resize(BATCH + 1);
    d_beam_ids.resize(BATCH * bp.beam_capacity);
    beam_ids.resize(BATCH * bp.beam_capacity);
    d_queries.resize(BATCH * bp.dim);
    beam_node_data = new NBD(BATCH * bp.beam_capacity, bp.max_degree, bp.dim, stream);
    CUDA_CHECK(cudaMallocAsync(&d_beam_node_data, sizeof(NBD), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_beam_node_data, beam_node_data, sizeof(NBD),
                               cudaMemcpyHostToDevice, stream));
    // beam_visited 用位图表示，每个 int 数值（32 位）可表示 32 个状态
    // CUDA_CHECK(cudaMallocAsync(&beam_visited, bp.base_n / 32 * sizeof(int), stream));
    // CUDA_CHECK(cudaMallocAsync(&beam_offsets, (BATCH + 1) * sizeof(int), stream));
    // CUDA_CHECK(cudaMallocAsync(&d_expand_num, sizeof(int), stream));
    // CUDA_CHECK(cudaMallocHost(&h_expand_num, sizeof(int)));
    // CUDA_CHECK(cudaMallocAsync(&d_hops, sizeof(int), stream));
    // CUDA_CHECK(cudaMallocHost(&h_hops, sizeof(int)));
    // CUDA_CHECK(cudaMallocAsync(&d_hits, sizeof(int), stream));
    // CUDA_CHECK(cudaMallocHost(&h_hits, sizeof(int)));
    // CUDA_CHECK(cudaMallocAsync(&beam_converge_num, sizeof(int), stream));
    // CUDA_CHECK(cudaMallocAsync(&d_query, bp.dim * sizeof(float), stream));
    // CUDA_CHECK(cudaMallocAsync(&d_gt_knns, bp.k * sizeof(int), stream));
    // h_beam_ids.resize(bp.beam_capacity);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

void GPUFuncs::search_free() {
    cudaStream_t& stream = streams["search"].stream;
    // CUDA_CHECK(cudaFreeAsync(d_start_node_ids, stream));
    // CUDA_CHECK(cudaFreeAsync(beam, stream));
    // CUDA_CHECK(cudaFreeAsync(beam_dists, stream));
    // CUDA_CHECK(cudaFreeAsync(beam_ids, stream));
    // CUDA_CHECK(cudaFreeAsync(beam_visited, stream));
    // CUDA_CHECK(cudaFreeAsync(d_expand_num, stream));
    // CUDA_CHECK(cudaFreeAsync(d_hops, stream));
    // CUDA_CHECK(cudaFreeAsync(d_hits, stream));
    // CUDA_CHECK(cudaFreeAsync(d_query, stream));
    // CUDA_CHECK(cudaFreeAsync(d_gt_knns, stream));

    // CUDA_CHECK(cudaFreeHost(h_hits));
    // CUDA_CHECK(cudaFreeHost(h_hops));
    // CUDA_CHECK(cudaFreeHost(h_expand_num));

    delete beam_node_data;
    CUDA_CHECK(cudaFreeAsync(d_beam_node_data, stream));
}
}  // namespace efanna2e

// 计算每个向量到质心距离 kernel
// __global__ void compute_centroid_l2_distance_kernel(const float* data, const float* center,
//                                                     float* distances, BP bp) {
//     for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < bp.base_n;
//          id += gridDim.x * blockDim.x) {
//         float sum = 0.0f;
//         for (int d = 0; d < bp.dim; ++d) {
//             float diff = center[d] - data[id * bp.dim + d];
//             sum += diff * diff;
//         }
//         distances[id] = sum;
//     }
// }

// __global__ void compute_centroid_ip_distance_kernel(const float* data, const float* center,
//                                                     float* distances, BP bp) {
//     for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < bp.base_n;
//          id += gridDim.x * blockDim.x) {
//         float sum = 0.0f;
//         for (int d = 0; d < bp.dim; ++d) sum += center[d] * data[id * bp.dim + d];
//         distances[id] = sum;
//     }
// }

// __global__ void find_closest_kernel(  // 找最近非孤立节点 kernel
//     const float* __restrict__ distances, const int* __restrict__ graph_deg,
//     int* __restrict__ closest_id_res, float* closest_val, float avg_degree, BP bp) {
//     for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < bp.base_n;
//          id += gridDim.x * blockDim.x) {
//         if (graph_deg[id] < avg_degree) continue;  // 忽略较孤立的节点

//         float dist = distances[id];
//         float old = atomicClosestFloat(closest_val, dist, bp);  // 原子比较更新最近距离和索引
//         __threadfence();  // 刷写全局/共享内存，使其他 block 可见

//         if (*closest_val - dist < EPS) *closest_id_res = id;
//     }
// }

// __global__ void find_closest_kernel(  // 找最近非孤立节点 kernel
//     const float* data, const float* center, const int* __restrict__ graph_deg,
//     int* __restrict__ closest_id_res, float* closest_val, float avg_degree, BP bp) {
//     for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < bp.base_n;
//          id += gridDim.x * blockDim.x) {
//         if (graph_deg[id] < avg_degree) continue;  // 忽略较孤立的节点

//         float dist = calc_distance(center, data, id, bp);
//         float old = atomicClosestFloat(closest_val, dist, bp);  // 原子比较更新最近距离和索引
//         __threadfence();  // 刷写全局/共享内存，使其他 block 可见

//         if (*closest_val - dist < EPS) *closest_id_res = id;
//     }
// }
