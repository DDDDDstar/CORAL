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
// 计算每个向量到质心距离 kernel
__global__ void compute_centroid_l2_distance_kernel(const float *data, const float *center,
                                                    float *distances, BP bp) {
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < bp.base_n;
         id += gridDim.x * blockDim.x) {
        float sum = 0.0f;
        for (int d = 0; d < bp.dim; ++d) {
            float diff = center[d] - data[id * bp.dim + d];
            sum += diff * diff;
        }
        distances[id] = sum;
    }
}

__global__ void compute_centroid_ip_distance_kernel(const float *data, const float *center,
                                                    float *distances, BP bp) {
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < bp.base_n;
         id += gridDim.x * blockDim.x) {
        float sum = 0.0f;
        for (int d = 0; d < bp.dim; ++d) sum += center[d] * data[id * bp.dim + d];
        distances[id] = sum;
    }
}

__global__ void find_closest_kernel(  // 找最近非孤立节点 kernel
    const float *__restrict__ distances, const int *__restrict__ graph_deg,
    int *__restrict__ closest_id_res, float *closest_val, float avg_degree, BP bp) {
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < bp.base_n;
         id += gridDim.x * blockDim.x) {
        if (graph_deg[id] < avg_degree) continue;  // 忽略较孤立的节点

        float dist = distances[id];
        float old = atomicClosestFloat(closest_val, dist, bp);  // 原子比较更新最近距离和索引
        __threadfence();  // 刷写全局/共享内存，使其他 block 可见

        if (*closest_val - dist < EPS) *closest_id_res = id;
    }
}

__global__ void find_closest_kernel(  // 找最近非孤立节点 kernel
    const float *data, const float *center, const int *__restrict__ graph_deg,
    int *__restrict__ closest_id_res, float *closest_val, float avg_degree, BP bp) {
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < bp.base_n;
         id += gridDim.x * blockDim.x) {
        if (graph_deg[id] < avg_degree) continue;  // 忽略较孤立的节点

        float dist = calc_distance(center, data, id, bp);
        float old = atomicClosestFloat(closest_val, dist, bp);  // 原子比较更新最近距离和索引
        __threadfence();  // 刷写全局/共享内存，使其他 block 可见

        if (*closest_val - dist < EPS) *closest_id_res = id;
    }
}

__global__ void beam_init_kernel(const float *__restrict__ queries, const float *__restrict__ base,
                                 BCD *__restrict__ beam, const int *start_base_id, BP bp) {
    const int qid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    BCD *beam_start = beam + qid * bp.beam_size;
    const float *query = queries + qid * bp.dim;

    for (int i = tid; i < bp.beam_size; i += tpb) {
        if (i == 0) {
            beam_start[0].id = *start_base_id;
            beam_start[0].dist = calc_distance(query, base, *start_base_id, bp);
            beam_start[0].expanded = false;  // 初始化为未扩展
        } else {
            beam_start[i].id = -1;
            beam_start[i].dist = farthest_dist_d(bp);
            beam_start[i].expanded = false;  // 初始化为未扩展
        }
    }
}

__global__ void beam_expand_kernel(const float *__restrict__ base,
                                   const float *__restrict__ queries,
                                   const int *__restrict__ graph,
                                   const int *__restrict__ graph_deg, int *__restrict__ visited,
                                   BCD *__restrict__ beam, bool *converge, int *converge_num,
                                   int *hops, int batch, BP bp) {
    const int qid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;

    extern __shared__ unsigned char smem[];
    float *sh_query = (float *)smem;
    int *sh_expand_num = (int *)(sh_query + bp.dim);

    for (int i = qid; i < batch; i += gridDim.x) {
        if (converge[i]) continue;  // 如果已经收敛，跳过继续拓展下一个 beam

        for (int j = tid; j < bp.dim; j += tpb)  // 将 beam 对应的查询加载到共享内存
            sh_query[j] = queries[i * bp.dim + j];

        if (tid == 0) *sh_expand_num = 0;
        __syncthreads();

        BCD *beam_start = beam + i * bp.beam_size;
        for (int j = tid; j < bp.beam_capacity; j += tpb) {  // 每个线程拓展 beam 中的一个节点
            if (beam_start[j].id == -1 || beam_start[j].expanded ||
                atomicAdd(sh_expand_num, 1) >= bp.beam_expand_num)
                continue;  // 如果节点为空或已经拓展或已达到拓展数量，跳过
            beam_start[j].expanded = true;
            // 拓展的新候选节点起始位:
            BCD *beam_new = beam_start + bp.beam_capacity + j * bp.max_degree;
            int new_nbr_i = 0;
#ifdef GPU_TEST
            assert(beam_start[j].id < bp.base_n);
#endif
            for (int nbr_i = 0; nbr_i < graph_deg[beam_start[j].id]; nbr_i++) {
                const int id = graph[beam_start[j].id * bp.max_degree + nbr_i];
#ifdef GPU_TEST
                assert(id < bp.base_n);
#endif
                if (atomicTestAndSetBit(visited + id * bp.visited_words_num, i) == 1)
                    continue;  // 原子比较更新，跳过已访问的节点
                beam_new[new_nbr_i].id = id;
                beam_new[new_nbr_i].dist = calc_distance(sh_query, base, id, bp);
                beam_new[new_nbr_i++].expanded = false;  // 初始化为未扩展
            }
            for (int invalid_i = new_nbr_i; invalid_i < bp.max_degree; invalid_i++) {
                beam_new[invalid_i].id = -1;  // 多余的位置重置为无效节点
                beam_new[invalid_i].dist = farthest_dist_d(bp);
            }
        }
        __syncthreads();

        if (tid == 0) {
            if (*sh_expand_num == 0) {  // 如果所有节点都拓展过，标记为收敛
                converge[i] = true;
                atomicAdd(converge_num, 1);
            } else
                atomicAdd(hops, *sh_expand_num);
        }
    }
}

void dist_extract_from_BCD(const BCD *beam, float *dists, cudaStream_t &stream, int total_num,
                           BP bp) {
    thrust::device_ptr<const BCD> beam_ptr(beam);
    thrust::device_ptr<float> dists_ptr(dists);

    // 提取 BCD 结构体数组 beam 中的 dist 字段，对于 IP
    // 距离，取负实现从大到小排序
    if (bp.metric == DIST_METRIC::L2)
        thrust::transform(thrust::cuda::par.on(stream), beam_ptr, beam_ptr + total_num, dists_ptr,
                          [] __host__ __device__(const BCD &bc) { return bc.dist; });
    else
        thrust::transform(thrust::cuda::par.on(stream), beam_ptr, beam_ptr + total_num, dists_ptr,
                          [] __host__ __device__(const BCD &bc) { return -bc.dist; });
}

__global__ void beam_test_knn_kernel(const BCD *__restrict__ beam, const int *__restrict__ knns,
                                     int *__restrict__ hits_res, int batch, BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;

    extern __shared__ unsigned char smem[];
    int *sh_knns = reinterpret_cast<int *>(smem), *sh_hits = sh_knns + bp.k;

    for (int i = bid; i < batch; i += gridDim.x) {
        const int *knns_start = knns + i * bp.k;
        for (int j = tid; j < bp.k; j += tpb) sh_knns[j] = knns_start[j];
        if (tid == 0) *sh_hits = 0;
        __syncthreads();

        const BCD *beam_start = beam + i * bp.beam_size;
        for (int j = tid; j < bp.k; j += tpb) {
            const int id = beam_start[j].id;
#ifdef GPU_TEST
            assert(id >= 0);
#endif
            for (int l = 0; l < bp.k; l++)
                if (id == sh_knns[l]) {
                    atomicAdd(sh_hits, 1);
                    break;
                }
        }
        __syncthreads();
        if (tid == 0) atomicAdd(hits_res, *sh_hits);
    }
}

Test_Result GPUFuncs::beam_search(float *queries, int batch, bool test) {
    const std::string stream_name = "search";
    cudaStream_t &stream = streams[stream_name].stream;
    float init_max = farthest_dist(bp), *d_queries, total_time = 0.0f;
    int converge_num = 0, step = 0, *d_hops, h_hops, total_hops = 0;

    if (test)
        d_queries = queries;  // 测试模式下，直接使用 test_query_data
    else {
        CUDA_CHECK(cudaMallocAsync(&d_queries, batch * bp.dim * sizeof(float), stream));
        CUDA_CHECK(cudaMemcpyAsync(d_queries, queries, batch * bp.dim * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
    }

    CUDA_CHECK(cudaMallocAsync(&d_hops, sizeof(int), stream));
    CUDA_CHECK(
        cudaMemcpyAsync(closest_val, &init_max, sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(beam_converge_num, 0, sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(beam_converge, 0, batch * sizeof(bool), stream));
    CUDA_CHECK(
        cudaMemsetAsync(beam_visited, 0, bp.visited_words_num * bp.base_n * sizeof(int), stream));

    std::shared_lock<std::shared_mutex> lock(beam_mtx);

    find_closest_kernel<<<1024, upd_threads, 0, stream>>>(
        d_dists, graph_deg.data().get(), graph_ep, closest_val, gs.total_degree / bp.base_n, bp);

    // 初始化 beam，将起始点放入 beam
    beam_init_kernel<<<batch, 128, 0, stream>>>(d_queries, d_b.data().get(), beam, graph_ep, bp);

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

    while (converge_num < batch) {
        event_record_time_start(stream_name);

        CUDA_CHECK(cudaMemsetAsync(d_hops, 0, sizeof(int), stream));
        beam_expand_kernel<<<batch - converge_num, 128, bp.dim * sizeof(float) + sizeof(int),
                             stream>>>(d_b.data().get(), d_queries, graph.data().get(),
                                       graph_deg.data().get(), beam_visited, beam, beam_converge,
                                       beam_converge_num, d_hops, batch, bp);

        dist_extract_from_BCD(beam, beam_dists, stream, batch * bp.beam_size, bp);

        segmented_sort_pairs_inplace(beam, beam_dists, beam_offsets, stream, batch, bp.beam_size);

        CUDA_CHECK(cudaMemcpyAsync(&converge_num, beam_converge_num, sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(&h_hops, d_hops, sizeof(int), cudaMemcpyDeviceToHost, stream));
        total_hops += h_hops;

        total_time += event_record_time_stop(stream_name, "beam_search");
        // fo.print("Beam Search step " + TOS(++step) + ": converge_num(" + TOS(converge_num) +
        //          "/" + TOS(batch) + "), total_time: " + TOS(total_time / 1000) +
        //          "s, avg_hops: " + TOS((float)total_hops / batch));
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

    if (!test) {
        CUDA_CHECK(cudaFreeAsync(d_queries, stream));
        CUDA_CHECK(cudaFreeAsync(d_hops, stream));
    }

    return Test_Result(total_hops, total_time / 1000);
}

void GPUFuncs::beam_search_verify(int *knns, Test_Result &res, int batch, bool test) {
    const std::string stream_name = "search";
    cudaStream_t &stream = streams[stream_name].stream;
    int *d_knns, *d_total_hits;

    event_record_time_start(stream_name);

    if (test)
        d_knns = knns;
    else {
        CUDA_CHECK(cudaMallocAsync(&d_knns, batch * bp.k * sizeof(int), stream));
        CUDA_CHECK(cudaMemcpyAsync(d_knns, knns, batch * bp.k * sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
    }

    CUDA_CHECK(cudaMallocAsync(&d_total_hits, sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(d_total_hits, 0, sizeof(int), stream));

    beam_test_knn_kernel<<<64, 32, (bp.k + 1) * sizeof(int), stream>>>(beam, d_knns, d_total_hits,
                                                                       batch, bp);
    CUDA_CHECK(
        cudaMemcpyAsync(&res.hits, d_total_hits, sizeof(int), cudaMemcpyDeviceToHost, stream));

    res.time += event_record_time_stop(stream_name, "beam_search") / 1000.0;

    if (!test) CUDA_CHECK(cudaFreeAsync(d_knns, stream));

    CUDA_CHECK(cudaFreeAsync(d_total_hits, stream));
}

void GPUFuncs::test_query_data_prepare(const float *data, const int *knns, int query_n) {
    cudaStream_t &stream = streams["search"].stream;
    CUDA_CHECK(cudaMemcpyAsync(test_query_data, data, query_n * bp.dim * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(test_query_knns, knns, query_n * bp.k * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
}

void GPUFuncs::search_prepare() {  // 图结构数据内存分配：
    cudaStream_t &stream = streams["search"].stream;

    CUDA_CHECK(cudaMallocAsync(&d_center, sizeof(float) * bp.dim, stream));
    CUDA_CHECK(cudaMemsetAsync(d_center, 0, sizeof(float) * bp.dim, stream));
    CUDA_CHECK(cudaMallocAsync(&d_dists, sizeof(float) * bp.base_n, stream));
    CUDA_CHECK(cudaMallocAsync(&graph_ep, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&closest_val, sizeof(float), stream));

    const int tpb = 32;
    const size_t shared_size = tpb * bp.dim * sizeof(float);
    compute_centroid_kernel<<<256, tpb, shared_size, stream>>>(d_b.data().get(), d_center,
                                                               bp.base_n, bp.dim);
    normalize_centroid_kernel<<<1, 128, 0, stream>>>(d_center, bp.base_n, bp.dim);

    if (bp.metric == DIST_METRIC::L2)
        compute_centroid_l2_distance_kernel<<<256, 1024, 0, stream>>>(d_b.data().get(), d_center,
                                                                      d_dists, bp);
    else
        compute_centroid_ip_distance_kernel<<<256, 1024, 0, stream>>>(d_b.data().get(), d_center,
                                                                      d_dists, bp);
#ifdef GIG
    // 每一个拓展节点需要的用于存储邻居的内存大小:
    bp.beam_expand_num =
        (GB(BEAM_MAX_MEM_SIZE) / BATCH / (sizeof(BCD) + sizeof(float)) - bp.beam_capacity) /
        bp.max_degree;  // 内存限制下最大可拓展节点数量
    bp.beam_expand_num = std::min(bp.beam_capacity, bp.beam_expand_num);
    bp.beam_size = bp.beam_capacity + bp.beam_expand_num * bp.max_degree;
    CUDA_CHECK(cudaMallocAsync(&beam, BATCH * bp.beam_size * sizeof(BCD), stream));
    CUDA_CHECK(cudaMallocAsync(&beam_dists, BATCH * bp.beam_size * sizeof(float), stream));
    // beam_visited 用位图表示，每个 int 数值（32 位）可表示 32 个状态
    CUDA_CHECK(
        cudaMallocAsync(&beam_visited, bp.visited_words_num * bp.base_n * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&beam_offsets, (BATCH + 1) * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&beam_converge, BATCH * sizeof(bool), stream));
    CUDA_CHECK(cudaMallocAsync(&beam_converge_num, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&test_query_data, TEST_SEARCH_QUERY_SIZE * bp.dim * sizeof(float),
                               stream));
    CUDA_CHECK(
        cudaMallocAsync(&test_query_knns, TEST_SEARCH_QUERY_SIZE * bp.k * sizeof(int), stream));

#endif
}

void GPUFuncs::search_free() {
    cudaStream_t &stream = streams["search"].stream;
    CUDA_CHECK(cudaFreeAsync(d_center, stream));
    CUDA_CHECK(cudaFreeAsync(d_dists, stream));
    CUDA_CHECK(cudaFreeAsync(graph_ep, stream));
    CUDA_CHECK(cudaFreeAsync(closest_val, stream));
#ifdef GIG
    CUDA_CHECK(cudaFreeAsync(beam, stream));
    CUDA_CHECK(cudaFreeAsync(beam_dists, stream));
    CUDA_CHECK(cudaFreeAsync(beam_offsets, stream));
    CUDA_CHECK(cudaFreeAsync(beam_visited, stream));
    CUDA_CHECK(cudaFreeAsync(beam_converge, stream));
    CUDA_CHECK(cudaFreeAsync(beam_converge_num, stream));
    CUDA_CHECK(cudaFreeAsync(test_query_data, stream));
    CUDA_CHECK(cudaFreeAsync(test_query_knns, stream));
#endif
}
}  // namespace efanna2e