#include <cfloat>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>

#include "fileout.h"
#include "gpufuncs.cuh"
#include "utils.cuh"

using namespace efanna2e;

#ifdef GIG
void GPUFuncs::cand_ignore_test(const int *h_knn_idxs, CN *h_cand_nbrs, int *h_graph_res,
                                int *h_graph_deg_res, float *h_graph_dist_res) {
    const std::string name = "upd";
    cudaStream_t &stream = streams[name].stream;
    int *idxs, *h_idxs, *idxs_sort, *offsets, *nbr_num;
    float *h_dists, *dists, *dists_sort;
    CN *cand_nbrs;
    const int cand_size = bp.max_degree + bp.k, total_num = TEST_BATCH * bp.k * cand_size;
    CUDA_CHECK(cudaMallocAsync(&idxs, total_num * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&idxs_sort, total_num * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&dists, total_num * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&dists_sort, total_num * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&cand_nbrs, total_num * sizeof(CN), stream));
    CUDA_CHECK(cudaMallocAsync(&offsets, (TEST_BATCH * bp.k + 1) * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&nbr_num, TEST_BATCH * bp.k * sizeof(int), stream));

    CUDA_CHECK(cudaMallocHost(&h_idxs, total_num * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&h_dists, total_num * sizeof(float)));

    CUDA_CHECK(cudaMemcpyAsync(cand_nbrs, h_cand_nbrs, total_num * sizeof(CN),
                               cudaMemcpyHostToDevice, stream));

    idx_dist_extract(cand_nbrs, idxs, dists, stream, total_num, bp);

    auto res = segmented_sort_pairs(idxs, dists, idxs_sort, dists_sort, offsets, stream,
                                    TEST_BATCH * bp.k, cand_size);

    CUDA_CHECK(cudaMemsetAsync(nbr_num, 0, TEST_BATCH * bp.k * sizeof(int), stream));
    for (int i = 0; i < cand_size; ++i)
        candidate_ignore_kernel<<<TEST_BATCH, 128, 0, stream>>>(
            d_b.data().get(), cand_nbrs, res.first, nbr_num, i, TEST_BATCH, cand_size, bp.k, bp);

    int *knn_idxs;
    CUDA_CHECK(cudaMallocAsync(&knn_idxs, TEST_BATCH * bp.k * sizeof(int), stream));
    CUDA_CHECK(cudaMemcpyAsync(knn_idxs, h_knn_idxs, TEST_BATCH * bp.k * sizeof(int),
                               cudaMemcpyHostToDevice, stream));

    update_new_nbrs_kernel<<<TEST_BATCH, 64, 0, stream>>>(  // 将新邻居更新到 Graph 中
        knn_idxs, res.first, graph.data().get(), graph_deg.data().get(), graph_dist.data().get(),
        cand_nbrs, bp.k, TEST_BATCH, bp);

    CUDA_CHECK(cudaMemcpyAsync(h_graph_res, graph.data().get(),
                               bp.base_n * bp.max_degree * sizeof(int), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(h_graph_deg_res, graph_deg.data().get(), bp.base_n * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_graph_dist_res, graph_dist.data().get(),
                               bp.base_n * bp.max_degree * sizeof(int), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFreeAsync(idxs, stream));
    CUDA_CHECK(cudaFreeAsync(idxs_sort, stream));
    CUDA_CHECK(cudaFreeAsync(dists, stream));
    CUDA_CHECK(cudaFreeAsync(dists_sort, stream));
    CUDA_CHECK(cudaFreeAsync(cand_nbrs, stream));
    CUDA_CHECK(cudaFreeAsync(offsets, stream));
    CUDA_CHECK(cudaFreeAsync(nbr_num, stream));
    CUDA_CHECK(cudaFreeHost(h_idxs));
    CUDA_CHECK(cudaFreeHost(h_dists));
}
#else
void GPUFuncs::cand_ignore_test(CN *h_cand_nbrs, int *h_new_nbr_ids_res,
                                float *h_new_nbr_dists_res) {
    const std::string name = "upd";
    cudaStream_t &stream = streams[name].stream;
    int *idxs, *h_idxs, *idxs_sort, *offsets, *nbr_num, *new_nbr_ids;
    float *h_dists, *dists, *dists_sort, *new_nbr_dists;
    CN *cand_nbrs;
    const int total_num = TEST_BATCH * bp.k * bp.cand_size,
              total_nbr_num = TEST_BATCH * bp.k * bp.max_degree;
    CUDA_CHECK(cudaMallocAsync(&idxs, total_num * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&idxs_sort, total_num * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&dists, total_num * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&dists_sort, total_num * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&cand_nbrs, total_num * sizeof(CN), stream));
    CUDA_CHECK(cudaMallocAsync(&offsets, (TEST_BATCH * bp.k + 1) * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&nbr_num, TEST_BATCH * bp.k * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&new_nbr_ids, total_nbr_num * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&new_nbr_dists, total_nbr_num * sizeof(float), stream));

    CUDA_CHECK(cudaMallocHost(&h_idxs, total_num * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&h_dists, total_num * sizeof(float)));

    CUDA_CHECK(cudaMemcpyAsync(cand_nbrs, h_cand_nbrs, total_num * sizeof(CN),
                               cudaMemcpyHostToDevice, stream));

    idx_dist_extract(cand_nbrs, idxs, dists, stream, total_num, bp);

    auto res = segmented_sort_pairs(idxs, dists, idxs_sort, dists_sort, offsets, stream,
                                    TEST_BATCH * bp.k, bp.cand_size);

    CUDA_CHECK(cudaMemsetAsync(nbr_num, 0, TEST_BATCH * bp.k * sizeof(int), stream));
    for (int i = 0; i < bp.cand_size; ++i)
        candidate_ignore_kernel<<<TEST_BATCH, 128, 0, stream>>>(d_b, cand_nbrs, res.first, nbr_num,
                                                                i, TEST_BATCH, bp);

    CUDA_CHECK(cudaMemsetAsync(new_nbr_ids, -1, total_nbr_num * sizeof(int), stream));
    get_new_nbrs_kernel<<<TEST_BATCH, 64, 0, stream>>>(cand_nbrs, res.first, new_nbr_ids,
                                                       new_nbr_dists, bp);
    CUDA_CHECK(cudaMemcpyAsync(h_new_nbr_ids_res, new_nbr_ids, total_nbr_num * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_new_nbr_dists_res, new_nbr_dists, total_nbr_num * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaFreeAsync(idxs, stream));
    CUDA_CHECK(cudaFreeAsync(idxs_sort, stream));
    CUDA_CHECK(cudaFreeAsync(dists, stream));
    CUDA_CHECK(cudaFreeAsync(dists_sort, stream));
    CUDA_CHECK(cudaFreeAsync(cand_nbrs, stream));
    CUDA_CHECK(cudaFreeAsync(offsets, stream));
    CUDA_CHECK(cudaFreeAsync(nbr_num, stream));
    CUDA_CHECK(cudaFreeAsync(new_nbr_ids, stream));
    CUDA_CHECK(cudaFreeAsync(new_nbr_dists, stream));
    CUDA_CHECK(cudaFreeHost(h_idxs));
    CUDA_CHECK(cudaFreeHost(h_dists));

    CUDA_CHECK(cudaStreamSynchronize(stream));
}
#endif

void GPUFuncs::cand_compute_test(const int *h_knn_idxs, CN *h_cand_nbrs_res) {
    cudaStream_t &stream = streams["upd"].stream;
    int *knn_idxs, *old_nbr_nums;
    CUDA_CHECK(cudaMallocAsync(&knn_idxs, TEST_BATCH * bp.k * sizeof(int), stream));
    CUDA_CHECK(cudaMemcpyAsync(knn_idxs, h_knn_idxs, TEST_BATCH * bp.k * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
#ifndef GIG
    CUDA_CHECK(cudaMallocAsync(&old_nbr_nums, TEST_BATCH * bp.k * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(old_nbr_nums, 0, TEST_BATCH * bp.k * sizeof(int), stream));
#endif

#ifdef algo0
    pivot_to_others_dist_compute_kernel<<<TEST_BATCH, upd_threads, bp.dim * sizeof(float),
                                          stream>>>(d_b, knn_idxs, d_cand_nbrs, bp);

    add_reverse_kernel<<<TEST_BATCH, 32, 0, stream>>>(d_cand_nbrs, nullptr, nullptr, old_nbr_nums,
                                                      bp);
#else
    const int cand_size = bp.max_degree + bp.k, size = TEST_BATCH * bp.k * cand_size;
    thrust::device_vector<Candidate_Neighbor> cand_nbrs(size);
    // 共享内存最大向量存储数量:
    const static size_t max_vec_num =
        (shared_mem_per_block - bp.k * sizeof(int)) / (bp.dim * sizeof(float));
    // 每个 block 根据共享内存大小限制分块处理对应的 KNN:
    const static size_t tile_k = std::min<size_t>(bp.k, max_vec_num);
    // 每个 block 的实际使用的共享内存大小:
    const static size_t shared_size = tile_k * bp.dim * sizeof(float) + bp.k * sizeof(int);
    // 验证是否超限:
    assert(shared_size <= shared_mem_per_block && "Shared memory exceeds limit");

    fo.iprint("knn_dist_compute tile_k: " + std::to_string(tile_k) + ", shared_size/max: " +
              std::to_string(shared_size) + "/" + std::to_string(shared_mem_per_block));
#ifdef GIG
    pair_dist_compute_kernel<<<TEST_BATCH, upd_threads, shared_size, stream>>>(
        d_b.data().get(), knn_idxs, graph.data().get(), graph_dist.data().get(),
        graph_deg.data().get(), cand_nbrs.data().get(), tile_k, bp.k, TEST_BATCH, bp);
#else
    knn_dist_compute_kernel<<<TEST_BATCH, upd_threads, shared_size, stream>>>(
        d_b.data().get(), knn_idxs, nullptr, nullptr, old_nbr_nums, d_cand_nbrs, tile_k, bp);
    CUDA_CHECK(cudaFreeAsync(old_nbr_nums, stream));
#endif
#endif
    CUDA_CHECK(cudaMemcpyAsync(h_cand_nbrs_res, cand_nbrs.data().get(),
                               TEST_BATCH * bp.k * cand_size * sizeof(CN), cudaMemcpyDeviceToHost,
                               stream));

    CUDA_CHECK(cudaFreeAsync(knn_idxs, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void GPUFuncs::compute_knn_dist_test(const float *h_test_query, float *h_res_dists,
                                     int *h_res_idxs) {
    cudaStream_t &stream = streams["knn"].stream;
    float *query, *dists;
    int *idxs;
    CUDA_CHECK(cudaMallocAsync(&query, TEST_BATCH * bp.dim * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&dists, TEST_BATCH * bp.base_n * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&idxs, TEST_BATCH * bp.base_n * sizeof(int), stream));
    CUDA_CHECK(cudaMemcpyAsync(query, h_test_query, TEST_BATCH * bp.dim * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

    compute_knn_dist_kernel<<<TEST_BATCH * blocknum_per_query, 1024, bp.dim * sizeof(float),
                              stream>>>(query, d_b.data().get(), dists, idxs, TEST_BATCH, bp);

    CUDA_CHECK(cudaMemcpyAsync(h_res_dists, dists, TEST_BATCH * bp.base_n * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_res_idxs, idxs, TEST_BATCH * bp.base_n * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaFreeAsync(query, stream));
    CUDA_CHECK(cudaFreeAsync(dists, stream));
    CUDA_CHECK(cudaFreeAsync(idxs, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void GPUFuncs::topk_test(const float *h_dists, const int *h_idxs, int *h_res_topk_idxs,
                         float *h_res_topk_dists) {
    cudaStream_t &stream = streams["knn"].stream;
    float *dists, *sort_dists, *res_topk_dists;
    int *idxs, *sort_idxs, *res_topk_idxs, *offsets;
    CUDA_CHECK(cudaMallocAsync(&dists, TEST_BATCH * bp.base_n * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&sort_dists, TEST_BATCH * bp.base_n * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&idxs, TEST_BATCH * bp.base_n * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&sort_idxs, TEST_BATCH * bp.base_n * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&res_topk_idxs, TEST_BATCH * bp.k * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&res_topk_dists, TEST_BATCH * bp.k * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&offsets, (TEST_BATCH + 1) * sizeof(int), stream));

    CUDA_CHECK(cudaMemcpyAsync(dists, h_dists, TEST_BATCH * bp.base_n * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(idxs, h_idxs, TEST_BATCH * bp.base_n * sizeof(int),
                               cudaMemcpyHostToDevice, stream));

    auto res = segmented_sort_pairs(idxs, dists, sort_idxs, sort_dists, offsets, stream,
                                    TEST_BATCH, bp.base_n);

    topk_copy_kernel<<<TEST_BATCH, 64, 0, stream>>>(res_topk_idxs, res.first, res_topk_dists,
                                                    res.second, TEST_BATCH, bp);

    CUDA_CHECK(cudaMemcpyAsync(h_res_topk_idxs, res_topk_idxs, TEST_BATCH * bp.k * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_res_topk_dists, res_topk_dists, TEST_BATCH * bp.k * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaFreeAsync(dists, stream));
    CUDA_CHECK(cudaFreeAsync(idxs, stream));
    CUDA_CHECK(cudaFreeAsync(sort_idxs, stream));
    CUDA_CHECK(cudaFreeAsync(sort_dists, stream));
    CUDA_CHECK(cudaFreeAsync(offsets, stream));
    CUDA_CHECK(cudaFreeAsync(res_topk_idxs, stream));
    CUDA_CHECK(cudaFreeAsync(res_topk_dists, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void GPUFuncs::beam_search_test() {}