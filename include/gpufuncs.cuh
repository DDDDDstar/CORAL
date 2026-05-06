#pragma once
#include <cuda_runtime.h>

#include <array>
#include <cfloat>
#include <cmath>
#include <condition_variable>
#include <future>
#include <iostream>
#include <mutex>
#include <numeric>  // accumulate
#include <shared_mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "cache.h"
#include "clock.cuh"
#include "efanna2e/neighbor.h"
#include "faiss_streamer.cuh"
#include "fileout.h"
#include "graph.cuh"
#include "ivf.cuh"
#include "myfaiss.h"
// #include "sample.cuh"
#include "uni.h"
#include "utils.cuh"

#ifndef KNN_CUH
#define KNN_CUH

namespace efanna2e {
#define SEARCH_BATCH 2048

struct MyStream {
   public:
    cudaStream_t stream;
    cudaEvent_t start, stop;
};

class Query_KNNs {
   public:
    Query_KNNs(const std::vector<int>& knns_in, int query_size, BP bp);
    Query_KNNs(Cache& cache, int query_size, BP bp);
    // void record(int qid, int* new_knns, int batch, cudaStream_t& stream);
    //     inline int* get(int qid) {
    // #ifdef GPU_TEST
    //         assert(qid < size);
    // #endif
    //         return knns[qid].data();
    //     }
    void get(const std::vector<int>& qids, std::vector<int>& knns_out, std::vector<int>& offsets);
    void get(const faiss::idx_t* ids, int* knns_out, int n);
    void get_and_add(const std::vector<int>& qids, const int* new_ids, std::vector<int>& knns_out,
                     std::vector<int>& offsets, int N, int query_per_ins = 1);
    void get_and_add(const std::vector<faiss::idx_t>& qids, const int* new_ids,
                     std::vector<int>& knns_out, size_t* offsets, int N, int query_num);
    void get_and_add(const std::vector<faiss::idx_t>& qids, const int* new_ids,
                     std::vector<int>& knns_out, size_t* offsets, int N);
    void get_and_add(const std::vector<faiss::idx_t>& qids, const int* new_ids, int* knns_out,
                     int N);
    // inline thrust::device_vector<int>::iterator get_iter(int qid = 0) {
    //     return knns.begin() + qid * bp.k;
    // }
    inline void add(int qid, int new_id) { knns[qid].push_back(new_id); }
    void del(const std::vector<int>& del_ids) {
// auto s = std::chrono::high_resolution_clock::now();
// std::vector<uint8_t> del_flag(bp.base_n, 0);
#pragma omp parallel for
        for (int id : del_ids) del_flag[id] = 1;
        // #pragma omp parallel for
        // std::unique_lock<std::shared_mutex> lock(mtx);
        // for (int k = 0; k < knns.size(); ++k) {
        //     auto& knn = knns[k];
        //     size_t i = 0;
        //     while (i < knn.size()) {
        //         int v = knn[i];
        //         if (del_flag[v]) {
        //             knn[i] = knn.back();
        //             knn.pop_back();
        //         } else {
        //             ++i;
        //         }
        //     }
        // }
        // float time = std::chrono::duration_cast<std::chrono::microseconds>(
        //                  std::chrono::high_resolution_clock::now() - s)
        //                  .count() /
        //              1000.0;
        // fo.print("delete nodes in query knns: " + TOS(time) + "s");
    }
    void get_query_of_base(int* ids, int N) {
        for (int i = 0; i < N; i++) {
            const int base_id = ids[i];
            ids[i] = query_of_base[base_id];
        }
    }
    void add_query_of_base(const int* ids, const int* qids, int N) {
#pragma omp parallel for
        for (int i = 0; i < N; i++) {
            assert(qids[i] < size);
            query_of_base[ids[i]] = qids[i];
        }
    }

   private:
    std::vector<std::vector<int>> knns;
    std::vector<int> query_of_base;
    std::vector<uint8_t> del_flag;
    std::shared_mutex mtx;
    int size;
    BP bp;
};

class QueryIndex {
   public:
    QueryIndex(Cache& knn_cache, float* h_queries, int nq, int base_n, int k, int dim,
               DIST_METRIC metric);
    QueryIndex(const std::string& filename, int base_n, int k, int dim, DIST_METRIC metric);
    QueryIndex(const std::string& filename, GPUResources* gpu_res, int device, cudaStream_t stream,
               int base_n, int k, int dim, DIST_METRIC metric);
    void set_gpu(cudaStream_t stream);
    ~QueryIndex() {
        delete query_knns;
        if (query_own) CUDA_CHECK(cudaFreeHost(h_queries));
        if (h_knns) CUDA_CHECK(cudaFreeHost(h_knns));
        if (d_queries) CUDA_CHECK(cudaFree(d_queries));
    };
    void build();
    void build_faiss();
    void Top1_Search(const float* h_queries, int* d_closest_ids_out, int num,
                     cudaStream_t& stream);
    void Top1_Search_faiss(const float* h_queries, std::vector<int>& closest_ids_out, int num);
    void knn_Search_faiss(const float* d_queries, int* h_knns_out, int num, int query_num);
    void knn_Search_faiss(const float* d_queries, const int* ids, std::vector<int>& knns_out,
                          size_t* offsets, int num);
    void knn_Search_faiss(const float* d_queries, const int* ids, int* knns_out, int num,
                          int query_num);
    void Topk_Search_faiss(const float* h_queries, std::vector<int>& closest_ids_out, int num,
                           int k);
    Query_KNNs& get_knns() { return *query_knns; }
    void get_query_of_base(const float* d_ins_vecs, int* ids, int* h_closest_qids, int batch,
                           int k, BP bp);

   private:
    bool query_own = false;
    int query_index_size;  // K;
    Query_KNNs* query_knns;
    BP bp;
    float *h_queries = nullptr, *d_queries = nullptr;

    GPUResources gr;
    GPUFlatL2* gpu_flat_index;
    std::vector<faiss::idx_t> res_faiss;
    // std::vector<int> res;
    std::vector<float> dists_res;
    int* h_knns = nullptr;

    thrust::device_vector<float> d_bucket_vecs, d_centroids;
    thrust::device_vector<int> d_bucket_vec_ids, d_cluster_offsets;
};

struct Enhance_Data {
    const int* d_iso_ids;
    int *d_iso_idxs, *d_iso_vec_idxs;
    int *d_idxs, *d_vec_idxs, *d_nbr_vec_idxs;  // 孤立节点的邻居的 idx 和邻居的旧邻居的 vec idx
    const int* d_search_nbr_ids;
    const float* d_search_nbr_dists;
    int batch;
    uint32_t pin_id;
    cudaStream_t stream;
    cudaEvent_t ev;

    Enhance_Data(int iso_n, BP bp) {
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        CUDA_CHECK(cudaMallocAsync(&d_iso_idxs, iso_n * sizeof(int), stream));
        CUDA_CHECK(cudaMallocAsync(&d_iso_vec_idxs, iso_n * sizeof(int), stream));
        CUDA_CHECK(cudaMallocAsync(&d_idxs, iso_n * bp.k * sizeof(int), stream));
        CUDA_CHECK(cudaMallocAsync(&d_vec_idxs, iso_n * bp.k * sizeof(int), stream));
        CUDA_CHECK(
            cudaMallocAsync(&d_nbr_vec_idxs, iso_n * bp.k * bp.max_degree * sizeof(int), stream));
    }
    ~Enhance_Data() {
        CUDA_CHECK(cudaFreeAsync(d_iso_idxs, stream));
        CUDA_CHECK(cudaFreeAsync(d_iso_vec_idxs, stream));
        CUDA_CHECK(cudaFreeAsync(d_idxs, stream));
        CUDA_CHECK(cudaFreeAsync(d_vec_idxs, stream));
        CUDA_CHECK(cudaFreeAsync(d_nbr_vec_idxs, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaEventDestroy(ev));
    }
    inline void prepare(const int* iso_ids, const int* search_nbr_ids,
                        const float* search_nbr_dists, int nap_batch, uint32_t id) {
        d_iso_ids = iso_ids;
        d_search_nbr_ids = search_nbr_ids;
        d_search_nbr_dists = search_nbr_dists;
        batch = nap_batch;
        pin_id = id;
    }
};

struct Enhance_Data1 {
    int* d_iso_ids = nullptr;
    Slot *d_iso_slots = nullptr, *h_iso_slots = nullptr, *d_search_slots = nullptr;
    // float* d_vecs;  // iso_vecs: [iso_n * dim] + search_vecs: [iso_n * k * dim] +
    // search_nbr_vecs: [iso_n * k * deg * dim]
    float* h_vecs = nullptr;  // [iso_n * k * dim]
    float* d_vecs = nullptr;  // [iso_n * k * dim]
    float *h_search_vecs = nullptr, *h_top_nbr_vecs = nullptr;
    float* d_search_vecs = nullptr;  // [iso_n * k * dim]
    const float* d_top_nbr_vecs = nullptr;
    int vec_num;
    float* d_search_dists = nullptr;
    float* d_dists2query = nullptr;
    int batch, max_batch;
    BP bp;
    bool need_write = false;
    cudaStream_t stream = nullptr;
    cudaEvent_t cev = nullptr, lev = nullptr;

    Enhance_Data1(int batch, BP bp) : max_batch(batch), bp(bp) {
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaEventCreate(&cev));
        CUDA_CHECK(cudaEventCreate(&lev));
        record(0);
        record(1);
        CUDA_CHECK(cudaMallocAsync(&d_iso_ids, batch * sizeof(int), stream));
        CUDA_CHECK(cudaMallocAsync(&d_iso_slots, batch * sizeof(Slot), stream));
        CUDA_CHECK(cudaMallocHost(&h_iso_slots, batch * sizeof(Slot)));
        vec_num = batch * bp.k;
        CUDA_CHECK(cudaMallocAsync(&d_search_slots, batch * bp.k * sizeof(Slot), stream));
        // CUDA_CHECK(cudaMallocAsync(&d_search_dists, batch * bp.k * sizeof(float), stream));
        CUDA_CHECK(
            cudaMallocAsync(&d_dists2query, batch * bp.k * bp.max_degree * sizeof(float), stream));
        if (PC.top_hit_num > 0) {
            vec_num += batch * bp.max_degree;
            CUDA_CHECK(cudaMallocAsync(&d_vecs, vec_num * bp.dim * sizeof(float), stream));
            CUDA_CHECK(cudaMallocHost(&h_vecs, vec_num * bp.dim * sizeof(float)));
            d_search_vecs = d_vecs + batch * bp.max_degree * bp.dim;
            h_search_vecs = h_vecs + batch * bp.max_degree * bp.dim;
            d_top_nbr_vecs = d_vecs;
            h_top_nbr_vecs = h_vecs;
        } else {
            CUDA_CHECK(cudaMallocAsync(&d_vecs, vec_num * bp.dim * sizeof(float), stream));
            CUDA_CHECK(cudaMallocHost(&h_vecs, vec_num * bp.dim * sizeof(float)));
            d_search_vecs = d_vecs;
            h_search_vecs = h_vecs;
        }
    }
    ~Enhance_Data1() {
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaFreeAsync(d_iso_ids, stream));
        CUDA_CHECK(cudaFreeAsync(d_iso_slots, stream));
        CUDA_CHECK(cudaFreeAsync(d_search_slots, stream));
        CUDA_CHECK(cudaFreeAsync(d_dists2query, stream));
        CUDA_CHECK(cudaFreeAsync(d_vecs, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaFreeHost(h_vecs));
        CUDA_CHECK(cudaFreeHost(h_iso_slots));

        CUDA_CHECK(cudaEventDestroy(cev));
        CUDA_CHECK(cudaEventDestroy(lev));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    void prepare(const int* h_iso_ids, int nap_batch) {
        assert(nap_batch <= max_batch);
        CUDA_CHECK(cudaMemcpyAsync(d_iso_ids, h_iso_ids, sizeof(int) * nap_batch,
                                   cudaMemcpyHostToDevice, stream));
        batch = nap_batch;
    }
    void load(float* d_search_dists, bool only_search = true) {
        this->d_search_dists = d_search_dists;
        if (only_search)
            CUDA_CHECK(cudaMemcpyAsync(d_search_vecs, h_search_vecs,
                                       batch * bp.k * bp.dim * sizeof(float),
                                       cudaMemcpyHostToDevice, stream));
        else
            CUDA_CHECK(cudaMemcpyAsync(d_vecs, h_vecs, vec_num * bp.dim * sizeof(float),
                                       cudaMemcpyHostToDevice, stream));
    }
    void write(Slot* h_iso_slots, Slot* h_search_slots) {
        CUDA_CHECK(cudaMemcpyAsync(h_iso_slots, d_iso_slots, sizeof(Slot) * batch,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_search_slots, d_search_slots, sizeof(Slot) * batch * bp.k,
                                   cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
        need_write = false;
    }
    inline void record(int type) {
        CUDA_CHECK(cudaEventRecord(type ? cev : lev, stream));  // 0 l, 1 c
    }
    inline void wait(cudaEvent_t ev) { CUDA_CHECK(cudaStreamWaitEvent(stream, ev)); }
};

struct NAPData {
    int *d_knn_ids, batch, k, batch_id;
    int *d_cache_idxs, *d_vec_cache_idxs, *d_nbr_vec_cache_idxs;
    CN *cand_nbrs, *cand_nbrs_sort;
    size_t* offsets;
    float *cand_dists, *cand_dists_sort;
    uint32_t* updated;
    cudaStream_t stream;
    cudaEvent_t start, end;

    NAPData(BP bp) : k(bp.k) {
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&end));
        const size_t pivot_num = PC.batch * bp.k,
                     total_cand_size = pivot_num * (bp.max_degree + bp.k);
        CUDA_CHECK(cudaMallocAsync(&d_knn_ids, sizeof(int) * pivot_num, stream));
        CUDA_CHECK(cudaMallocAsync(&d_cache_idxs, pivot_num * sizeof(int), stream));
        CUDA_CHECK(cudaMallocAsync(&d_vec_cache_idxs, pivot_num * sizeof(int), stream));
        CUDA_CHECK(cudaMallocAsync(&d_nbr_vec_cache_idxs, pivot_num * bp.max_degree * sizeof(int),
                                   stream));
        CUDA_CHECK(cudaMallocAsync(&cand_nbrs, total_cand_size * sizeof(CN), stream));
        CUDA_CHECK(cudaMallocAsync(&cand_nbrs_sort, total_cand_size * sizeof(CN), stream));
        CUDA_CHECK(cudaMallocAsync(&offsets, (pivot_num + 1) * sizeof(size_t), stream));
        CUDA_CHECK(cudaMallocAsync(&cand_dists, total_cand_size * sizeof(float), stream));
        CUDA_CHECK(cudaMallocAsync(&cand_dists_sort, total_cand_size * sizeof(float), stream));
        CUDA_CHECK(cudaMallocAsync(&updated, ((GPU_N + 31) / 32) * sizeof(uint32_t), stream));
    }
    ~NAPData() {
        CUDA_CHECK(cudaFreeAsync(d_knn_ids, stream));
        CUDA_CHECK(cudaFreeAsync(d_cache_idxs, stream));
        CUDA_CHECK(cudaFreeAsync(d_vec_cache_idxs, stream));
        CUDA_CHECK(cudaFreeAsync(d_nbr_vec_cache_idxs, stream));
        CUDA_CHECK(cudaFreeAsync(cand_nbrs, stream));
        CUDA_CHECK(cudaFreeAsync(cand_nbrs_sort, stream));
        CUDA_CHECK(cudaFreeAsync(offsets, stream));
        CUDA_CHECK(cudaFreeAsync(cand_dists, stream));
        CUDA_CHECK(cudaFreeAsync(cand_dists_sort, stream));
        CUDA_CHECK(cudaFreeAsync(updated, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(end));
    }
    void load_knns(const int* knn_ids, int b, int bid) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_knn_ids, knn_ids, b * k * sizeof(int), cudaMemcpyHostToDevice,
                                   stream));
        batch = b;
        batch_id = bid;
    }
    float sync() {
        float milliseconds;
        cudaEventRecord(end, stream);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&milliseconds, start, end);
        fo.print("batch " + TOS(batch_id) + " finished: time=" + TOS(milliseconds / 1000.0f) +
                 "s");
    }
};

struct UpdData {
    bool need_write = false;
    int *d_knn_ids, batch, batch_id, *h_knn_ids;
    BP bp;
    Slot* d_slots;      // batch * k，pivot 节点的数据
    float* d_vecs;      // batch * k * dim，pivot 节点的向量数据
    float* d_nbr_vecs;  // batch * k * deg * dim，旧节点的向量数据
    CN *cand_nbrs, *cand_nbrs_sort;
    size_t* offsets;
    float *cand_dists, *cand_dists_sort;
    // uint32_t* updated;
    cudaStream_t stream;
    cudaEvent_t start, end;
    UpdData() = default;
    UpdData(BP bp) : bp(bp) {
        const size_t pivot_num = PC.batch * bp.k,
                     total_cand_size = pivot_num * (bp.max_degree + bp.k);
        h_knn_ids = new int[pivot_num];
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&end));

        CUDA_CHECK(cudaMallocAsync(&d_knn_ids, sizeof(int) * pivot_num, stream));
        CUDA_CHECK(cudaMallocAsync(&d_slots, sizeof(Slot) * pivot_num, stream));
        CUDA_CHECK(cudaMallocAsync(
            &d_vecs, sizeof(float) * pivot_num * bp.dim * (1 + bp.max_degree), stream));

        CUDA_CHECK(cudaMallocAsync(&cand_nbrs, total_cand_size * sizeof(CN), stream));
        CUDA_CHECK(cudaMallocAsync(&cand_nbrs_sort, total_cand_size * sizeof(CN), stream));
        CUDA_CHECK(cudaMallocAsync(&offsets, (pivot_num + 1) * sizeof(size_t), stream));
        CUDA_CHECK(cudaMallocAsync(&cand_dists, total_cand_size * sizeof(float), stream));
        CUDA_CHECK(cudaMallocAsync(&cand_dists_sort, total_cand_size * sizeof(float), stream));
        // CUDA_CHECK(cudaMallocAsync(&updated, ((GPU_N + 31) / 32) * sizeof(uint32_t), stream));
    }
    ~UpdData() {
        CUDA_CHECK(cudaFreeAsync(d_knn_ids, stream));
        CUDA_CHECK(cudaFreeAsync(d_slots, stream));
        CUDA_CHECK(cudaFreeAsync(d_vecs, stream));
        CUDA_CHECK(cudaFreeAsync(cand_nbrs, stream));
        CUDA_CHECK(cudaFreeAsync(cand_nbrs_sort, stream));
        CUDA_CHECK(cudaFreeAsync(offsets, stream));
        CUDA_CHECK(cudaFreeAsync(cand_dists, stream));
        CUDA_CHECK(cudaFreeAsync(cand_dists_sort, stream));
        // CUDA_CHECK(cudaFreeAsync(updated, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(end));
        delete[] h_knn_ids;
    }
    void load_knns(const int* knn_ids, int b, int bid) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_knn_ids, knn_ids, b * bp.k * sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
        std::memcpy(h_knn_ids, knn_ids, b * bp.k * sizeof(int));
        batch = b;
        batch_id = bid;
        d_nbr_vecs = d_vecs + b * bp.k * bp.dim;
    }
    void load(const Slot* slots, const float* vecs) {
        const int pivot_num = batch * bp.k;
        CUDA_CHECK(cudaMemcpy(d_slots, slots, sizeof(Slot) * pivot_num, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vecs, vecs,
                              sizeof(float) * pivot_num * bp.dim * (1 + bp.max_degree),
                              cudaMemcpyHostToDevice));
    }
    void write(Slot* slots) {
        const int pivot_num = batch * bp.k;
        CUDA_CHECK(cudaMemcpyAsync(slots, d_slots, sizeof(Slot) * pivot_num,
                                   cudaMemcpyDeviceToHost, stream));
        // float milliseconds;
        cudaEventRecord(end, stream);
        cudaEventSynchronize(end);
        // if (batch_id % 10 == 0) {
        //     cudaEventElapsedTime(&milliseconds, start, end);
        //     fo.print("batch " + TOS(batch_id) + " finished: time=" + TOS(milliseconds / 1000.0f)
        //     +
        //              "s");
        // }
    }
};

template <typename TData>
struct NAPQueue {
    const int L = 2;
    std::vector<TData> datas;
    Slot* h_upd_slots = nullptr;
    float* h_upd_vecs = nullptr;
    std::atomic<int> num = 0;
    int head = 0, tail = 0;
    bool stop_flag = false;
    NAPQueue(BP bp) {
        datas.reserve(L);
        CUDA_CHECK(cudaMallocHost(&h_upd_slots, sizeof(Slot) * PC.batch * bp.k));
        CUDA_CHECK(cudaMallocHost(&h_upd_vecs,
                                  sizeof(float) * PC.batch * bp.k * bp.dim * (1 + bp.max_degree)));
        for (int i = 0; i < L; i++) datas.emplace_back(bp);
    }
    ~NAPQueue() {
        std::vector<TData>().swap(datas);
        if (h_upd_slots) CUDA_CHECK(cudaFreeHost(h_upd_slots));
        if (h_upd_vecs) CUDA_CHECK(cudaFreeHost(h_upd_vecs));
    }
    inline int getnum() { return num; }
    inline int size() { return L; }
    inline bool notfull() { return num < L; }
    inline bool notempty() { return num > 0; }
    inline bool empty() { return num == 0; }
    inline bool stop() { return stop_flag; }
    TData* newdata() {
        assert(num < L);
        if (tail == L) tail = 0;
        // CUDA_CHECK(cudaMemcpyAsync(data.d_knn_ids, h_knn_ids, k * batch * sizeof(int),
        //                            cudaMemcpyHostToDevice, data.stream));
        // data.batch = batch;
        // data.batch_id = batch_id;
        return datas.data() + tail;
    }
    int write_finish() {
        tail++;
        return ++num;
    }
    TData* data() {
        assert(num > 0);
        if (head == L) head = 0;
        return datas.data() + head;
    }
    int read_finish() {
        head++;
        return --num;
    }
};

struct Beam_Data1 {
    int *d_beam_ids = nullptr, *d_slot_idxs = nullptr, *d_ids = nullptr, *d_nums = nullptr;
    Slot* d_slots = nullptr;
    float *d_dists = nullptr, *d_farthests = nullptr;  //*d_nbr_vecs = nullptr
    uint8_t* d_expanded = nullptr;
    int* d_beams_hash = nullptr;
    // void load(Slot* h_slots, const float* h_nbr_vecs, int N, BP bp) {
    //     CUDA_CHECK(cudaMemcpy(d_slots, h_slots, N * sizeof(Slot), cudaMemcpyHostToDevice));
    //     CUDA_CHECK(cudaMemcpy(d_nbr_vecs, h_nbr_vecs, N * bp.max_degree * bp.dim *
    //     sizeof(float),
    //                           cudaMemcpyHostToDevice));
    // }
    void resize(int batch, BP bp) {
        if (d_nums) cudaFree(d_nums);
        if (d_farthests) cudaFree(d_farthests);
        if (d_expanded) cudaFree(d_expanded);
        if (d_ids) cudaFree(d_ids);
        if (d_dists) cudaFree(d_dists);
        if (d_beam_ids) cudaFree(d_beam_ids);
        if (d_slot_idxs) cudaFree(d_slot_idxs);
        if (d_slots) cudaFree(d_slots);
        // if (d_nbr_vecs) cudaFree(d_nbr_vecs);
        if (d_beams_hash) cudaFree(d_beams_hash);

        const size_t num1 = (size_t)batch * bp.beam_size, num2 = (size_t)batch * bp.beam_capacity;
        cudaMalloc(&d_nums, batch * sizeof(int));
        cudaMalloc(&d_farthests, batch * sizeof(float));
        cudaMalloc(&d_expanded, num2);
        cudaMalloc(&d_ids, num1 * sizeof(int));
        cudaMalloc(&d_dists, num1 * sizeof(float));
        cudaMalloc(&d_beam_ids, num2 * sizeof(int));
        cudaMalloc(&d_slots, num2 * sizeof(Slot));
        cudaMalloc(&d_slot_idxs, num2 * sizeof(int));
        cudaMalloc(&d_beams_hash, (size_t)batch * bp.hash_n * sizeof(int));
    }
    void load(const int* h_slot_idxs, int N) {
        CUDA_CHECK(cudaMemcpy(d_slot_idxs, h_slot_idxs, N * sizeof(int), cudaMemcpyHostToDevice));
    }
    ~Beam_Data1() {
        if (d_nums) cudaFree(d_nums);
        if (d_farthests) cudaFree(d_farthests);
        if (d_expanded) cudaFree(d_expanded);
        if (d_ids) cudaFree(d_ids);
        if (d_dists) cudaFree(d_dists);
        if (d_beam_ids) cudaFree(d_beam_ids);
        if (d_slots) cudaFree(d_slots);
        if (d_slot_idxs) cudaFree(d_slot_idxs);
        if (d_beams_hash) cudaFree(d_beams_hash);
        // if (d_visited) cudaFree(d_visited);
    }
};

template <typename BData>
struct Search_Data {
    int Q_, batch_;
    BCD* d_beams_ = nullptr;
    int *d_qids_ = nullptr, *d_finish_num_ = nullptr;
    uint32_t* d_visited_ = nullptr;
    int* d_beams_hash_ = nullptr;
    BData beam_data_;
    int* finish_num_ = nullptr;
    cudaStream_t stream_ = nullptr;
    Search_Data(int Q, BP bp, GraphType gt) : Q_(Q) {
        if (gt == GraphType::GPU) {
            // batch_ = 1;
            batch_ = std::min(SEARCH_BATCH * 2, Q);
            // fo.print("hash_n: " + TOS(bp.hash_n) + ", batch: " + TOS(batch_));
            CUDA_CHECK(cudaMalloc(&d_beams_, sizeof(BCD) * (size_t)batch_ * bp.beam_size));
            CUDA_CHECK(cudaMalloc(&d_visited_, sizeof(uint32_t) * (size_t)batch_ * bp.visit_n));
            // CUDA_CHECK(cudaMalloc(&d_beams_hash_, sizeof(int) * (size_t)batch_ * bp.hash_n));
        } else {
            batch_ = std::min(SEARCH_BATCH, Q / 2);
            // CUDA_CHECK(cudaMalloc(&d_beams_hash_, sizeof(int) * (size_t)(batch_ * bp.hash_n)));
            CUDA_CHECK(cudaMalloc(&d_qids_, sizeof(int) * batch_));
            CUDA_CHECK(cudaMalloc(&d_finish_num_, sizeof(int)));
            beam_data_.resize(batch_, bp);
            CUDA_CHECK(cudaMallocHost(&finish_num_, sizeof(int)));
            cudaStreamCreate(&stream_);
        }
    }
    ~Search_Data() {
        if (d_visited_) CUDA_CHECK(cudaFree(d_visited_));
        if (d_beams_) CUDA_CHECK(cudaFree(d_beams_));
        if (d_qids_) CUDA_CHECK(cudaFree(d_qids_));
        if (d_beams_hash_) CUDA_CHECK(cudaFree(d_beams_hash_));
        if (d_finish_num_) CUDA_CHECK(cudaFree(d_finish_num_));
        if (finish_num_) CUDA_CHECK(cudaFreeHost(finish_num_));
        if (stream_) cudaStreamDestroy(stream_);
    }
    inline void reset() {
        if (d_qids_) cudaMemset(d_qids_, 0xFF, sizeof(int) * batch_);
        if (d_finish_num_) cudaMemset(d_finish_num_, 0, sizeof(int));
    }
    bool converge() {
        cudaMemcpyAsync(finish_num_, d_finish_num_, sizeof(int), cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
        const int fn = *finish_num_;
        if (fn) fo.print("finish_num: " + TOS(fn));
        return fn >= batch_;
    }
};

class GPUFuncs {
   public:
    TaskExecutor executor;

    GPUFuncs(Graph* graph, const fpath& base_file, const float* h_base, const float* h_queries,
             int base_n, int query_n, int dim, int k, int max_degree, DIST_METRIC m,
             int beam_capacity, Mode mode, bool have_gt);
    GPUFuncs(Graph* graph, QueryIndex* qindex, const float* h_base, int base_n, int dim, int k,
             int max_degree, DIST_METRIC m, int beam_capacity, Mode mode);
    ~GPUFuncs();
    inline cudaStream_t& get_stream(std::string name) { return streams[name].stream; }
    // void query_data_kmeans(float* h_queries, int K);
    bool rebuild();
    int* knn_compute_faiss_hd(const float* queries, int batch, float& time_s);  // hot nodes delete
    void count_hits(const int* d_knns, int batch);
    void get_top_hit_nodes(int* d_top_ids, int topn);
    int* knn_compute_faiss(const float* queries, int batch, float& time_s);
    void knn_compute_faiss(const float* queries, int* knn_res, int batch, float& time_s);
    const int* knn_compute(const float* queries, int batch, int& index_size);
    void knn_compute(const float* queries, int batch, int* knn_res);

    void handle_knn_updates(const int* d_knn_ids, int batch_id, int batch);
    void graph_update(const int* knn_ids, int batch_id, int batch);
    // void neibor_aware_test(const std::vector<int>& h_knn_ids, NBD& h_upd_node_data);

    void search_verify(const int* ids_res, const int* ids_gt, Test_Result& tr, int Q, int k,
                       cudaStream_t stream = nullptr);
    void search_prepare(int Q, int beam_capacity = PC.test_beam_capacity);
    std::pair<int*, float*> test_query_prepare(const std::vector<float>& query_data);
    std::pair<int*, float*> test_query_load(const std::vector<float>& query_data,
                                            const std::vector<int>& gt_data, int k);
    void beam_search_test();

    float vector_insert(const float* insert_vector, const int* ins_ids, int batch);
    float vector_insert_base(const float* insert_vector, const int* ins_ids, int batch);
    float vector_delete(const int* del_ids, const std::vector<int>& del_innbr_offsets,
                        const int* del_innbrs, int del_num, int innbrs_num
                        // , int* del_old_nbrs, int* del_new_nbrs,int* del_new_degs
    );
    void del_node_mark(const std::vector<int>& del_ids);
    void gpu_neibor_aware(const int* d_vec_ids, int batch, int vec_num, cudaStream_t stream);
    void cpu_neibor_aware(NAPData* data, const DeviceView view);
    void disk_neibor_aware(UpdData* data);
    void ivf_build();

    inline void connect_enhance(QueryIndex* query_index) {
        this->query_index = query_index;

        enhance_prepare();
        if (graph->Type() == GraphType::GPU) {
            gpu_connect_enhance();
        } else if (!MEM_MODE) {
            cpu_connect_enhance();
        } else {
            disk_connect_enhance();
        }
        enhance_free();
    }

    inline void Build_Free() {
        cuda_check_last_error("build free");
        static bool flag = true;  // 只执行一次
        if (flag) {
            assert(nap_queue == nullptr || nap_queue->stop_flag);
            assert(upd_queue == nullptr || upd_queue->stop_flag);
            flag = false;
            knn_free();
            upd_free();
            fo.print("Build free(knn and upd).");
        }
    }

    inline void NAP_Stop() {
        if (nap_queue) nap_queue->stop_flag = true;
        if (upd_queue) upd_queue->stop_flag = true;
        write_cv.notify_one();
        read_cv.notify_one();
    }

    inline Test_Result Search(const float* queries, int* ids_res, float* dists_res, int Q,
                              std::vector<int>& start_node_ids, int k = 0) {
        if (graph->Type() == GraphType::GPU)
            return search_gpu(queries, ids_res, dists_res, Q, k, start_node_ids, false);
        // else if (!MEM_MODE)
        //     return search_cpu(queries, ids_res, dists_res, Q, k, reset);
        else if (graph->Type() == GraphType::CPU)
            return search_in_cpu(queries, ids_res, dists_res, Q, k, start_node_ids);
        else
            return search_in_cpu_for_disk(queries, ids_res, dists_res, Q, k, start_node_ids);
    }

    Test_Result test_search(std::vector<int>& start_node_ids) {
        Test_Result res;

        search_prepare(TEST_SEARCH_QUERY_SIZE, PC.test_beam_capacity);

        int* d_ids_res;
        cudaMalloc(&d_ids_res, TEST_SEARCH_QUERY_SIZE * gtk * sizeof(int));

        res = Search(test_query_data, d_ids_res, nullptr, TEST_SEARCH_QUERY_SIZE, start_node_ids,
                     gtk);
        search_verify(d_ids_res, test_query_knns, res, TEST_SEARCH_QUERY_SIZE, gtk);

        cudaFree(d_ids_res);
        return res;
    }

    std::vector<int> get_search_start_nodes_ivf(int M, int reset) {
        std::vector<int> res;
        ivf_builder->get_top_hit_nodes(res, M);
        if (reset) {
            auto& start_ids = graph->get_start_ids();
            start_ids = res;
        }
        return res;
    }

    void save_hits() {
        if (bp.base_n <= GPU_N) {
            cudaMemcpy(h_hits, d_hits, bp.base_n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            double var = welford_variance(h_hits, bp.base_n);

            std::vector<int> idx(bp.base_n);
            std::iota(idx.begin(), idx.end(), 0);
            std::nth_element(idx.begin(), idx.begin() + bp.base_n / 100, idx.end(),
                             [&](int a, int b) { return h_hits[a] > h_hits[b]; });
            size_t hit_count = 0;
            for (int i = 0; i < bp.base_n / 100; i++) hit_count += h_hits[idx[i]];

            fo.iprint("The var of hits: " + TOS(var) + ", top1%-hits: " + TOS(hit_count));
        } else
            ivf_builder->save_hits();
    }

    // Distance* distance_;

   private:
    Mode mode_;

    bool construct;
    QueryIndex* query_index = nullptr;
    Graph* graph = nullptr;
    IVFBuilder* ivf_builder = nullptr;
    const float *h_base = nullptr, *h_queries = nullptr;
    BP bp;
    int gtk;

    float* d_q;

    // knn 相关：
    fpath base_file;
    // std::ifstream base_file_in;
    bool round_copy = true;
    float* d_queries = nullptr;
    thrust::device_vector<faiss::idx_t> knn_res_faiss;
    thrust::device_vector<float> knn_dists_res;
    thrust::device_vector<int> d_knn_res;

    uint32_t *d_hits = nullptr, *h_hits = nullptr;  //*d_hot_deleted = nullptr;
    uint32_t *d_over_cnt = nullptr, *h_over_cnt = nullptr;
    faiss::idx_t *d_compact_ids = nullptr, *h_compact_ids = nullptr;

    // std::vector<uint8_t> hits;
    // size_t over_cnt = 0;

    MyFaiss* myfaiss = nullptr;
    // ExactKnnStreamer* knn_streamer = nullptr;

    // upd 相关：
    thrust::device_vector<CN> cand_nbrs, cand_nbrs_sort;
    thrust::device_vector<size_t> offsets;
    thrust::device_vector<float> cand_dists, cand_dists_sort;
    thrust::device_vector<uint32_t> d_updated;
    int tile_k;
    std::shared_mutex write_mtx, read_mtx;
    std::condition_variable_any write_cv, read_cv;
    NAPQueue<NAPData>* nap_queue = nullptr;
    NAPQueue<UpdData>* upd_queue = nullptr;
    std::thread nap_thread;
    ull *d_total_degree, *h_total_degree, *d_noniso_num, *h_noniso_num;

    // enhance 相关：
    thrust::device_vector<CN> d_cand_nbrs, d_cand_nbrs_sort, d_iso_cand_nbrs, d_iso_cand_nbrs_sort;
    // thrust::device_vector<size_t> d_offsets;
    thrust::device_vector<int> d_cand_ids;
    thrust::device_vector<uint32_t> d_enhance_updated;
    thrust::device_vector<float> d_cand_dists, d_cand_dists_sort;
    std::vector<int> h_cand_idxs, h_cand_ids;
    std::vector<float> h_cand_dists;
    int *h_search_ids = nullptr, *h_iso_ids = nullptr;
    // float* h_iso_vecs = nullptr;
    Slot *h_search_slots = nullptr, *h_iso_slots = nullptr;
    // float* h_search_vecs = nullptr;
    size_t *h_offsets = nullptr, *d_offsets = nullptr;
    thrust::device_vector<int> d_search_res;
    thrust::device_vector<float> d_search_dist_res;
    std::vector<Enhance_Data> edatas;
    std::vector<Enhance_Data1*> edatas1;

    // search 相关：
    // uint32_t *d_deleted;
    thrust::device_vector<int> d_start_vec_idxs;
    std::vector<Search_Data<Beam_Data>*> search_data;
    std::vector<Search_Data<Beam_Data1>*> search_data1;
    float* test_query_data;
    int* test_query_knns;
    float *h_start_vecs = nullptr, *h_nbr_vecs = nullptr;
    int *h_beam_ids = nullptr, *h_slot_idxs = nullptr;
    Slot* h_slots = nullptr;

    // insert 相关：
    // int *h_upd_nbrs, *h_new_nbrs;
    std::vector<int> h_ins_ids, h_k_plus_ones, h_paired_query_ids;
    thrust::device_vector<int> d_ins_ids, d_paired_query_ids, d_new_nbr_ids, d_upd_nbrs,
        d_paired_knns, d_knn_offsets, d_paired_knns1, d_knn_offsets1, d_search_knns;
    thrust::device_vector<float> d_insert_vectors;
    thrust::device_vector<CN> ins_cand_nbrs, ins_cand_nbrs_sort;
    std::vector<int> paired_query_knns, query_knn_offsets;
    float* d_ins_vecs = nullptr;

    // delete 相关：
    // NBD* d_del_innbr_data_buffer;
    size_t del_hash_cap;
    // int *d_innbr_num = nullptr, *h_innbr_num = nullptr, *h_del_old_nbrs = nullptr,
    //     *h_del_new_nbrs = nullptr, *h_del_new_degs = nullptr;
    thrust::device_vector<uint32_t> d_deleted;
    thrust::device_vector<int> d_del_old_nbrs, d_del_new_nbrs, d_del_new_degs, d_del_hashset;
    thrust::device_vector<int> d_del_innbr_offsets, d_del_ids, d_all_del_ids, d_del_innbr_ids,
        d_innbr_ids;
    thrust::device_vector<size_t> del_offsets;
    thrust::device_vector<float> del_cand_dists, del_cand_dists_sort;
    thrust::device_vector<CN> del_cand_nbrs, del_cand_nbrs_sort;
    cudaEvent_t del_event, cpy_event, upd_event;

    const std::string stream_names[4] = {"knn", "upd", "search", "upd_cpy"};
    std::unordered_map<std::string, MyStream> streams;

    int blocknum_per_query = 4;
    int knn_threads = 1024;
    int upd_threads = 64;

    size_t shared_mem_per_block;

    // knn_compute 预分配内存指针
    float *d_all_dist, *d_all_dist_sort;  // 存储所有查询的距离与索引结果，用于之后排序
    int *d_all_idx, *d_all_idx_sort, *d_topk_offsets;

    inline GraphType graph_type() { return graph->Type(); }

    void knn_prepare(bool have_gt, int batch = PC.batch);
    void knn_free();
    void upd_prepare(bool construct, int batch, int nap_num);
    void upd_free();
    void search_free();
    void update_prepare();
    void update_free();
    void enhance_prepare();
    void enhance_free();
    void enhance_neibor_aware_gpu(const int* d_top_ids, const int* d_search_nbr_ids,
                                  const float* d_search_nbr_dists, int batch, cudaStream_t stream);
    void enhance_neibor_aware_gpu(const int* d_iso_ids, const int* d_search_nbr_ids,
                                  const float* d_search_nbr_dists, int batch, int knn_num,
                                  cudaStream_t stream);
    void enhance_neibor_aware_gpu(const int* d_iso_ids, const int* d_search_nbr_ids,
                                  const float* d_search_nbr_dists, const size_t* d_offsets,
                                  int batch, int total_num, cudaStream_t stream);
    void enhance_neibor_aware_cpu(const DeviceView& view, Enhance_Data& data);
    void enhance_neibor_aware_disk(Enhance_Data1& data);
    void enhance_neibor_aware_disk_top(Enhance_Data1& data);

    Test_Result search_cpu(const float* queries, int* ids_res, float* dists_res, int Q, int k,
                           int reset);
    Test_Result search_in_cpu(const float* queries, int* ids_res, float* dists_res, int Q, int k,
                              std::vector<int>& start_node_ids);
    Test_Result search_in_cpu_for_disk(const float* queries, int* ids_res, float* dists_res, int Q,
                                       int k, std::vector<int>& start_node_ids);
    Test_Result search_disk(const float* queries, int* ids_res, float* dists_res, int Q, int k,
                            int reset);
    Test_Result search_gpu(const float* queries, int* ids_res, float* dists_res, int Q, int k,
                           std::vector<int>& start_node_ids, bool have_delete,
                           cudaStream_t stream = nullptr);

    void NAP_loop();
    void Upd_loop();

    void gpu_connect_enhance();
    void cpu_connect_enhance();
    void disk_connect_enhance();
    int collect_iso_nodes(int max_num);
    void search_and_nap(int num, GpuClockCache* gpu_cache, const DeviceView& view);
    void search_and_nap(int num);
    void search_and_nap(const int* d_iso_ids, const float* d_iso_vecs, int iso_num,
                        cudaStream_t stream);

    int obtain_paired_knns(const float* d_insert_vectors, const int* ins_ids, int*& h_knns_out,
                           int batch);
};

void idx_dist_extract(CN* nbrs, int* idxs, float* dists, cudaStream_t& stream, int num, BP bp);
void dist_extract_from_CN(CN* nbrs, float* dists, cudaStream_t stream, int num, BP bp);

// __global__ void candidate_ignore_kernel(CN* cand_nbrs, int* sort_idxs, int* nbr_num,
//                                         const int new_nbr_i, const int pivot_num, int
//                                         cand_size, BP bp);
__global__ void candidate_ignore_kernel(const float* __restrict__ vecs, CN* __restrict__ cand_nbrs,
                                        const int pivot_num, int cand_size, BP bp,
                                        float lamda = 0.0);
__global__ void candidate_ignore_kernel_for_iso(const float* __restrict__ vecs,
                                                CN* __restrict__ cand_nbrs, const int pivot_num,
                                                int cand_size, BP bp);
__global__ void update_new_nbrs_kernel(const int* __restrict__ pivot_ids, Slot* __restrict__ slots,
                                       CN* __restrict__ cand_nbrs, uint32_t* __restrict__ updated,
                                       ull* __restrict__ total_degree,
                                       ull* __restrict__ noniso_num, int pivot_num, int cand_size,
                                       BP bp, int need_discard);
// __global__ void update_new_nbrs_kernel(Slot* __restrict__ slots, CN* __restrict__ cand_nbrs,
//                                        ull* __restrict__ total_degree,
//                                        ull* __restrict__ noniso_num, int pivot_num, int
//                                        cand_size, BP bp);
// 根据 candidate_ignore_kernel 的结果（d_cand_nbrs）,
// 提取每个批次的 KNN 中每个 pivot 的邻居 base_id,
// 结果存在 new_nbrs 中，大小：batch * k * max_degree
__global__ void get_new_nbrs_kernel(CN* cand_nbrs, const int* sort_idxs, int* new_nbr_ids,
                                    float* new_nbr_dists, int cand_size, BP bp);

__global__ void topk_copy_kernel(int* d_knn_res, const int* d_all_idx, int batch, BP bp, int n);

__device__ __forceinline__ CN* get_CN(CN* cand_nbrs, const int* sort_idxs, int i, int group_id,
                                      int pivot_idx, int vec_num, int cand_size) {
    // 排序后第 i 个邻居向量在原始未排序 kNNs 中的索引
    return cand_nbrs + group_id * vec_num * cand_size + pivot_idx * cand_size +
           sort_idxs[group_id * vec_num * cand_size + pivot_idx * cand_size + i];
}

__device__ __forceinline__ CN* get_CN(CN* cand_nbrs, const int* sort_idxs, int nbr_i, int pivot_i,
                                      int cand_size) {
    // 排序后第 i 个邻居向量在原始未排序 kNNs 中的索引
    const int offset = pivot_i * cand_size;
    return cand_nbrs + offset + sort_idxs[offset + nbr_i];
}

// sample.cu:
__global__ void centroid_init_kernel(const float* __restrict__ vecs, const int* __restrict__ idxs,
                                     float* __restrict__ centroids, BP bp, int K);

__global__ void gather_sub_queries_kernel(const float* __restrict__ queries,
                                          const int* __restrict__ idxs,
                                          float* __restrict__ sub_queries, int sub_n, int dim);

__global__ void cluster_assign_kernel(const float* __restrict__ queries,
                                      const float* __restrict__ centroids,
                                      int* __restrict__ labels,  // 输出: 每个样本的簇 ID
                                      int* __restrict__ not_converge, BP bp, int K, int n);

__global__ void compute_centroid_kernel(const float* __restrict__ queries,
                                        const int* __restrict__ labels,
                                        float* __restrict__ vec_sums,
                                        float* __restrict__ centroids, BP bp, int K, int sub_n);

__global__ void cluster_size_statistics_kernel(const int* __restrict__ labels,
                                               int* __restrict__ cluster_sizes, int K, int n);

}  // namespace efanna2e

#endif  // KNN_CUH
