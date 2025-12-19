#pragma once
#include <cuda_runtime.h>
#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>

#include <array>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <mutex>
#include <numeric>  // accumulate
#include <shared_mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "fileout.h"
#include "graph.cuh"
#include "uni.h"

#ifndef KNN_CUH
#define KNN_CUH

namespace efanna2e {
#define UNROLL_FACTOR 4  // 根据 dim 调整，建议 4 或 8

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl;                   \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

#define NVML_CHECK(call)                                                                        \
    do {                                                                                        \
        nvmlReturn_t result = call;                                                             \
        if (result != NVML_SUCCESS) {                                                           \
            std::cerr << "NVML Error: " << nvmlErrorString(result) << " at " << __FILE__ << ":" \
                      << __LINE__ << std::endl;                                                 \
            exit(EXIT_FAILURE);                                                                 \
        }                                                                                       \
    } while (0)

struct MyStream {
   public:
    cudaStream_t stream;
    cudaEvent_t start, stop;
};

struct Beam_Candidate_D {
    int id;
    float dist;
    bool expanded;
};
using BCD = Beam_Candidate_D;

class Cluster_Member {
   public:
    int c_i;    // cluster idx
    float dtc;  // distance to centroid

    Cluster_Member() : c_i(-1) {}
};
using CM = Cluster_Member;

struct Graph_Data_D {
    int *graph, *graph_deg, *ep;
    float* graph_dist;
    std::shared_mutex& mtx;

    Graph_Data_D(int* g, int* g_deg, float* g_dist, int* ep, std::shared_mutex& m)
        : graph(g), graph_deg(g_deg), graph_dist(g_dist), ep(ep), mtx(m) {}
    Graph_Data_D(std::shared_mutex& m) : mtx(m) {}
};
using GDD = Graph_Data_D;

class Avg_Time {
   private:
    int n;
    float time;

   public:
    Avg_Time() : n(0), time(0) {}
    Avg_Time(const std::string& time_str, const std::string& n_str) {
        time = std::stof(time_str);
        n = std::stoi(n_str);
    }
    void add(float t) {
        n++;
        time += t;
    }
    inline float get() { return time / n; };
    inline float get_total() { return time; };
    inline int get_n() { return n; };
    // 重载 < 运算符，用于排序
    bool operator<(const Avg_Time& other) const {
        return time < other.time;  // 按年龄升序
    }
    std::string serialize() { return std::to_string(time) + "," + std::to_string(n); }
    void deserialize(const std::string& str) {
        std::istringstream iss(str);
        std::string temp;
        std::getline(iss, temp, ',');
        time = std::stof(temp);
        std::getline(iss, temp, ',');
        n = std::stoi(temp);
    }
};

class Query_KNNs {
   public:
    Query_KNNs(const std::vector<int>& knns_in, int query_size, BP bp);
    // void record(int qid, int* new_knns, int batch, cudaStream_t& stream);
    inline int* get(int qid = 0) {
#ifdef GPU_TEST
        assert(qid < size);
#endif
        return knns.data().get() + qid * bp.k;
    }
    inline thrust::device_vector<int>::iterator get_iter(int qid = 0) {
        return knns.begin() + qid * bp.k;
    }

   private:
    thrust::device_vector<int> knns;
    int size;
    BP bp;
};

class QueryIndex {
   public:
    QueryIndex(const std::string& filename, int K, int k, int dim, DIST_METRIC metric);
    ~QueryIndex() { delete query_knns; };
    void Top1_Search(const float* h_queries, int* d_closest_ids_out, int num,
                     cudaStream_t& stream);
    Query_KNNs& get_knns() { return *query_knns; }

   private:
    int query_index_size, K;
    Query_KNNs* query_knns;
    BP bp;

    thrust::device_vector<float> d_bucket_vecs, d_centroids;
    thrust::device_vector<int> d_bucket_vec_ids, d_cluster_offsets;
};

class GPUFuncs {
   public:
    GPUFuncs(Graph* graph, const float* h_base, float* h_queries, int base_n, int query_n, int dim,
             int k, int max_degree, DIST_METRIC m, int beam_capacity);
    GPUFuncs(Graph* graph, QueryIndex* qindex, int base_n, int dim, int k, int max_degree,
             DIST_METRIC m, int beam_capacity);
    ~GPUFuncs();
    inline cudaStream_t& get_stream(std::string name) { return streams[name].stream; }
    void query_data_kmeans(float* h_queries, int K = KMEANS_CENTROID_NUM);
    void compute_knn_dist_test(const float* h_test_query, float* h_res_dists, int* h_res_idxs);
    void topk_test(const float* h_dists, const int* h_idxs, int* h_res_topk_idxs,
                   float* h_res_topk_dists);
    void cand_compute_test(const int* d_knn_idxs, CN* h_cand_nbrs_res);
    int* knn_compute(const float* h_queries, int start_id, int batch, float& time_s);
    int* knn_compute_nosort(const float* h_queries, int start_id, int batch, float& time_s);
    // int* knn_compute(int batch, int start_id, float& time_ms, int n);
    // int* knn_compute(const float* h_queries, int batch, float& time_ms);

    float handle_knn_updates(const int* d_knn_ids, int batch_id, int batch);
    // int SaveGraphData(const int* vec_ids, int batch);
    // int LoadNeededData(const int* d_knn_ids, std::vector<int>& h_graph,
    //                    std::vector<int>& h_graph_deg, std::vector<float>& h_graph_dist, int
    //                    batch);
    void neibor_aware_test(const std::vector<int>& h_knn_ids, NBD& h_upd_node_data);

    Test_Result test_search_and_verify(int test_query_idx, int batch,
                                       const std::vector<int>& start_node_ids);

    Test_Result beam_search_for_insert(float* insert_vectors, int* d_knn_res, int batch,
                                       const std::string& stream_name);
    // inline GDD get_graph() { return GDD(graph_mtx); }
    void test_query_data_prepare(const float* data, const int* knns);
    void beam_search_test();

    void vector_insert(float* insert_vector, int batch);
    void vector_delete(std::vector<int>& del_ids);
    void neibor_aware(const int* d_vec_ids, NBD* d_node_data, int batch, int vec_num,
                      cudaStream_t& stream);

    void record_time(const std::string func, float time_s);
    void print_times();
    std::string avg_time_serialize() {
        std::string str = std::to_string(avg_times.size()) + ",";
        for (auto& it : avg_times) {
            str += it.first + "," + it.second.serialize() + ",";
        }
        return str;
    }
    void avg_time_deserialize(const std::string& str) {
        // fo.print("avg_time_deserialize str:\n" + str);

        std::istringstream iss(str);
        int size;
        std::string temp;
        std::getline(iss, temp, ',');
        size = std::stoi(temp);
        for (int i = 0; i < size; i++) {
            std::string time_str, n_str;
            std::getline(iss, temp, ',');
            std::getline(iss, time_str, ',');
            std::getline(iss, n_str, ',');
            avg_times[temp] = Avg_Time(time_str, n_str);
        }

        print_times();  // 打印反序列化后的结果
    }

   private:
    bool construct;
    QueryIndex* query_index;

    Graph* graph;
    BP bp;
    const float* h_base;

    // int *graph, *graph_deg;
    // float *graph_dist;
    thrust::device_vector<float> d_b;  // only for knn compute
    // thrust::device_vector<int> from_edge_nums;

    // std::atomic<int> base_n_upd;

    // int* base_query_ids;      // 记录每个 base 节点的以其为最近邻的查询模态节点索引
    // float* base_query_dists;  // 记录每个 base 节点到以其为最近邻的查询模态节点的距离

    std::shared_mutex search_mtx, base_query_mtx;
    // GSD *new_gs, gs;

    float* d_q;
    // float *d_center, *d_dists,*closest_val;
    // int *graph_ep;

    // knn 相关：
    bool round_copy = true;
    thrust::device_vector<int> d_knn_res;
    thrust::device_vector<float> d_knn_dists_res;
    int* d_knn_1b;          // 存储 knn_compute 每次的结果（一个 batch 的 KNN idxs）
    float* d_knn_dists_1b;  // 存储 knn_compute 每次的结果（一个 batch 的 KNN dists）
    int *partial_ids_a, *partial_ids_b, *last_topk_ids;
    float *partial_dists_a, *partial_dists_b, *last_topk_dists;
    int knn_tpb = 256, items_per_thread = 128, blocks_per_batch;
    faiss::gpu::StandardGpuResources gpu_res;
    faiss::gpu::GpuIndexFlatL2 knn_index;

    // upd 相关：
    // int* d_new_nbr_ids;      // handle_knn_updates 的结果
    // float* d_new_nbr_dists;  // handle_knn_updates 的结果
    // int *new_vec_ids, *finded_idxs, *h_finded_idxs, *h_knn_ids;
    NBD *d_node_data_buffer, *d_del_node_data_buffer, *h_node_data, *h_del_node_data;
    thrust::device_vector<CN> cand_nbrs;
    thrust::device_vector<int> cand_idxs, cand_idxs_sort, offsets, nbr_nums;
    thrust::device_vector<float> cand_dists, cand_dists_sort;
    int tile_k;

    // search 相关：
    // int *d_start_node_ids, *d_hops, *h_hops, *d_expand_num, *h_expand_num, *d_gt_knns, *d_hits,
    //     *h_hits;
    // NBD *d_beam_node_data, *h_beam_node_data;
    // int *beam_visited,* beam_ids;
    // std::vector<int> h_beam_ids;
    // float *beam_dists;
    const float* test_query_data;
    const int* test_query_knns;
    thrust::device_vector<BCD> d_beams, d_beams_sort;
    thrust::device_vector<float> d_beam_dists, d_beam_dists_sort;
    thrust::device_vector<int> d_beam_ids, d_beam_offsets;
    thrust::device_vector<float> d_queries;
    NBD *beam_node_data, *d_beam_node_data;
    std::vector<int> beam_ids;

    // insert 相关：
    std::vector<int> h_ins_ids, h_k_plus_ones;
    thrust::device_vector<int> d_ins_ids, d_paired_query_ids, d_k_plus_ones;
    thrust::device_vector<float> d_insert_vectors;
    NBD *d_k_plus_one_node_data, *h_k_plus_one_node_data;

    const std::string stream_names[3] = {"knn", "upd", "search"};
    std::unordered_map<std::string, MyStream> streams;

    int blocknum_per_query = 4;
    int knn_threads = 1024;
    int upd_threads = 64;

    size_t shared_mem_per_block;

    // knn_compute 预分配内存指针
    float *d_all_dist, *d_all_dist_sort;  // 存储所有查询的距离与索引结果，用于之后排序
    int *d_all_idx, *d_all_idx_sort, *d_topk_offsets;

    // handle_knn_updates 预分配内存指针
    // Candidate_Neighbor
    //     *d_cand_nbrs;  // 每个批次的所有 KNN 中每个 pivot 的候选邻居向量信息，大小：batch * k *
    //     k
    // float *d_knn_dists, *d_knn_dists_sort;
    // int *d_idxs, *d_idxs_sort;
    // int *d_offsets;
    // int *sort_start, sort_end;
    // int *d_nbr_num;  // 记录每个 pivot 的邻居数量，大小：batch * k

    std::unordered_map<std::string, Avg_Time> avg_times;

    void event_record_time_start(const std::string name);
    float event_record_time_stop(const std::string name, std::string msg);
    void knn_prepare(const float* h_queries);
    void knn_free();
    void upd_prepare(bool construct);
    void upd_free();
    void search_prepare();
    void search_free();
    void update_prepare();
    void update_free();

    void graph_resize(int new_size);

    int get_free_bytes();

    int beam_search(const std::vector<int>& start_node_ids, const float* h_queries, int batch,
                    cudaStream_t& stream);

    // inline int beam_search(int test_query_idx, const std::vector<int>& start_node_ids, BCD*
    // beam,
    //                        cudaStream_t& stream) {
    //     return beam_search(start_node_ids, test_query_data + test_query_idx * bp.dim, beam,
    //                        stream);
    // }

    int beam_search_verify(const int* h_gt_knns, int batch, cudaStream_t& stream);
    // inline int beam_search_verify(int test_query_idx, const BCD* beam, cudaStream_t& stream) {
    //     return beam_search_verify(test_query_knns + test_query_idx * bp.k, beam, stream);
    // }
};

void idx_dist_extract(CN* nbrs, int* idxs, float* dists, cudaStream_t& stream, int num, BP bp);

__global__ void candidate_ignore_kernel(CN* cand_nbrs, int* sort_idxs, int* nbr_num,
                                        const int new_nbr_i, const int pivot_num, int cand_size,
                                        BP bp);

__global__ void pair_dist_compute_kernel(const int* __restrict__ vec_ids,
                                         const NBD* __restrict__ node_data,
                                         CN* __restrict__ cand_nbrs, int tile_k, int vec_num,
                                         int batch, BP bp);

__global__ void update_new_nbrs_kernel(const int* __restrict__ sort_idxs,
                                       NBD* __restrict__ node_data, CN* __restrict__ cand_nbrs,
                                       int pivot_num, int cand_size, BP bp);

// 根据 candidate_ignore_kernel 的结果（d_cand_nbrs）,
// 提取每个批次的 KNN 中每个 pivot 的邻居 base_id,
// 结果存在 new_nbrs 中，大小：batch * k * max_degree
__global__ void get_new_nbrs_kernel(CN* cand_nbrs, const int* sort_idxs, int* new_nbr_ids,
                                    float* new_nbr_dists, int cand_size, BP bp);

__global__ void compute_knn_dist_kernel(const float* __restrict__ query,
                                        const float* __restrict__ bases, float* dists, int* idxs,
                                        BP bp, int base_n, int start_base_id);

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