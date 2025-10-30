#pragma once
#include <cuda_runtime.h>
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
    Query_KNNs(int size, BP bp);
    void record(int qid, int* new_knns, int batch, cudaStream_t& stream);
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

class GPUFuncs {
   public:
    GPUFuncs(const float* h_base, float* h_queries, int base_n, int query_n, int dim, int k,
             int max_degree, DIST_METRIC m, int beam_capacity);
    ~GPUFuncs();
    void query_data_kmeans(float* h_queries, int K = KMEANS_CENTROID_NUM);
    void compute_knn_dist_test(const float* h_test_query, float* h_res_dists, int* h_res_idxs);
    void topk_test(const float* h_dists, const int* h_idxs, int* h_res_topk_idxs,
                   float* h_res_topk_dists);
    void cand_compute_test(const int* d_knn_idxs, CN* h_cand_nbrs_res);
    int* knn_compute(int batch, int start_id, float& time_ms);
    int* knn_compute(const float* h_queries, int batch, float& time_ms);
#ifdef GIG
    GSD handle_knn_updates(const int* d_knn_idxs, const int batch);
    void cand_ignore_test(const int* h_knn_idxs, CN* h_cand_nbrs, int* h_graph_res,
                          int* h_graph_deg_res, float* h_graph_dist_res);
#else
    std::pair<int*, float*> handle_knn_updates(const int* d_knn_idxs, const int* d_old_nbrs,
                                               const float* d_old_nbr_dists,
                                               const int* d_old_nbr_nums, const int batch);
    void cand_ignore_test(CN* h_cand_nbrs, int* h_new_nbr_ids_res, float* h_new_nbr_dists_res);
#endif
    Test_Result beam_search(float* queries, int batch, bool test = false);
    inline Test_Result beam_search(int test_query_idx, int batch) {
        return beam_search(test_query_data + test_query_idx * bp.dim, batch, true);
    }
    void beam_search_verify(int* knns, Test_Result& res, int batch, bool test = false);
    inline void beam_search_verify(int test_query_idx, Test_Result& res, int batch) {
        beam_search_verify(test_query_knns + test_query_idx * bp.k, res, batch, true);
    }
    inline Test_Result beam_search_and_verify(int test_query_idx, int batch) {
        std::unique_lock<std::shared_mutex> lock(beam_mtx);
        Test_Result res = beam_search(test_query_idx, batch);
        beam_search_verify(test_query_idx, res, batch);
        return res;
    }
    Test_Result beam_search_for_insert(float* insert_vectors, int* d_knn_res, int batch,
                                       cudaStream_t& stream);
    inline GDD get_graph() {
        return GDD(graph.data().get(), graph_deg.data().get(), graph_dist.data().get(), graph_ep,
                   graph_mtx);
    }
    void test_query_data_prepare(const float* data, const int* knns, int query_n);
    void beam_search_test();

    void vector_insert(float* insert_vector, int batch);
    void vector_delete(int* del_ids, int batch);
    void neibor_aware(const int* vec_ids, int batch, int vec_num, int base_n, cudaStream_t& stream,
                      int* from_edge_nums = nullptr);

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
    BP bp;

    // int *graph, *graph_deg;
    // float *graph_dist;
    thrust::device_vector<int> graph, graph_deg;
    thrust::device_vector<float> graph_dist;
    thrust::device_vector<float> d_b;
    thrust::device_vector<int> from_edge_nums;

    std::atomic<int> base_n_upd;

    int* base_query_ids;      // 记录每个 base 节点的以其为最近邻的查询模态节点索引
    float* base_query_dists;  // 记录每个 base 节点到以其为最近邻的查询模态节点的距离
    Query_KNNs query_knns;    // 记录每个查询模态节点的 knns，初始先记录 0.1M 个，后续可以扩大

    std::shared_mutex graph_mtx, beam_mtx, base_query_mtx;
    GSD *new_gs, gs;

    int *graph_ep, *beam_offsets, *beam_visited, *beam_converge_num, *test_query_knns;
    float *closest_val, *beam_dists, *test_query_data;
    BCD* beam;
    bool* beam_converge;

    float *d_q, *d_center, *d_dists;

    int* d_knn_res;          // 存储 knn_compute 每次的结果（一个 batch 的 KNN idxs）
    int* d_new_nbr_ids;      // handle_knn_updates 的结果
    float* d_new_nbr_dists;  // handle_knn_updates 的结果

    const std::string stream_names[5] = {"knn", "upd", "search", "ins_del"};
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

    // sample.cu:

    void event_record_time_start(const std::string name);
    float event_record_time_stop(const std::string name, std::string msg);
    void knn_prepare(const float* h_base, const float* h_queries);
    void knn_free();
    void upd_prepare();
    void upd_free();
    void search_prepare();
    void search_free();
    void update_prepare();
    void update_free();

    void graph_resize(int new_size);

    int get_free_bytes();
};

void idx_dist_extract(CN* nbrs, int* idxs, float* dists, cudaStream_t& stream, int num, BP bp);

__global__ void candidate_ignore_kernel(
    const float* __restrict__ base,
    // 每个批次的所有 KNN 中每个 pivot 的候选邻居向量到 pivot 的距离等信息，大小：batch * k *
    // (k+max_degree)
    CN* cand_nbrs,
    // 每个批次的 KKN 中向量每个 pivot 的候选邻居向量根据到 pivot 的距离排序后的原索引
    // 大小：batch * k * (cand_size)，范围：0 ~ (cand_size - 1)
    int* sort_idxs,
    int* nbr_num,         // 记录当前每个 pivot 的邻居数量，大小：batch * k
    const int new_nbr_i,  // 1 <= new_nbr_i <= cand_size - 1
    const int batch, int cand_size, int vec_num, BP bp);

#ifdef GIG
__global__ void pair_dist_compute_kernel(const float* __restrict__ base,
                                         const int* __restrict__ vec_ids, const int* graph,
                                         const float* graph_dist, const int* graph_deg,
                                         CN* cand_nbrs, int tile_k, int vec_num, int batch, BP bp);

__global__ void update_new_nbrs_kernel(const int* __restrict__ vec_ids,
                                       const int* __restrict__ sort_idxs, int* __restrict__ graph,
                                       int* __restrict__ graph_deg, float* __restrict__ graph_dist,
                                       CN* __restrict__ cand_nbrs, int vec_num, int batch, BP bp);
#else
/*
计算每组 KNN 中所有基础向量之间的距离，对于一个 knn 中的 k 个基础节点：
- 如果节点 i 无 old nbrs（孤立节点），则保留其他所有 k-1 个基础节点为 i 的候选邻居
- 如果节点 j 有 old nbrs（非孤立节点），则仅保留那些孤立的基础节点作为候选邻居（相当于加反边）
*/
__global__ void knn_dist_compute_kernel(
    const float* __restrict__ base,
    const int* __restrict__ knns,  // 查询的 knn 结果 batch * k
    const int* d_old_nbrs, const float* d_old_nbr_dists, const int* d_old_nbr_nums,
    CN* cand_nbrs,  // 存储结果：K 个向量两两之间距离（batch*K*(K + max_degree) 矩阵）
    int tile_k, BP bp);
#endif
// 根据 candidate_ignore_kernel 的结果（d_cand_nbrs）,
// 提取每个批次的 KNN 中每个 pivot 的邻居 base_id,
// 结果存在 new_nbrs 中，大小：batch * k * max_degree
__global__ void get_new_nbrs_kernel(CN* cand_nbrs, const int* sort_idxs, int* new_nbr_ids,
                                    float* new_nbr_dists, int cand_size, BP bp);

__global__ void compute_knn_dist_kernel(const float* __restrict__ d_q,  // 查询向量
                                        const float* __restrict__ d_b,  // 基础向量库
                                        float* dists,  // 输出：所有距离（每个查询占 base_n 个）
                                        int* idxs,     // 输出：所有索引（与距离一一对应）
                                        int batch, BP bp);

__global__ void topk_copy_kernel(int* d_knn_res, const int* d_all_idx, float* d_dist_res,
                                 const float* d_all_dist, int batch, BP bp);

__device__ __forceinline__ CN* get_CN(CN* cand_nbrs, const int* sort_idxs, const int i,
                                      const int group_id, const int pivot_idx, int k,
                                      int cand_size) {
    // 排序后第 i 个邻居向量在原始未排序 kNNs 中的索引
    return cand_nbrs + group_id * k * cand_size + pivot_idx * cand_size +
           sort_idxs[group_id * k * cand_size + pivot_idx * cand_size + i];
}
#ifdef algo0
__global__ void pivot_to_others_dist_compute_kernel(
    const float* __restrict__ base,
    const int* __restrict__ knns,  // 查询的 knn 结果 batch * k
    CN* cand_nbrs,                 // 存储结果：pivot 和 k - 1 个向量间距离
    BP bp);
__global__ void add_reverse_kernel(
    CN* cand_nbrs,  // K 个向量两两之间距离（batch*K*(cand_size) 矩阵），待补充旧邻居
    const int* d_old_nbrs, const float* d_old_nbr_dists, const int* d_old_nbr_nums, BP bp);
#endif
}  // namespace efanna2e

#endif  // KNN_CUH