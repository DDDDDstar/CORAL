#ifndef HYPERPARA_H
#define HYPERPARA_H

#include <float.h>

#include <cassert>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

#define QUEUE_SIZE 20
#define BATCH 128
#define KMEANS_CENTROID_NUM 512
#define KMEANS_SIZE_RATIO 0.1  // 从所有查询数据中选择用来 kmeans 获得质心的比例
#define KMEANS_MAX_ITERS 100
// #define MAX_K 100
#define TEST_SEARCH_L 1000
#define TEST_SEARCH_QUERY_SIZE (BATCH * 8)
#define BEAM_MAX_MEM_SIZE 5ULL  // GB
#define TEST_BATCH_INTERVAL 200

#define USE_GT_CACHE true
#define USE_GRAPH_CACHE true

#define TEST_BATCH 3

#define IP_CLOSEST INFINITY
#define IP_FARTHEST (-INFINITY)
#define L2_CLOSEST 0.0
#define L2_FARTHEST INFINITY
#define EPS 1e-6f

// #define algo0 "add_reverse"
// #define algo1 "add_all_nbrs"
#define algo2 "add_nbrs_to_non_full_nodes"
#define GIG "graph in gpu"
// #define GPU_TEST "gpu test"

#define LOG_PREFIX "../logs/"
#define RECALL_PREFIX "../logs/recall/"
#define INDEX_PREFIX "../indexes/"
#define GT_PREFIX "../gts/"

#define TOS(x) std::to_string(x)
#define GB(x) (x * (1ULL << 30))

using TIME = std::chrono::_V2::system_clock::time_point;

enum DIST_METRIC { L2, IP };

enum STATUS {
    Initial,      // 初始状态
    Discarded,    // 淘汰状态
    AllRetained,  // 全部保留状态，即候选邻居数量小于 max_degree
    Retained,     // 保留状态
    Repeated      // 节点和前面的节点重复
};

struct Graph_Struct_D {
    int iso_num;       // 孤立点数量
    int total_degree;  // 总度数
};
using GSD = Graph_Struct_D;

class Base_Parameters {
   public:
    int dim, max_degree, base_n, query_n, k, beam_capacity, beam_expand_num, beam_size,
        visited_words_num = BATCH / 32;
    DIST_METRIC metric;

    Base_Parameters() = default;
    Base_Parameters(int dim, int max_degree, int base_n, int query_n, int k, int beam_capacity,
                    DIST_METRIC metric)
        : dim(dim),
          max_degree(max_degree),
          base_n(base_n),
          query_n(query_n),
          k(k),
          beam_capacity(beam_capacity),
          metric(metric) {}
    inline std::string str() {
        return std::string("Base_Parameters: dim=") + TOS(dim) +
               ", max_degree=" + TOS(max_degree) + ", base_n=" + TOS(base_n) +
               ", query_n=" + TOS(query_n) + ", k=" + TOS(k) +
               ", beam_capacity=" + TOS(beam_capacity) +
               ", beam_expand_num=" + TOS(beam_expand_num) + ",beam_size=" + TOS(beam_size) +
               ", metric=" + TOS(metric);
    }
};
using BP = Base_Parameters;

class Candidate_Neighbor {
   public:
    int id;         // 候选邻居向量在基础向量集中的索引
    int idx;        // 候选邻居向量在其所在 KNN 簇中的索引
    STATUS status;  // 候选邻居向量淘汰情况（0 待定，1 保留，2 淘汰，-1 全部保留）
    float dist;     // 候选邻居向量与 pivot 向量的距离

    Candidate_Neighbor &operator=(const Candidate_Neighbor &other) {
        if (this != &other) {
            id = other.id;
            idx = other.idx;
            status = other.status;
            dist = other.dist;
        }
        return *this;
    }
};
using CN = Candidate_Neighbor;

class BatchResult {
   public:
    std::vector<int> knn;
    int batch;
    double time_ms;

    BatchResult() = default;
    BatchResult(int *d_knn, const int k, const int batch, const double time_ms);
    BatchResult(const int k, const int batch);
};

class Test_Result {
   public:
    Test_Result() : recall(0.0), hops(0.0), time(0.0), hits(0) {};
    Test_Result(const float recall, const float hops, const float time)
        : recall(recall), hops(hops), time(time) {}
    Test_Result(const float hops, const float time) : hops(hops), time(time) {}
    inline float get_recall() const { return recall; }
    inline float get_hops() const { return hops; }
    inline float get_time() const { return time; }
    float calc_recall(int total_num) { return recall = 100.0 * hits / total_num; }
    void accumulate(const Test_Result res) {
        hops += res.hops;
        time += res.time;
        hits += res.hits;
    }

    float recall, hops, time;
    int hits;
};

class Graph_Update_Info {
   public:
    Graph_Update_Info() : is_update(false) {}
    inline void update(int batch_id, GSD gs) {
        batch_id = batch_id;
        gs = gs;
        is_update = true;
    }
    inline void reset() { is_update = false; }
    inline bool vaild() { return is_update; }
    inline int get_batch_id() { return batch_id; }
    inline GSD get_GSD() { return gs; }

   private:
    bool is_update;
    int batch_id;
    GSD gs;
};
using GUI = Graph_Update_Info;

void writeCSV(const std::vector<std::vector<std::string>> &data, const std::string &filename,
              const std::ios_base::openmode mode = std::ios::app);

std::string getCurrentDateTimeString();

template <typename T>
inline bool vec_exists(std::vector<T> &vec, const T &val) {
    return std::find(vec.begin(), vec.end(), val) != vec.end();
}

inline float farthest_dist(BP bp) {
    return bp.metric == DIST_METRIC::L2 ? L2_FARTHEST : IP_FARTHEST;
}

int numDigits(int n);  // 计算数字 n 的位数

#endif  // HYPERPARA_H