#include <cuda_runtime.h>

#include <atomic>
#include <boost/container/set.hpp>
#include <boost/dynamic_bitset.hpp>
#include <cassert>
#include <condition_variable>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <stack>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "cache.h"
#include "efanna2e/index.h"
#include "efanna2e/neighbor.h"
#include "efanna2e/parameters.h"
#include "efanna2e/util.h"
#include "fileout.h"
#include "gpufuncs.cuh"
#include "knn_queue.h"
#include "uni.h"
#include "visited_list_pool.h"

namespace efanna2e {

// define likely unlikely
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

struct Graph_Update_Info {
    // int *knn_ids, *new_nbr_ids;
    std::unique_ptr<int[]> knn_ids, new_nbr_ids;
    std::unique_ptr<float[]> new_nbr_dists;
    int batch, batch_id;
    Graph_Update_Info() = default;
    Graph_Update_Info(int *d_knn_idxs, int *d_new_nbr_ids, float *d_new_nbr_dists, const int k,
                      const int batch, const int batch_id, const int max_degree)
        : batch(batch), batch_id(batch_id) {
        knn_ids = std::make_unique<int[]>(batch * k);
        new_nbr_ids = std::make_unique<int[]>(batch * k * max_degree);
        new_nbr_dists = std::make_unique<float[]>(batch * k * max_degree);
        CUDA_CHECK(cudaMemcpy(knn_ids.get(), d_knn_idxs, batch * k * sizeof(int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(new_nbr_ids.get(), d_new_nbr_ids,
                              batch * k * max_degree * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(new_nbr_dists.get(), d_new_nbr_dists,
                              batch * k * max_degree * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < batch * k; ++i) {
            int vaild_num = 0;
            for (int j = 0; j < max_degree; ++j)
                if (new_nbr_ids[i * max_degree + j] >= 0) vaild_num++;

            if (vaild_num == 0)
                fo.eprint("vaild_num == 0 with batch_id = " + std::to_string(batch_id) + " i = " +
                          std::to_string(i) + " pivot_id = " + std::to_string(knn_ids[i]));
        }
    }
};

using LockGuard = std::lock_guard<std::mutex>;
using SharedLockGuard = std::lock_guard<std::shared_mutex>;

class IndexGPU : public Index {
    typedef std::vector<std::vector<int>> CompactGraph;

   public:
    explicit IndexGPU(const size_t dimension, const size_t n, Metric m, Index *initializer);
    virtual ~IndexGPU();
    virtual void Save(const char *filename) override {}
    virtual void Load(const char *filename) override {}
    virtual void Build(size_t n, const float *data, const Parameters &parameters) override;
    virtual void Search(const float *query, const float *x, size_t k, const Parameters &parameters,
                        unsigned *indices, float *res_dists) override;
    void BuildGPU(size_t n_sq, float *sq_data, size_t n_bp, float *bp_data,
                  Parameters &parameters);
    std::pair<int, int> SearchPipe(const float *query, size_t k, size_t &qid,
                                   const Parameters &parameters, int *indices,
                                   std::vector<float> &res_dists);
    inline void SetParameters(Parameters &parameters) { parameters_ = parameters; }
    void SaveIndex(const char *filename);

    inline void LoadSearchNeededData(const char *base_file, const char *sampled_query_file) {
        LoadVectorData(base_file, sampled_query_file);
    }

    void LoadGraph(const char *filename);

    void InitVisitedListPool(int num_threads) {
        visited_list_pool_ = new VisitedListPool(num_threads, nd_);
    };

    Index *initializer_;
    TimeMetric dist_cmp_metric;
    TimeMetric memory_access_metric;
    TimeMetric block_metric;
    VisitedListPool *visited_list_pool_{nullptr};
    bool need_normalize = false;

   protected:
    std::vector<std::vector<int>> bipartite_graph_;
    std::vector<std::vector<int>> final_graph_;
    std::vector<std::vector<int>> projection_graph_;
    std::vector<int> projection_graph_deg_;
    std::vector<std::vector<float>> projection_graph_dist_;
    std::vector<std::vector<int>> supply_nbrs_;
    std::vector<std::vector<int>> learn_base_knn_;
    std::vector<std::vector<int>> base_learn_knn_;

   private:
    DIST_METRIC metric;
    size_t total_pts_;
    int u32_nd_;
    int u32_nd_sq_;
    int projection_ep_;

    Parameters parameters_;

    int k;  // number of neighbors

    float *h_base, *h_queries;
    GPUFuncs *gpufuncs;

    std::shared_mutex queue_mtx;
    std::queue<Graph_Update_Info> graph_update_queue;
    std::condition_variable_any cv;
    std::atomic<bool> stop_flag{false};
    int nonisolated_num = 0, total_degree = 0;

    int processed = 0;
    int batch_id = 0;

    int *h_knn_idxs;
    int *h_old_nbr_nums;

    float total_time = 0.0;
    TIME start_time;

    std::vector<float> test_query_data;
    std::vector<int> test_query_knns;

    GUI gui;

    void GPUPrepare();
    void GPUFree();
    void KNNTask(KNN_Queue &knn_queue, Cache &cache);
    void GraphTask(KNN_Queue &knn_queue, Cache &cache);
    void update_graph(Cache &cache, std::vector<std::shared_mutex> &graph_mtx,
                      KNN_Queue &knn_queue);
    void get_old_nbrs(int *d_knn_idxs, int *d_old_nbrs, float *d_old_nbr_dists,
                      int *d_old_nbr_nums, const int batch, const int max_degree,
                      std::vector<std::shared_mutex> &graph_mtx);
    void LoadVectorData(const char *base_file, const char *sampled_query_file);

    Test_Result test_search();
    void test_search(int *d_graph, int *d_graph_deg, int *d_graph_ep);
    void test_search_prepare(GPUFuncs *gpufuncs);

    float compute_duration_time();
    void CalculateGraphEP();
};

}  // namespace efanna2e