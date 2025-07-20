#include <boost/container/set.hpp>
#include <boost/dynamic_bitset.hpp>
#include <cassert>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <stack>
#include <string>
#include <queue>
#include <cstring>
#include <future>
#include <atomic>
#include <thread>
#include <unordered_map>
#include <vector>
#include <condition_variable>
#include <cuda_runtime.h>
#include <memory>

#include "efanna2e/index.h"
#include "efanna2e/neighbor.h"
#include "efanna2e/parameters.h"
#include "efanna2e/util.h"
#include "visited_list_pool.h"
#include "knn.cuh"
#include "knn_queue.h"
#include "gt_cache.h"

struct Graph_Update_Info
{
    // uint32_t *knn_ids, *new_nbr_ids;
    std::unique_ptr<uint32_t[]> knn_ids, new_nbr_ids;
    int batch;
    Graph_Update_Info() = default;
    Graph_Update_Info(
        uint32_t *d_knn_idxs, uint32_t *d_new_nbr_ids,
        const int k, const int batch, const int max_degree) : batch(batch)
    {
        knn_ids.reset(new uint32_t[batch * k]);
        new_nbr_ids.reset(new uint32_t[batch * k * max_degree]);
        // CUDA_CHECK(cudaMallocHost(&knn_ids, batch * k * sizeof(uint32_t)));
        // CUDA_CHECK(cudaMallocHost(
        //     &new_nbr_ids, batch * k * max_degree * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(
            knn_ids.get(), d_knn_idxs, batch * k * sizeof(uint32_t),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            new_nbr_ids.get(), d_new_nbr_ids,
            batch * k * max_degree * sizeof(uint32_t),
            cudaMemcpyDeviceToHost));
    }
    // // 移动构造函数
    // Graph_Update_Info(Graph_Update_Info &&other) noexcept
    //     : knn_ids(other.knn_ids),
    //       new_nbr_ids(other.new_nbr_ids),
    //       batch(other.batch)
    // {
    //     other.knn_ids = nullptr;     // 置空源对象指针
    //     other.new_nbr_ids = nullptr; // 避免析构时释放
    // }
    // // 移动赋值运算符
    // Graph_Update_Info &operator=(Graph_Update_Info &&other) noexcept
    // {
    //     if (this != &other)
    //     {
    //         // 释放当前对象的资源
    //         CUDA_CHECK(cudaFreeHost(knn_ids));
    //         CUDA_CHECK(cudaFreeHost(new_nbr_ids));

    //         // 转移所有权
    //         knn_ids = other.knn_ids;
    //         new_nbr_ids = other.new_nbr_ids;
    //         batch = other.batch;

    //         // 置空源对象指针
    //         other.knn_ids = nullptr;
    //         other.new_nbr_ids = nullptr;
    //     }
    //     return *this;
    // }
    // ~Graph_Update_Info()
    // {
    //     if (knn_ids)
    //         CUDA_CHECK(cudaFreeHost(knn_ids));
    //     if (new_nbr_ids)
    //         CUDA_CHECK(cudaFreeHost(new_nbr_ids));
    // }
};

namespace efanna2e
{
    using LockGuard = std::lock_guard<std::mutex>;
    using SharedLockGuard = std::lock_guard<std::shared_mutex>;

    class IndexPipeline : public Index
    {
        typedef std::vector<std::vector<uint32_t>> CompactGraph;

    public:
        explicit IndexPipeline(const size_t dimension, const size_t n, Metric m, Index *initializer);
        virtual ~IndexPipeline();
        virtual void Save(const char *filename) override;
        virtual void Load(const char *filename) override;
        virtual void Build(size_t n, const float *data, const Parameters &parameters) override;
        virtual void Search(const float *query, const float *x, size_t k, const Parameters &parameters,
                            unsigned *indices, float *res_dists) override;
        // virtual void BuildPipeline(size_t n_sq, const float *sq_data, size_t n_bp, const float *bp_data,
        //                             const Parameters &parameters) override;
        void BuildPipeline(size_t n_sq, float *sq_data, size_t n_bp, float *bp_data, Parameters &parameters);
        std::pair<uint32_t, uint32_t> SearchPipe(const float *query, size_t k, size_t &qid, const Parameters &parameters,
                                                 unsigned *indices, std::vector<float> &res_dists);
        inline void SetParameters(Parameters &parameters) { parameters_ = parameters; }
        void ProjectionReserveSpace();
        void CalculateProjectionep();
        void SaveIndex(const char *filename);

        inline void LoadSearchNeededData(const char *base_file, const char *sampled_query_file)
        {
            LoadVectorData(base_file, sampled_query_file);
        }
        void LoadVectorData(const char *base_file, const char *sampled_query_file);

        void LoadGraph(const char *filename);

        void InitVisitedListPool(uint32_t num_threads) { visited_list_pool_ = new VisitedListPool(num_threads, nd_); };

        Index *initializer_;
        TimeMetric dist_cmp_metric;
        TimeMetric memory_access_metric;
        TimeMetric block_metric;
        VisitedListPool *visited_list_pool_{nullptr};
        bool need_normalize = false;

    protected:
        std::vector<std::vector<uint32_t>> bipartite_graph_;
        std::vector<std::vector<uint32_t>> final_graph_;
        std::vector<std::vector<uint32_t>> projection_graph_;
        std::vector<std::vector<uint32_t>> supply_nbrs_;
        std::vector<std::vector<uint32_t>> learn_base_knn_;
        std::vector<std::vector<uint32_t>> base_learn_knn_;

    private:
        size_t total_pts_;
        uint32_t u32_nd_;
        uint32_t u32_nd_sq_;
        uint32_t projection_ep_;

        // std::vector<std::atomic<uint32_t>> visit_flags;
        // uint32_t visit_count;
        Parameters parameters_;

        int k; // number of neighbors

        float *h_base, *h_queries;
        GPUFuncs *gpufuncs;

        std::shared_mutex mtx;
        std::queue<Graph_Update_Info> graph_update_queue;
        std::condition_variable_any cv;
        std::atomic<bool> stop_flag{false};

        void GPUPrepare();
        void GPUFree();
        void KNNTask(KNN_Queue &knn_queue, GTCache &cache);
        void GraphTask(KNN_Queue &knn_queue);
        void update_graph();
    };

} // namespace efanna2e