#include <boost/container/set.hpp>
#include <boost/dynamic_bitset.hpp>
#include <cassert>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

#include "efanna2e/index.h"
#include "efanna2e/neighbor.h"
#include "efanna2e/parameters.h"
#include "efanna2e/util.h"
#include "visited_list_pool.h"

namespace efanna2e {
using LockGuard = std::lock_guard<std::mutex>;
using SharedLockGuard = std::lock_guard<std::shared_mutex>;

class IndexPipeline : public Index {
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
    void LinkProjection();                                
    std::pair<uint32_t, uint32_t> SearchPipe(const float *query, size_t k, size_t &qid, const Parameters &parameters,
                                   unsigned *indices, std::vector<float>& res_dists);
    inline void SetParameters(Parameters &parameters) {parameters_ = parameters;}
    void ProjectionReserveSpace();
    void PruneBiSearchBaseGetBase(std::vector<Neighbor> &candicate_nbrs, uint32_t tgt_base,
                                             std::vector<uint32_t> &pruned_list, int flag);
    void ProjectionAddReverse(uint32_t src_node, std::unordered_map<uint32_t, float>& dis_record);
    void CalculateProjectionep();
    void SaveIndex(const char *filename);

    inline void LoadSearchNeededData(const char *base_file, const char *sampled_query_file) {
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
    const size_t total_pts_const_;
    size_t total_pts_;
    Distance *l2_distance_;
    // boost::dynamic_bitset<> sq_en_flags_;
    // boost::dynamic_bitset<> bp_en_flags_;
    uint32_t width_;
    std::set<uint32_t> sq_en_set_;
    std::set<uint32_t> bp_en_set_;
    std::mutex sq_set_mutex_;
    std::mutex bp_set_mutex_;
    std::vector<std::mutex> locks_;
    uint32_t u32_nd_;
    uint32_t u32_nd_sq_;
    uint32_t u32_total_pts_;
    uint32_t projection_ep_;

    // std::vector<std::atomic<uint32_t>> visit_flags;
    // uint32_t visit_count;
    Parameters parameters_;
};

}  // namespace efanna2e