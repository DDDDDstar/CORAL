#ifndef GT_CACHE_H
#define GT_CACHE_H

#include <cuda_runtime.h>

#include <cstdint>
#include <shared_mutex>
#include <string>
#include <vector>

#include "uni.h"

namespace efanna2e {

class Cache {
   public:
    std::vector<BatchResult> results;

    Cache(const std::string cache_file, const std::string graph_file, int& last_batch_id,
          const int k, const int n);
    Cache(const std::string cache_file, int& last_batch_id, const int k, const int n);

    ~Cache();
    int size() { return results.size(); }
    inline int get_query_num() { return total_query_num; }
    void WriteGTCache(const int* d_knns, const int* qids, const int batch_id, const int len,
                      const double time_ms);
    void WriteGTCacheHost(const int* knns, int batch_id, int batch);
    void SaveGraphCache(const std::vector<std::vector<int>>& graph,
                        const std::vector<std::vector<float>>& graph_dist,
                        const std::vector<int>& graph_deg,
                        std::vector<std::shared_mutex>& graph_mtx, const int last_batch_id,
                        const float total_time, const Test_Result tres, const int n);

   private:
    bool use_gt_cache = false;
    const int k;
    const std::string cache_file, graph_file;
    int save_thres, new_num = 0;
    std::future<void> save_future;
    int total_query_num = 0;

    // GPUFuncs* gpufuncs;

    // 从二进制文件加载缓存结果
    void LoadGTCache();
    void SaveGTCache();
    void LoadGraphCache(std::vector<std::vector<int>>& graph, std::vector<int>& graph_deg,
                        std::vector<std::vector<float>>& graph_dist, int& noniso_num,
                        int& last_batch_id, int& total_degree, float& total_time,
                        Test_Result& tres, const int n);
};
}  // namespace efanna2e

#endif  // GT_CACHE_H