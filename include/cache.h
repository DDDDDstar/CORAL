#ifndef GT_CACHE_H
#define GT_CACHE_H

#include <cuda_runtime.h>

#include <cstdint>
#include <future>
#include <shared_mutex>
#include <string>
#include <vector>

#include "fileout.h"
#include "gpufuncs.cuh"
#include "uni.h"

namespace efanna2e {

class Cache {
   public:
    std::vector<BatchResult> results;

    Cache(const std::string cache_file, const std::string graph_file, GPUFuncs* gpufuncs,
          std::vector<std::vector<int>>& graph, std::vector<int>& graph_deg,
          std::vector<std::vector<float>>& graph_dist, int& noniso_num, int& last_batch_id,
          int& total_degree, float& total_time, Test_Result& tres, const int k, const int n);
    ~Cache();
    int size() { return results.size(); }
    BatchResult WriteGTCache(int* d_knn, const int batch_id, const int len, const double time_ms);
    void SaveGraphCache(const std::vector<std::vector<int>>& graph,
                        const std::vector<std::vector<float>>& graph_dist,
                        const std::vector<int>& graph_deg,
                        std::vector<std::shared_mutex>& graph_mtx, const int last_batch_id,
                        const float total_time, const Test_Result tres, const int n);

   private:
    const int k;
    const std::string cache_file, graph_file;
    int save_thres, new_num = 0;
    std::future<void> save_future;

    GPUFuncs* gpufuncs;

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