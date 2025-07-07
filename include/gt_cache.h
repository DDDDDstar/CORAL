#ifndef GT_CACHE_H
#define GT_CACHE_H

#include <vector>
#include <string>
#include <cstdint>

#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include <mutex>
#include "efanna2e/distance.h"

namespace efanna2e {
struct Result {
    uint32_t* closest_points;
    float* dist_closest_points;
    double time_ms;
    bool is_new;
    int filter_len;
};

class GTCache {
public:
    GTCache(const std::string cache_file_, 
            const uint32_t k_, 
            const uint32_t u32_nd_, 
            const float *data_sq_, 
            const float *data_bp_,
            const int dimension_,
            const Metric metric_, 
            const uint32_t cache_size = 500);
    ~GTCache();

    // 保存缓存结果到二进制文件
    void SaveCacheOld(std::unordered_map<uint32_t, Result> new_results);

    Result GetGT(const uint32_t it_sq);

private:
    const size_t k;  // 最近邻个数
    const uint32_t u32_nd;
    const int dimension;
    
    std::unordered_map<uint32_t, Result> results;
    mutable std::mutex mtx_;

    const std::string cache_file;
    const float *data_sq;
    const float *data_bp;
    const Metric metric;

    std::unordered_map<uint32_t, Result> new_results;
    std::mutex new_results_lock, results_lock;
    uint32_t cache_size;

    // 从二进制文件加载缓存结果
    void LoadCache();

    void ComputeQueryGT(
        uint32_t it_sq,
        uint32_t *const closest_points,
        float *const dist_closest_points);
    
    void SaveCacheSafely();

    void compute_l2sq(float *const points_l2sq, const float *const matrix,
                  const uint32_t num_points, const int dim);
    template<class T>
    T div_round_up(const T numerator, const T denominator);
    void distsq_to_points(
        const size_t dim,
        float       *dist_matrix,
        size_t npoints, const float *const points,
        const float *const points_l2sq,
        size_t nqueries, const float *const queries,
        const float *const queries_l2sq,
        float *ones_vec = NULL);
    void inner_prod_to_points(
        const size_t dim,
        float       *dist_matrix,  // Col Major, cols are queries, rows are points
        size_t npoints, const float *const points, size_t nqueries,
        const float *const queries,
        float *ones_vec = NULL);
};
}

// void Filter(struct Result &res, int k);

#endif // GT_CACHE_H