#include "gt_cache.h"
#include <fstream>
#include <sstream>
#include <chrono>
#include <iostream>
#include <cblas.h>
#include <queue>
#include <algorithm>
#include <cassert>
#include <unordered_map>
#include <unistd.h>     // for fsync
#include <cstdio>       // for std::rename
#include <fcntl.h>      // for open/close
#include <cmath>

using pairIF = std::pair<int, float>;
struct cmpmaxstruct {
  bool operator()(const pairIF &l, const pairIF &r) {
    return l.second < r.second;
  };
};
using maxPQIFCS =
    std::priority_queue<pairIF, std::vector<pairIF>, cmpmaxstruct>;

namespace efanna2e {
GTCache::GTCache(const std::string cache_file_, 
            const uint32_t k_, 
            const uint32_t u32_nd_, 
            const float *data_sq_, 
            const float *data_bp_,
            const int dimension_,
            const Metric metric_, 
            const uint32_t cache_size)
    : cache_file(cache_file_), k(k_), 
      u32_nd(u32_nd_), data_sq(data_sq_), data_bp(data_bp_),
      dimension(dimension_), metric(metric_), cache_size(cache_size) {
    LoadCache();
    new_results.reserve(cache_size);
}

GTCache::~GTCache() {
    for (const auto& pair : results) {
        Result res = pair.second;
        delete[] res.closest_points;
        delete[] res.dist_closest_points;
    }
}

void GTCache::LoadCache() {
    std::ifstream file(cache_file, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Cache file not exist: " << cache_file << std::endl;
        return;
    }

    size_t n_vectors = 0;
    file.read(reinterpret_cast<char*>(&n_vectors), sizeof(size_t));
    std::cout << "Load GT cache from " << cache_file << ", size: " << n_vectors << std::endl;

    uint32_t print_flag = 9;
    for (size_t i = 0; i < n_vectors; ++i) {
        uint32_t it;
        uint32_t* closest_points = new uint32_t[k];
        float* dist_closest_points = new float[k];
        double time;
        if (!file.read(reinterpret_cast<char*>(&it), sizeof(uint32_t))) {
            std::cout << "Read it failed." << std::endl;
            std::cout << "result size: " << results.size() << std::endl;
            return;
        }
        if (!file.read(reinterpret_cast<char*>(closest_points), sizeof(uint32_t) * k)) {
            std::cout << "Read closest_points failed." << std::endl;
            std::cout << "result size: " << results.size() << std::endl;
            return;
        }
        for (uint32_t j = 0; j < k; j++) {
            uint32_t id = closest_points[j];
            if (id >= u32_nd) {
                std::cout << "closest_point out of range: " << id << std::endl;
                std::cout << "result size: " << results.size() << std::endl;
                return;
            }
        }
        if (!file.read(reinterpret_cast<char*>(dist_closest_points), sizeof(float) * k)) {
            std::cout << "Read dist_closest_points failed." << std::endl;
            std::cout << "result size: " << results.size() << std::endl;
            return;
        }
        if (!file.read(reinterpret_cast<char*>(&time), sizeof(double))) {
            std::cout << "Read time failed." << std::endl;
            std::cout << "result size: " << results.size() << std::endl;
            return;
        }

        struct Result res;
        res.closest_points = closest_points;
        res.dist_closest_points = dist_closest_points;
        res.time_ms = time;
        res.is_new = false;
        results[it] = res;

        double finish_percent = i * 100.0 / n_vectors;
        if (finish_percent > print_flag) {
            std::cout << "finish: " << finish_percent << "%" << std::endl;
            print_flag += 10;
        }
    }
    std::cout << "Load GT cache finished." << std::endl;
}

void GTCache::SaveCacheOld(std::unordered_map<uint32_t, Result> new_results) {
    std::ofstream file(cache_file, std::ios::binary);
    if (!file) return;
    results.insert(new_results.begin(), new_results.end());
    size_t n_vectors = results.size();
    file.write(reinterpret_cast<const char*>(&n_vectors), sizeof(size_t));
    
    for (const auto& pair : results) {
        uint32_t it = pair.first;
        Result res = pair.second;
        file.write(reinterpret_cast<const char*>(&it), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(res.closest_points), sizeof(uint32_t) * k);
        file.write(reinterpret_cast<const char*>(res.dist_closest_points), sizeof(float) * k);
        file.write(reinterpret_cast<const char*>(&res.time_ms), sizeof(double));
    }
}

void GTCache::SaveCacheSafely() {
    // 合并新数据
    std::unordered_map<uint32_t, Result> new_results_;
    {
        std::lock_guard<std::mutex> guard(new_results_lock);
        if (!new_results.size()) return;
        new_results_ = std::move(new_results);
        new_results.clear();
    }
    std::lock_guard<std::mutex> guard(results_lock);
    results.insert(new_results_.begin(), new_results_.end());
    // 1. 生成临时文件名
    std::string tmp_file = cache_file + ".tmp";
    // 2. 使用底层文件描述符确保 fsync
    int fd = open(tmp_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd == -1) {
        std::cerr << "Failed to create temporary file: " << tmp_file << std::endl;
        return;
    }
    // 3. 将数据写入临时文件
    FILE* file = fdopen(fd, "wb");
    if (!file) {
        close(fd);
        std::cerr << "Failed to open file stream" << std::endl;
        return;
    }
    
    size_t n_vectors = results.size();
    // 序列化数据
    fwrite(&n_vectors, sizeof(size_t), 1, file);
    // double avgk = 0;
    for (const auto& pair : results) {
        uint32_t it = pair.first;
        const Result& res = pair.second;
        fwrite(&it, sizeof(uint32_t), 1, file);
        fwrite(res.closest_points, sizeof(uint32_t), k, file);
        fwrite(res.dist_closest_points, sizeof(float), k, file);
        fwrite(&res.time_ms, sizeof(double), 1, file);
        // avgk += res.filter_len;
    }
    // avgk /= results.size();


    // 4. 强制数据落盘
    fflush(file);
    fsync(fd);          // 确保数据写入物理磁盘
    fclose(file);       // 自动关闭 fd

    // 5. 原子重命名临时文件
    if (std::rename(tmp_file.c_str(), cache_file.c_str()) != 0) {
        std::cerr << "Failed to rename temporary file to " << cache_file << std::endl;
        std::remove(tmp_file.c_str());  // 清理临时文件
    } else {
        std::cout << "gt cache saved, size: " << n_vectors << std::endl;
        // std::cout << "gt cache saved, size: " << n_vectors << ", avg k: " << avgk << std::endl;
    }
}

Result GTCache::GetGT(const uint32_t it_sq) {
    if (results.find(it_sq) != results.end()) {
        results[it_sq].is_new = false;
        // if (results[it_sq].filter_len == 0) Filter(results[it_sq], k);
        return results[it_sq];
    }

    struct Result res;
    res.closest_points = new uint32_t[k];
    res.dist_closest_points = new float[k];

    auto start = std::chrono::high_resolution_clock::now();
    ComputeQueryGT(it_sq, res.closest_points, res.dist_closest_points);
    auto end = std::chrono::high_resolution_clock::now();

    res.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    res.is_new = true;
    // Filter(res, k);
    {
        std::lock_guard<std::mutex> guard(new_results_lock);
        new_results[it_sq] = res;
    }
    if (new_results.size() >= cache_size) SaveCacheSafely();
    
    return res;
}

void GTCache::ComputeQueryGT(
    uint32_t it_sq,
    uint32_t *const closest_points,
    float *const dist_closest_points)
{
    float *points_l2sq = new float[u32_nd];
    float *queries_l2sq = new float[1];
    compute_l2sq(points_l2sq, data_bp, u32_nd, dimension);
    compute_l2sq(queries_l2sq, data_sq + it_sq * dimension, 1, dimension);

    const float *points = data_bp;
    const float *queries = data_sq + it_sq * dimension;

//     if (metric == Metric::COSINE) {  // we convert cosine distance as
//                                             // normalized L2 distnace
//         points = new float[u32_nd * dimension];
//         queries = new float[dimension];
// #pragma omp parallel for schedule(static, 4096)
//         for (uint32_t i = 0; i < u32_nd; i++) {
//             float norm = std::sqrt(points_l2sq[i]);
//             if (norm == 0) {
//                 norm = std::numeric_limits<float>::epsilon();
//             }
//             for (uint32_t j = 0; j < dimension; j++) {
//                 points[i * dimension + j] = data_bp[i * dimension + j] / norm;
//             }
//         }
//         float norm = std::sqrt(queries_l2sq[0]);
//         if (norm == 0) {
//             norm = std::numeric_limits<float>::epsilon();
//         }
//         for (uint32_t j = 0; j < dimension; j++) {
//             queries[j] = data_sq[it_sq * dimension + j] / norm;
//         }
//         // recalculate norms after normalizing, they should all be one.
//         compute_l2sq(points_l2sq, points, u32_nd, dimension);
//         compute_l2sq(queries_l2sq, queries, 1, dimension);
//     }

//   std::cout << "Going to compute " << k << " NNs for " << it_sq
//             << " queries over " << u32_nd << " points in " << dimension
//             << " dimensions using";
    // if (metric == Metric::INNER_PRODUCT)
    //     std::cout << " MIPS ";
    // else if (metric == Metric::COSINE)
    //     std::cout << " Cosine ";
    // else if (metric == Metric::L2)
    //     std::cout << " L2 ";
    // else if (metric == Metric::FAST_L2)
    //     std::cout << " FAST_L2 ";
    // else if (metric == Metric::PQ)
    //     std::cout << " PQ ";
    // std::cout << "distance fn. " << std::endl;

    size_t q_batch_size = (1 << 10);
    float *dist_matrix = new float[(size_t) q_batch_size * size_t(u32_nd)];
    // 分配64字节对齐内存
    // float* dist_matrix = (float*)mkl_malloc(
    //     (size_t)q_batch_size * size_t(u32_nd) * sizeof(float), 
    //     64  // 对齐要求
    // );
    // size_t nqueries = 1;

    if (metric == Metric::L2 || metric == Metric::COSINE) {
        distsq_to_points(dimension, dist_matrix, size_t(u32_nd), points, points_l2sq,
                    1, queries, queries_l2sq);
    } else {
        if (dist_matrix == nullptr) {
            std::cerr << "Memory allocation failed!" << std::endl;
            exit(EXIT_FAILURE);
        }
        inner_prod_to_points(dimension, dist_matrix, size_t(u32_nd), points, 1, queries);
    }
    // std::cout << "Computed distances for queries: [" << q_b << "," << q_e << ")"
    //         << std::endl;

    maxPQIFCS point_dist;
    for (uint32_t p = 0; p < k; p++)
        point_dist.emplace(
            p, dist_matrix[(ptrdiff_t) p]);
    for (uint32_t p = k; p < u32_nd; p++) {
        if (point_dist.top().second >
            dist_matrix[(ptrdiff_t) p])
            point_dist.emplace(
                p, dist_matrix[(ptrdiff_t) p]);
        if (point_dist.size() > k)
            point_dist.pop();
    }
    for (ptrdiff_t l = 0; l < (ptrdiff_t) k; ++l) {
        closest_points[(ptrdiff_t) (k - 1 - l)] = point_dist.top().first;
        dist_closest_points[(ptrdiff_t) (k - 1 - l)] =
            point_dist.top().second;
        point_dist.pop();
    }
    assert(std::is_sorted(
        dist_closest_points,
        dist_closest_points + (ptrdiff_t) k));
    // std::cout << "Computed exact k-NN for queries: [" << q_b << "," << q_e
    //       << ")" << std::endl;

    delete[] dist_matrix;
    // mkl_free(dist_matrix);

    delete[] points_l2sq;
    delete[] queries_l2sq;

    // if (metric == Metric::COSINE) {
    //     delete[] points;
    //     delete[] queries;
    // }
}


// void Filter(struct Result &res, int k) {
//     // find_elbow:
//     float *d = res.dist_closest_points;
//     float x0 = 0, y0 = d[0];
//     float x1 = k - 1, y1 = d[k - 1];
//     float max_dist = -1;
//     int idx = 0;
//     for (int i = 0; i < k; ++i) {
//         float xi = i, yi = d[i];
//         float num = fabs((y1 - y0)*xi - (x1 - x0)*yi + x1*y0 - y1*x0);
//         float den = sqrt((y1 - y0)*(y1 - y0) + (x1 - x0)*(x1 - x0));
//         float dist_line = num / den;
//         if (dist_line > max_dist) { 
//             max_dist = dist_line; 
//             idx = i; 
//         }
//     }
//     res.filter_len = idx + 1;
// }

void GTCache::compute_l2sq(float *const points_l2sq, const float *const matrix,
                  const uint32_t num_points, const int dim) {
    assert(points_l2sq != NULL);
#pragma omp parallel for schedule(static, 65536)
    for (uint32_t d = 0; d < num_points; ++d)
        points_l2sq[d] = cblas_sdot(dim, matrix + (ptrdiff_t) d * (ptrdiff_t) dim,
                                    1, matrix + (ptrdiff_t) d * (ptrdiff_t) dim, 1);
}

template<class T>
T GTCache::div_round_up(const T numerator, const T denominator) {
    return (numerator % denominator == 0) ? (numerator / denominator)
                                        : 1 + (numerator / denominator);
}

void GTCache::distsq_to_points(
    const size_t dim,
    float       *dist_matrix,  // Col Major, cols are queries, rows are points
    size_t npoints, const float *const points,
    const float *const points_l2sq,  // points in Col major
    size_t nqueries, const float *const queries,
    const float *const queries_l2sq,  // queries in Col major
    float *ones_vec)  // Scratchspace of num_data size and init to 1.0
{
    bool ones_vec_alloc = false;
    if (ones_vec == NULL) {
        ones_vec = new float[nqueries > npoints ? nqueries : npoints];
        std::fill_n(ones_vec, nqueries > npoints ? nqueries : npoints, (float) 1.0);
        ones_vec_alloc = true;
    }
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, npoints, nqueries, dim,
                (float) -2.0, points, dim, queries, dim, (float) 0.0, dist_matrix,
                npoints);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, npoints, nqueries, 1,
                (float) 1.0, points_l2sq, npoints, ones_vec, nqueries,
                (float) 1.0, dist_matrix, npoints);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, npoints, nqueries, 1,
                (float) 1.0, ones_vec, npoints, queries_l2sq, nqueries,
                (float) 1.0, dist_matrix, npoints);
    if (ones_vec_alloc)
        delete[] ones_vec;
}

void GTCache::inner_prod_to_points(
    const size_t dim,
    float       *dist_matrix,  // Col Major, cols are queries, rows are points
    size_t npoints, const float *const points, size_t nqueries,
    const float *const queries,
    float *ones_vec)  // Scratchspace of num_data size and init to 1.0
{
    bool ones_vec_alloc = false;
    if (ones_vec == NULL) {
        ones_vec = new float[nqueries > npoints ? nqueries : npoints];
        std::fill_n(ones_vec, nqueries > npoints ? nqueries : npoints, (float) 1.0);
        ones_vec_alloc = true;
    }
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, npoints, nqueries, dim,
                (float) -1.0, points, dim, queries, dim, (float) 0.0, dist_matrix,
                npoints);

    if (ones_vec_alloc)
        delete[] ones_vec;
}

}