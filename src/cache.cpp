#include "cache.h"

#include <fcntl.h>   // for open/close
#include <unistd.h>  // for fsync

#include <cstdio>  // for std::rename
#include <filesystem>
#include <fstream>
#include <iostream>

#include "fileout.h"
#include "uni.h"

using namespace efanna2e;
// 构造函数，用于初始化Cache对象
Cache::Cache(const std::string cache_file, const std::string graph_file, int& last_batch_id,
             const int k, const int n)
    : cache_file(cache_file), graph_file(graph_file), k(k), save_thres(50) {
    if (PC.use_knn_cache && cache_file.size() > 0) {
        LoadGTCache();
        use_gt_cache = true;
    }

    // if (USE_GRAPH_CACHE)
    //     LoadGraphCache(graph, graph_deg, graph_dist, noniso_num, last_batch_id, total_degree,
    //                    total_time, tres, n);
}

Cache::Cache(const std::string cache_file, int& last_batch_id, const int k, const int n)
    : cache_file(cache_file), k(k), save_thres(10) {
    if (PC.use_knn_cache && cache_file.size() > 0) {
        LoadGTCache();
        use_gt_cache = true;
    }
}

Cache::~Cache() {}

void Cache::WriteGTCache(const int* d_knns, const int* qids, const int batch_id, const int batch,
                         const double time_s) {
    results.emplace_back(d_knns, qids, k, batch, time_s);
    total_query_num += batch;

    if (!use_gt_cache) return;

    if (save_future.valid()) {
        if (save_future.wait_for(std::chrono::seconds(5)) == std::future_status::timeout)
            save_thres += 5;
        save_future.wait();
    }

    if (++new_num >= save_thres) {
        save_future = std::async(std::launch::async, [this]() { this->SaveGTCache(); });
        new_num = 0;
    }
}

void Cache::WriteGTCacheHost(const int* knns, int batch_id, int batch) {
    results.emplace_back(knns, k, batch);
    total_query_num += batch;

    if (!use_gt_cache) return;

    if (save_future.valid()) {
        if (save_future.wait_for(std::chrono::seconds(5)) == std::future_status::timeout)
            save_thres += 5;
        save_future.wait();
    }

    if (++new_num >= save_thres) {
        save_future = std::async(std::launch::async, [this]() { this->SaveGTCache(); });
        new_num = 0;
    }
}

void Cache::LoadGTCache() {
    std::ifstream file(cache_file, std::ios::binary);
    if (!file.is_open()) {
        fo.print("Cache file not exist: " + cache_file);
        return;
    }

    int n = 0;
    total_query_num = 0;
    file.read(reinterpret_cast<char*>(&n), sizeof(int));
    fo.iprint("Load GT cache from " + cache_file + ", total batch size: " + std::to_string(n));

    results.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        int batch;
        if (!file.read(reinterpret_cast<char*>(&batch), sizeof(int)) || batch <= 0) {
            fo.iprint("Read batch failed, result size: " + std::to_string(results.size()));
            break;
        }
        BatchResult res(k, batch);
        // if (!file.read(reinterpret_cast<char*>(res.qids.data()), sizeof(int) * batch)) {
        //     fo.iprint("Read qids failed, result size: " + TOS(results.size()) +
        //               ", batch: " + TOS(batch));
        //     break;
        // }
        if (!file.read(reinterpret_cast<char*>(res.knn.data()), sizeof(int) * k * batch)) {
            fo.iprint("Read closest_points failed, result size: " + TOS(results.size()) +
                      ", batch: " + TOS(batch));
            break;
        }
        // if (!file.read(reinterpret_cast<char *>(&res.time_ms), sizeof(double)))
        //     fo.eprint("Read time failed, result size: " + std::to_string(results.size()));

        total_query_num += batch;
        results.push_back(std::move(res));
    }

    // fo.print("Load GT cache finished with " + TOS(results.size()) + " batches");
}

void Cache::SaveGTCache() {
    // 1. 生成临时文件名
    std::string tmp_file = cache_file + ".tmp";
    // 提取目录路径并创建（若不存在）
    std::filesystem::path dir_path = std::filesystem::path(tmp_file).parent_path();
    if (!std::filesystem::exists(dir_path))
        std::filesystem::create_directories(dir_path);  // 递归创建多级目录
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
    // 序列化数据
    int n = results.size();
    fwrite(&n, sizeof(int), 1, file);
    for (const auto& res : results) {
        fwrite(&res.batch, sizeof(int), 1, file);
        // fwrite(res.qids.data(), sizeof(int), res.batch, file);
        fwrite(res.knn.data(), sizeof(int), k * res.batch, file);
        // fwrite(&res.time_ms, sizeof(double), 1, file);
    }

    // 4. 强制数据落盘
    fflush(file);
    fsync(fd);     // 确保数据写入物理磁盘
    fclose(file);  // 自动关闭 fd
    // 5. 原子重命名临时文件
    if (std::rename(tmp_file.c_str(), cache_file.c_str()) != 0) {
        std::cerr << "Failed to rename temporary file to " << cache_file << std::endl;
        std::remove(tmp_file.c_str());  // 清理临时文件
    }
    // else fo.print("gt cache saved, total batch num: " + std::to_string(n));
}

void Cache::SaveGraphCache(const std::vector<std::vector<int>>& graph,
                           const std::vector<std::vector<float>>& graph_dist,
                           const std::vector<int>& graph_deg,
                           std::vector<std::shared_mutex>& graph_mtx, const int last_batch_id,
                           const float total_time, const Test_Result tres, const int n) {
    // if (!USE_GRAPH_CACHE)
    return;

    // 1. 生成临时文件名
    std::string tmp_file = graph_file + ".tmp";
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

    fwrite(&total_time, sizeof(float), 1, file);
    fwrite(&last_batch_id, sizeof(int), 1, file);
    fwrite(&tres, sizeof(Test_Result), 1, file);

    // 序列化时间数据
    const std::string avg_times_str = avg_time_serialize();
    size_t size = avg_times_str.size();
    fwrite(&size, sizeof(size_t), 1, file);
    fwrite(avg_times_str.data(), sizeof(char), size, file);

    // 序列化图数据
    for (int i = 0; i < n; ++i) {
        std::shared_lock<std::shared_mutex> lock(graph_mtx[i]);
        size = graph_deg[i];
        fwrite(&size, sizeof(size_t), 1, file);
        if (size > 0) {
            fwrite(graph[i].data(), sizeof(int), size, file);
            fwrite(graph_dist[i].data(), sizeof(float), size, file);
        }
    }

    // 4. 强制数据落盘
    fflush(file);
    fsync(fd);     // 确保数据写入物理磁盘
    fclose(file);  // 自动关闭 fd
    // 5. 原子重命名临时文件
    if (std::rename(tmp_file.c_str(), graph_file.c_str()) != 0) {
        std::cerr << "Failed to rename temporary file to " << graph_file << std::endl;
        std::remove(tmp_file.c_str());  // 清理临时文件
    } else
        fo.print("graph cache saved with last_batch_id: " + std::to_string(last_batch_id) +
                 ", total_time: " + std::to_string(total_time));
}

void Cache::LoadGraphCache(std::vector<std::vector<int>>& graph, std::vector<int>& graph_deg,
                           std::vector<std::vector<float>>& graph_dist, int& noniso_num,
                           int& last_batch_id, int& total_degree, float& total_time,
                           Test_Result& tres, const int n) {
    std::ifstream file(graph_file, std::ios::binary);
    if (!file.is_open()) {
        fo.print("Cache file not exist: " + graph_file);
        return;
    }

    fo.print("Load Graph cache from " + graph_file);

    if (!file.read(reinterpret_cast<char*>(&total_time), sizeof(float)))
        fo.eprint("Read total_time failed");

    if (!file.read(reinterpret_cast<char*>(&last_batch_id), sizeof(int)))
        fo.eprint("Read last_batch_id failed");

    if (!file.read(reinterpret_cast<char*>(&tres), sizeof(Test_Result)))
        fo.eprint("Read test_result failed");

    // 1. 读取时间数据
    size_t size;
    if (!file.read(reinterpret_cast<char*>(&size), sizeof(size_t)))
        fo.eprint("Read avg_times_str size failed");
    fo.print("avg_times_str size: " + std::to_string(size));
    std::string avg_times_str(size, '\0');  // 需要初始化长度，否则后续读取失败！
    if (!file.read(reinterpret_cast<char*>(avg_times_str.data()), sizeof(char) * size))
        fo.eprint("Read avg_times_str failed, size: " + std::to_string(size));
    avg_time_deserialize(avg_times_str);

    // 2. 读取图数据
    noniso_num = 0;
    total_degree = 0;
    for (int i = 0; i < n; ++i) {
        if (!file.read(reinterpret_cast<char*>(&size), sizeof(size_t)))
            fo.eprint("Read graph size failed, i: " + std::to_string(i));

        if (size > 0) {
            total_degree += size;
            graph_deg[i] = size;
            graph[i].resize(size);
            graph_dist[i].resize(size);
            noniso_num++;
            if (!file.read(reinterpret_cast<char*>(graph[i].data()), sizeof(int) * size))
                fo.eprint("Read graph failed, i: " + std::to_string(i));
            if (!file.read(reinterpret_cast<char*>(graph_dist[i].data()), sizeof(float) * size))
                fo.eprint("Read graph_dist failed, i: " + std::to_string(i));
        }
    }

    std::string iso_ratio_str = std::to_string(100.0 * noniso_num / n);
    fo.iprint("Load Graph cache finished with isolated ratio = " +
              iso_ratio_str.substr(0, iso_ratio_str.find(".") + 3) + "%, total_degree: " +
              std::to_string(total_degree) + "\nlast_batch_id: " + std::to_string(last_batch_id) +
              ", total_time: " + std::to_string(total_time) + "\nrecall: " +
              std::to_string(tres.get_recall()) + ", hops: " + std::to_string(tres.get_hops()));
}
