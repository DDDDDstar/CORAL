#include <fstream>
#include <iostream>
#include <unistd.h> // for fsync
#include <cstdio>   // for std::rename
#include <fcntl.h>  // for open/close

#include "gt_cache.h"
#include "knn.cuh"
#include "fileout.h"

using namespace efanna2e;
GTCache::GTCache(const std::string cache_file, const uint32_t k)
    : cache_file(cache_file), k(k), save_thres(5)
{
    LoadCache();
}

GTCache::~GTCache() {}

void GTCache::WriteCache(uint32_t *d_knn, double time_ms)
{
    BatchResult new_res;
    new_res.knn.resize(k * BATCH);
    cudaMemcpy(new_res.knn.data(), d_knn, BATCH * k * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    new_res.time_ms = time_ms;
    results.push_back(new_res);

    if (++new_num >= save_thres)
    {
        if (save_future.valid())
        {
            save_future.get();
            save_thres += 5;
        }
        save_future = std::async(std::launch::async, [this]()
                                 { this->SaveCache(); });
        new_num = 0;
    }
}

void GTCache::LoadCache()
{
    std::ifstream file(cache_file, std::ios::binary);
    if (!file.is_open())
    {
        fo.print("Cache file not exist: " + cache_file);
        return;
    }

    int n = 0;
    file.read(reinterpret_cast<char *>(&n), sizeof(int));
    fo.print("Load GT cache from " + cache_file + ", total batch size: " + std::to_string(n));

    results.reserve(n);
    for (size_t i = 0; i < n; ++i)
    {
        BatchResult res;
        res.knn.resize(k * BATCH);
        if (!file.read(
                reinterpret_cast<char *>(res.knn.data()), sizeof(uint32_t) * k * BATCH))
        {
            fo.eprint(
                "Read closest_points failed, result size: " + std::to_string(results.size()));
        }
        if (!file.read(reinterpret_cast<char *>(&res.time_ms), sizeof(double)))
            fo.eprint("Read time failed, result size: " + std::to_string(results.size()));

        results.push_back(res);
    }
    fo.iprint("Load GT cache finished.");
}

void GTCache::SaveCache()
{
    // 1. 生成临时文件名
    std::string tmp_file = cache_file + ".tmp";
    // 2. 使用底层文件描述符确保 fsync
    int fd = open(tmp_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd == -1)
    {
        std::cerr << "Failed to create temporary file: " << tmp_file << std::endl;
        return;
    }
    // 3. 将数据写入临时文件
    FILE *file = fdopen(fd, "wb");
    if (!file)
    {
        close(fd);
        std::cerr << "Failed to open file stream" << std::endl;
        return;
    }
    // 序列化数据
    int n = results.size();
    fwrite(&n, sizeof(int), 1, file);
    for (const auto &res : results)
    {
        fwrite(res.knn.data(), sizeof(uint32_t), k * BATCH, file);
        fwrite(&res.time_ms, sizeof(double), 1, file);
    }
    // 4. 强制数据落盘
    fflush(file);
    fsync(fd);    // 确保数据写入物理磁盘
    fclose(file); // 自动关闭 fd
    // 5. 原子重命名临时文件
    if (std::rename(tmp_file.c_str(), cache_file.c_str()) != 0)
    {
        std::cerr << "Failed to rename temporary file to " << cache_file << std::endl;
        std::remove(tmp_file.c_str()); // 清理临时文件
    }
    else
    {
        fo.iprint("gt cache saved, total batch num: " + std::to_string(n));
    }
}
