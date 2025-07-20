#ifndef GT_CACHE_H
#define GT_CACHE_H

#include <vector>
#include <string>
#include <cstdint>
#include <future>
#include <cuda_runtime.h>

namespace efanna2e
{
    struct BatchResult
    {
        std::vector<uint32_t> knn;
        int batch;
        double time_ms;

        BatchResult() = default;
        BatchResult(uint32_t *d_knn, const int k, const int batch, const double time_ms)
            : batch(batch), time_ms(time_ms)
        {
            knn.resize(k * batch);
            cudaMemcpy(knn.data(), d_knn, batch * k * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        }
        BatchResult(const int k, const int batch) : batch(batch)
        {
            knn.resize(k * batch);
        }
    };

    class GTCache
    {
    public:
        std::vector<BatchResult> results;

        GTCache(const std::string cache_file, const uint32_t k);
        ~GTCache();
        void WriteCache(uint32_t *d_knn, const int len, const double time_ms);

    private:
        const uint32_t k;
        const std::string cache_file;
        uint32_t save_thres, new_num = 0;
        std::future<void> save_future;

        // 从二进制文件加载缓存结果
        void LoadCache();
        void SaveCache();
    };
}

#endif // GT_CACHE_H