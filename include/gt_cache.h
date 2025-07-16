#ifndef GT_CACHE_H
#define GT_CACHE_H

#include <vector>
#include <string>
#include <cstdint>
#include <future>

namespace efanna2e
{
    struct BatchResult
    {
        std::vector<uint32_t> knn;
        double time_ms;
    };

    class GTCache
    {
    public:
        std::vector<BatchResult> results;

        GTCache(const std::string cache_file, const uint32_t k);
        ~GTCache();
        void WriteCache(uint32_t *knn, double time_ms);

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