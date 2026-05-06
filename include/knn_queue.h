// #include <semaphore.h>
#include <nvml.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <semaphore>  // C++20 信号量
#include <vector>

#include "gpufuncs.cuh"

namespace efanna2e {
// enum KNNType { host, device };
class KNN_Queue {
   private:
    // std::atomic<int> head{0};
    // std::atomic<int> tail{0};
    std::atomic<int> size{0};
    std::atomic<bool> stop_flag{false};
    int head = 0, tail = 0;
    int k, L;
    int* d_knns;
    std::vector<int> knns;
    // int *h_knn;
    std::vector<int> batch_size, batch_ids;

    // std::counting_semaphore<> empty_sem, filled_sem;
    // std::binary_semaphore half_full_sem, half_empty_sem; // 半满条件信号量（初始为0）
    std::condition_variable_any write_cv, read_cv;
    std::mutex write_mtx, read_mtx;

    nvmlDevice_t device;

    GPUFuncs* gpufuncs;

   public:
    KNN_Queue(const int k, const int L, GPUFuncs* gpufuncs);
    ~KNN_Queue();
    bool WriteTail(const int* d_new_knns, const int batch, const int batch_id);
    bool WriteTailHost(const int* new_knns, const int batch, const int batch_id);
    int ReadHead(int& batch_id, int* d_knn_res);
    int ReadHeadHost(int& batch_id, int* knn_res);
    void Stop();
};
}  // namespace efanna2e