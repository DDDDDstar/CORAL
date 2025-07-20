// #include <semaphore.h>
#include <semaphore> // C++20 信号量
#include <atomic>
#include <vector>
namespace efanna2e
{
    enum KNNType
    {
        host,
        device
    };
    class KNN_Queue
    {
    private:
        int head = 0, tail = 0, k, L;
        uint32_t *d_knns, *h_knn;
        std::vector<int> batch_size;
        // sem_t empty_sem, filled_sem;
        std::counting_semaphore<> empty_sem;
        std::counting_semaphore<> filled_sem;

    public:
        KNN_Queue(const int k, const int L);
        ~KNN_Queue();
        void WriteTail(uint32_t *new_knn, const int batch, KNNType type);
        int ReadHead(uint32_t *knn);
    };
}