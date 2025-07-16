#include <semaphore.h>
#include <atomic>
namespace efanna2e {
enum KNNType { host, device };
class KNN_Queue {
    private:
        int head = 0, tail = 0, k, L;
        uint32_t *d_knns, *h_knn;
        sem_t empty_sem, filled_sem;
        
    public:
        KNN_Queue(const int k, const int L);
        ~KNN_Queue();
        void WriteTail(uint32_t *new_knn, KNNType type);
        void ReadHead(uint32_t *knn);
};
}