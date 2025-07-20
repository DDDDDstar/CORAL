#include <cassert>
#include <iostream>
#include <cstring>
#include "knn_queue.h"
#include "knn.cuh"
#include "fileout.h"

using namespace efanna2e;
KNN_Queue::KNN_Queue(const int k, const int L) : k(k), L(L), empty_sem(L), filled_sem(0)
{
    CUDA_CHECK(cudaMalloc(&d_knns, L * BATCH * k * sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocHost(&h_knn, BATCH * k * sizeof(uint32_t)));
    batch_size.resize(L);
    // sem_init(&empty_sem, 0, L);
    // sem_init(&filled_sem, 0, 0);
}
KNN_Queue::~KNN_Queue()
{
    CUDA_CHECK(cudaFree(d_knns));
    CUDA_CHECK(cudaFreeHost(h_knn));
    // sem_destroy(&empty_sem);
    // sem_destroy(&filled_sem);
}

void KNN_Queue::WriteTail(uint32_t *new_knn, const int batch, KNNType type)
{
    empty_sem.acquire();
    // sem_wait(&empty_sem);

    if (type == KNNType::host)
        memcpy(h_knn, new_knn, BATCH * k * sizeof(uint32_t));

    if (tail > L)
        tail = 0;

    fo.qprint(head, tail, L, -1, tail);

    batch_size[tail] = batch;
    CUDA_CHECK(cudaMemcpy(
        d_knns + (tail++) * BATCH * k, type == KNNType::host ? h_knn : new_knn,
        batch * k * sizeof(uint32_t),
        type == KNNType::host ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));

    // sem_post(&filled_sem);
    filled_sem.release();
}

int KNN_Queue::ReadHead(uint32_t *knn)
{
    // sem_wait(&filled_sem);
    filled_sem.acquire();

    assert(head != tail);

    if (head > L)
        head = 0;

    const int batch = batch_size[head];
    CUDA_CHECK(cudaMemcpy(
        knn, d_knns + (head++) * BATCH * k, batch * k * sizeof(uint32_t),
        cudaMemcpyDeviceToDevice));

    // sem_post(&empty_sem);
    empty_sem.release();

    fo.qprint(head, tail, L, head - 1);

    return batch;
}