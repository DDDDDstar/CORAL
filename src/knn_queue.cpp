#include <cassert>
#include <iostream>
#include <cstring>
#include <chrono>
#include <nvml.h>

#include "knn_queue.h"
#include "fileout.h"
#include "gpufuncs.cuh"

#define WAIT_TIME 60

using namespace efanna2e;
KNN_Queue::KNN_Queue(const int k, const int L, GPUFuncs *gpufuncs)
    : k(k), L(L), gpufuncs(gpufuncs), size(0), stop_flag(false)
{
    CUDA_CHECK(cudaMalloc(&d_knns, L * BATCH * k * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&h_knn, BATCH * k * sizeof(int)));
    batch_size.resize(L);
    batch_ids.resize(L);
    NVML_CHECK(nvmlInit());
    NVML_CHECK(nvmlDeviceGetHandleByIndex(0, &device));
}
KNN_Queue::~KNN_Queue()
{
    CUDA_CHECK(cudaFree(d_knns));
    CUDA_CHECK(cudaFreeHost(h_knn));
}

void KNN_Queue::Stop()
{
    if (stop_flag.load())
        return;

    stop_flag.store(true);
    write_cv.notify_one();
    read_cv.notify_one();
    fo.print("Stop KNN_Queue...");
}

bool KNN_Queue::WriteTail(int *new_knn, const int batch, const int batch_id, KNNType type)
{
    int cur_size = size.load();
    assert(cur_size >= 0 && cur_size <= L);
    if (cur_size == L)
    {
        std::unique_lock<std::mutex> lock(write_mtx);
        write_cv.wait(lock, [this]()
                      { return size.load() < L / 2 || stop_flag.load(); });
    }

    if (stop_flag.load())
        return false;

    if (type == KNNType::host)
        memcpy(h_knn, new_knn, BATCH * k * sizeof(int));

    batch_size[tail] = batch;
    batch_ids[tail] = batch_id;

    CUDA_CHECK(cudaMemcpy(
        d_knns + tail * BATCH * k,
        type == KNNType::host ? h_knn : new_knn,
        batch * k * sizeof(int),
        type == KNNType::host
            ? cudaMemcpyHostToDevice
            : cudaMemcpyDeviceToDevice));

    const int t = tail;
    if (tail == L - 1)
        tail = 0;
    else
        tail++;

    // fo.qprint(head, t, L, -1, t);

    size.fetch_add(1);
    read_cv.notify_one();

    return true;
}

int KNN_Queue::ReadHead(int *knn, int &batch_id)
{
    int cur_size = size.load();
    assert(cur_size >= 0 && cur_size <= L);
    if (cur_size == 0)
    {
        std::unique_lock<std::mutex> lock(read_mtx);
        read_cv.wait(lock, [this]()
                     { return size.load() > L / 2 || stop_flag.load(); });
    }

    if (stop_flag.load())
        return -1;

    const int batch = batch_size[head];
    batch_id = batch_ids[head];

    CUDA_CHECK(cudaMemcpy(
        knn, d_knns + head * BATCH * k, batch * k * sizeof(int),
        cudaMemcpyDeviceToDevice));

    const int h = head;
    if (head == L - 1)
        head = 0;
    else
        head++;

    // fo.qprint(h, tail, L, h);

    size.fetch_sub(1);
    write_cv.notify_one();

    return batch;
}