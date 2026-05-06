#include "knn_queue.h"

#include <nvml.h>

#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>

#include "fileout.h"
#include "gpufuncs.cuh"

#define WAIT_TIME 60

using namespace efanna2e;
KNN_Queue::KNN_Queue(const int k, const int L, GPUFuncs* gpufuncs)
    : k(k), L(L), gpufuncs(gpufuncs), size(0), stop_flag(false) {
    CUDA_CHECK(cudaMalloc(&d_knns, L * PC.batch * k * sizeof(int)));
    knns.resize(L * PC.batch * k);
    // CUDA_CHECK(cudaMallocHost(&h_knn, PC.batch * k * sizeof(int)));
    batch_size.resize(L);
    batch_ids.resize(L);
    NVML_CHECK(nvmlInit());
    NVML_CHECK(nvmlDeviceGetHandleByIndex(0, &device));
}
KNN_Queue::~KNN_Queue() {
    CUDA_CHECK(cudaFree(d_knns));
    // CUDA_CHECK(cudaFreeHost(h_knn));
}

void KNN_Queue::Stop() {
    if (stop_flag.load()) return;

    stop_flag.store(true);
    // write_cv.notify_one();
    read_cv.notify_one();
    fo.print("Stop KNN_Queue...");
}

bool KNN_Queue::WriteTail(const int* d_new_knns, const int batch, const int batch_id) {
    int cur_size = size.load();
    assert(cur_size >= 0 && cur_size <= L);
    if (cur_size == L) {
        std::unique_lock<std::mutex> lock(write_mtx);
        write_cv.wait(lock, [this]() { return size.load() < L; });
    }

    // if (stop_flag.load()) return false;

    batch_size[tail] = batch;
    batch_ids[tail] = batch_id;

    CUDA_CHECK(cudaMemcpy(d_knns + tail * PC.batch * k, d_new_knns, batch * k * sizeof(int),
                          cudaMemcpyDeviceToDevice));

    const int t = tail;
    if (tail == L - 1)
        tail = 0;
    else
        tail++;

    // fo.qprint(head, t, L, -1, t);

    if (size.fetch_add(1) >= L / 2) read_cv.notify_one();

    return true;
}

bool KNN_Queue::WriteTailHost(const int* new_knns, const int batch, const int batch_id) {
    int cur_size = size.load();
    assert(cur_size >= 0 && cur_size <= L);
    if (cur_size == L) {
        std::unique_lock<std::mutex> lock(write_mtx);
        write_cv.wait(lock, [this]() { return size.load() < L; });
    }

    // if (stop_flag.load()) return false;

    batch_size[tail] = batch;
    batch_ids[tail] = batch_id;

    // CUDA_CHECK(cudaMemcpy(d_knns + tail * PC.batch * k, new_knns, batch * k * sizeof(int),
    //                       cudaMemcpyHostToDevice));
    std::memcpy(knns.data() + tail * PC.batch * k, new_knns, batch * k * sizeof(int));

    const int t = tail;
    if (tail == L - 1)
        tail = 0;
    else
        tail++;

    // fo.qprint(head, t, L, -1, t);

    if (size.fetch_add(1) >= L / 2) read_cv.notify_one();

    return true;
}

int KNN_Queue::ReadHead(int& batch_id, int* d_knn_res) {
    int cur_size = size.load();
    assert(cur_size >= 0 && cur_size <= L);
    if (cur_size == 0) {
        std::unique_lock<std::mutex> lock(read_mtx);
        read_cv.wait(lock, [this]() { return size.load() > 0 || stop_flag.load(); });
    }

    if (size.load() == 0) return -1;

    const int batch = batch_size[head];
    batch_id = batch_ids[head];

    CUDA_CHECK(cudaMemcpy(d_knn_res, d_knns + head * PC.batch * k, batch * k * sizeof(int),
                          cudaMemcpyDeviceToDevice));

    const int h = head;
    if (head == L - 1)
        head = 0;
    else
        head++;

    // fo.qprint(h, tail, L, h);

    if (size.fetch_sub(1) <= L / 2) write_cv.notify_one();

    return batch;
}

int KNN_Queue::ReadHeadHost(int& batch_id, int* knn_res) {
    int cur_size = size.load();
    assert(cur_size >= 0 && cur_size <= L);
    if (cur_size == 0) {
        std::unique_lock<std::mutex> lock(read_mtx);
        read_cv.wait(lock, [this]() { return size.load() > 0 || stop_flag.load(); });
    }

    if (size.load() == 0) return -1;

    const int batch = batch_size[head];
    batch_id = batch_ids[head];

    std::memcpy(knn_res, knns.data() + head * PC.batch * k, batch * k * sizeof(int));

    const int h = head;
    if (head == L - 1)
        head = 0;
    else
        head++;

    // fo.qprint(h, tail, L, h);

    if (size.fetch_sub(1) <= L / 2) write_cv.notify_one();

    return batch;
}