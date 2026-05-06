#pragma once
#include <cuda_runtime.h>
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIDMap.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexFlat.h>

#include "uni.h"
#include "utils.cuh"

#define BLOCK_SIZE 20'000'000

namespace efanna2e {
class MyFaiss {
   public:
    MyFaiss(int dim, int k, DIST_METRIC m, bool gpu = true) : k(k), dim(dim), m(m) {
        if (m == DIST_METRIC::L2_) {
            if (gpu) {
                index = new GPUFlatL2(&res, dim);
            } else
                index = new CPUFlatL2(dim);
        } else if (m == DIST_METRIC::IP_) {
            if (gpu) {
                index = new GPUFlatIP(&res, dim);
            } else
                index = new CPUFlatIP(dim);
        }
    }
    ~MyFaiss() {
        fo.print("Faiss destructor called.");  // 打印析构函数被调用的日志信息
        if (index != nullptr) delete index;
    }
    void Rebuild(const float* bases, const faiss::idx_t* ids, int nb) {
        std::unique_lock<std::shared_mutex> lock(mutex);

        index->reset();
        index->add(nb, bases);

        // auto flat = std::make_unique<faiss::IndexFlatL2>(dim);
        // auto idmap = std::make_unique<faiss::IndexIDMap2>(flat.release());
        // idmap->own_fields = true;

        // idmap->add_with_ids((faiss::idx_t)nb, bases, ids);

        // faiss::gpu::GpuClonerOptions opts;
        // opts.useFloat16 = false;
        // new_idx = faiss::gpu::index_cpu_to_gpu(&res, 0, idmap.get(), &opts);

        // index = new_idx;
    }

    inline void Set(cudaStream_t stream, int device = 0) { res.setDefaultStream(device, stream); }
    inline void Add(const float* bases, size_t bn, bool reset = true) {
        if (reset) index->reset();
        index->add(bn, bases);

        fo.print("GPU Faiss add n=" + TOS(bn));
    }

    // void Search(const float* bases, size_t bn, const float* queries, int64_t* ids_res,
    //             float* dists_res, size_t qn) {
    //     // search batch on GPU
    //     const int batch_num = (bn + BLOCK_SIZE - 1) / BLOCK_SIZE;
    //     std::vector<float> D_batch(qn * k * batch_num);
    //     std::vector<int64_t> I_batch(qn * k * batch_num);
    //     for (size_t i = 0; i < bn; i += BLOCK_SIZE) {
    //         const size_t batch = std::min<size_t>(bn - i, (size_t)BLOCK_SIZE),
    //                      batch_i = i / BLOCK_SIZE;
    //         index->reset();
    //         index->add(batch, bases + i * dim);
    //         auto* Iptr = I_batch.data() + batch_i * qn * k;
    //         index->search((faiss::idx_t)qn, queries, k, D_batch.data() + batch_i * qn * k,
    //         Iptr); for (size_t t = 0; t < qn * k; ++t)
    //             if (Iptr[t] >= 0) Iptr[t] += i;
    //     }
    //     merge_topk(D_batch, I_batch, ids_res, dists_res, k, qn, batch_num);
    // }

    inline void Search(const float* queries, std::vector<faiss::idx_t>& ids_res,
                       std::vector<float>& dists_res, size_t qn) {
        std::shared_lock<std::shared_mutex> lock(mutex);
        index->search((faiss::idx_t)qn, queries, k, dists_res.data(), ids_res.data());
    }

    inline void Search(const float* queries, faiss::idx_t* ids_res, float* dists_res, size_t qn,
                       int k) {
        std::shared_lock<std::shared_mutex> lock(mutex);
        index->search((faiss::idx_t)qn, queries, k, dists_res, ids_res);
    }

    inline void Search(const float* queries, faiss::idx_t* ids_res, float* dists_res, size_t qn) {
        std::shared_lock<std::shared_mutex> lock(mutex);
        index->search((faiss::idx_t)qn, queries, k, dists_res, ids_res);
    }

    void CPUSearch(const float* queries, int* ids_res, int nq) {
        std::shared_lock<std::shared_mutex> lock(mutex);
        std::vector<faiss::idx_t> ids(nq * k);
        std::vector<float> dists(nq * k);
        index->search((faiss::idx_t)nq, queries, k, dists.data(), ids.data());
        for (size_t i = 0; i < nq * k; ++i) ids_res[i] = (int)ids[i];
    }

   private:
    GPUResources res;
    faiss::Index* index = nullptr;
    std::shared_mutex mutex;

    int dim, k;
    DIST_METRIC m;

    void merge_topk(const std::vector<float>& D_batch, const std::vector<int64_t>& I_batch,
                    int64_t* I, float* D, size_t k, int qn, int batch_num) {
        for (size_t q = 0; q < qn; q++) {
            std::priority_queue<std::pair<float, int64_t>> pq;
            for (int bi = 0; bi < batch_num; bi++) {
                for (int j = 0; j < k; j++) {
                    const size_t idx = (size_t)bi * qn * k + (size_t)q * k + j;
                    if (I_batch[idx] < 0) continue;
                    pq.push({D_batch[idx], I_batch[idx]});
                    if (pq.size() > k) pq.pop();
                }
            }
            for (int j = k - 1; j >= 0; j--) {
                I[q * k + j] = pq.top().second;
                D[q * k + j] = pq.top().first;
                pq.pop();
            }
        }
    }
};
};  // namespace efanna2e