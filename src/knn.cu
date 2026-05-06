#include <curand_kernel.h>

#include <cfloat>
#include <iostream>
#include <string>
#include <utility>

#include "gpufuncs.cuh"
#include "uni.h"
#include "utils.cuh"

namespace efanna2e {
__global__ void hits_count_kernel(const int* __restrict__ knns, int batch, int k,
                                  uint32_t* __restrict__ hits, uint32_t max_hit) {
    const int i = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    if (i >= batch) return;

    const auto* knn = knns + i * k;
    for (int j = tid; j < k; j += tpb) {
        const auto id = knn[j];
        assert(id >= 0);

        uint32_t* hit = hits + id;
        // if (*hit > max_hit) continue;
        atomicAdd(hit, 1);
    }
    __syncthreads();
}

__global__ void hits_update_kernel(const faiss::idx_t* __restrict__ knn_idxs,
                                   const faiss::idx_t* __restrict__ compact_ids, int batch, int k,
                                   int* __restrict__ knn_ids_res, uint32_t* __restrict__ hits,
                                   uint32_t* __restrict__ over_cnt, uint32_t max_hit) {
    const int i = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    if (i >= batch) return;

    __shared__ uint32_t s_over_cnt;
    if (!tid) s_over_cnt = 0;
    __syncthreads();

    const auto* knn = knn_idxs + i * k;
    auto* knn_res = knn_ids_res + i * k;
    for (int j = tid; j < k; j += tpb) {
        const auto idx = knn[j];
        assert(idx >= 0);
        const int id = compact_ids[idx];
        assert(id >= 0);

        knn_res[j] = id;

        uint32_t* hit = hits + id;
        // if (*hit > max_hit) continue;
        if (atomicAdd(hit, 1) == max_hit) atomicAdd(&s_over_cnt, 1);
    }
    __syncthreads();

    if (!tid) atomicAdd(over_cnt, s_over_cnt);
}

__global__ void hits_update_kernel(const faiss::idx_t* __restrict__ knn_ids, int batch, int k,
                                   int* __restrict__ knn_ids_out, uint32_t* __restrict__ hits) {
    const int i = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    if (i >= batch) return;

    const auto* knn = knn_ids + i * k;
    int* knn_res = knn_ids_out + i * k;
    for (int j = tid; j < k; j += tpb) {
        const auto id = knn[j];
        assert(id >= 0);
        knn_res[j] = (int)id;
        atomicAdd(hits + id, 1);
    }
}

bool GPUFuncs::rebuild() {
    static int last_n = 0;
    static std::vector<float> base_compact;
    if (!last_n) {
        last_n = bp.base_n;
        base_compact.resize(last_n * bp.dim);
        std::memcpy(base_compact.data(), h_base, last_n * bp.dim * sizeof(float));
        return false;
    }

    auto rs = now_time();

    cudaMemcpy(h_hits, d_hits, bp.base_n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    std::vector<float> base_new(last_n * bp.dim);
    int cnt = 0;
    for (int i = 0; i < last_n; ++i) {
        const faiss::idx_t id = h_compact_ids[i];
        if (h_hits[id] <= PC.knn_max_hit) {
            std::memcpy(base_new.data() + cnt * bp.dim, base_compact.data() + i * bp.dim,
                        bp.dim * sizeof(float));
            // base_new.insert(base_new.end(), base_compact.data() + i * bp.dim,
            //                 base_compact.data() + (i + 1) * bp.dim);
            h_compact_ids[cnt++] = id;
        }
    }
    base_new.resize(cnt * bp.dim);
    base_compact.swap(base_new);
    last_n = cnt;
    cudaMemcpy(d_compact_ids, h_compact_ids, sizeof(faiss::idx_t) * cnt, cudaMemcpyHostToDevice);
    myfaiss->Rebuild(base_compact.data(), h_compact_ids, last_n);
    fo.iprint("myfaiss rebuild with n=" + TOS(cnt) + ", time=" + TOS(time_diff(rs)) + "s");
    return true;
}
void GPUFuncs::count_hits(const int* d_knns, int batch) {
    hits_count_kernel<<<batch, 128>>>(d_knns, batch, bp.k, d_hits, PC.knn_max_hit);
    cudaGetLastError();
}
int* GPUFuncs::knn_compute_faiss_hd(const float* queries, int batch, float& time_s) {
    auto t0 = now_time();

    assert(bp.base_n <= GPU_N);
    myfaiss->Search(queries, knn_res_faiss.data().get(), knn_dists_res.data().get(), batch);

    hits_update_kernel<<<batch, 128>>>(knn_res_faiss.data().get(), d_compact_ids, batch, bp.k,
                                       d_knn_res.data().get(), d_hits, d_over_cnt, PC.knn_max_hit);
    cudaGetLastError();
    cudaMemcpy(h_over_cnt, d_over_cnt, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    static uint32_t last_cnt = 0;
    const uint32_t over_cnt = *h_over_cnt;
    if (over_cnt - last_cnt >= PC.knn_rebuild_thres) {
        fo.print("Rebuilding faiss index...");
        if (!rebuild()) fo.eprint("Rebuild failed!");
        last_cnt = over_cnt;
    }

    static int cnt = 0;
    static float total_time = 0.0f;
    total_time += time_diff(t0);
    if ((++cnt) % 50 == 0) {
        fo.print("knn compute faiss avg time: " + TOS(total_time / cnt) +
                 "s, over_cnt=" + TOS(over_cnt));
    }

    return d_knn_res.data().get();
}
void GPUFuncs::get_top_hit_nodes(int* d_top_ids, int topn) {
    int* d_ids = nullptr;
    std::vector<int> ids(bp.base_n);
    std::iota(ids.begin(), ids.end(), 0);
    cudaMalloc(&d_ids, bp.base_n * sizeof(int));
    cudaMemcpy(d_ids, ids.data(), bp.base_n * sizeof(int), cudaMemcpyHostToDevice);

    sort_pairs_desc(d_hits, d_ids, bp.base_n);
    cudaMemcpy(d_top_ids, d_ids, topn * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaFree(d_ids);

    if (PC.calc_hit) {
        std::ofstream hit_stat_file(fpath("../logs/hit_stat") / (TOS(PC.knn_rebuild) + ".txt"));
        std::vector<uint32_t> hits(bp.base_n);
        float avg_hit = 0.0f, var = 0.0f;
        cudaMemcpy(hits.data(), d_hits, bp.base_n * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        hit_stat_file << bp.base_n << std::endl;
        for (int i = 0; i < bp.base_n; ++i) {
            hit_stat_file << hits[i] << " ";
            avg_hit += hits[i];
        }
        avg_hit /= bp.base_n;
        hit_stat_file << avg_hit << std::endl;
        for (int i = 0; i < bp.base_n; ++i) {
            var += (hits[i] - avg_hit) * (hits[i] - avg_hit);
        }
        var /= bp.base_n;
        hit_stat_file << var << std::endl;

        hit_stat_file.close();
    }
}

int* GPUFuncs::knn_compute_faiss(const float* queries, int batch, float& time_s) {
    auto t0 = now_time();
    myfaiss->Search(queries, knn_res_faiss.data().get(), knn_dists_res.data().get(), batch, bp.k);
    // thrust::transform(knn_res_faiss.begin(), knn_res_faiss.begin() + batch * bp.k,
    //                   d_knn_res.begin(),
    //                   [] __host__ __device__(faiss::idx_t x) { return static_cast<int>(x); });
    hits_update_kernel<<<batch, 128>>>(knn_res_faiss.data().get(), batch, bp.k,
                                       d_knn_res.data().get(), d_hits);
    static int cnt = 0;
    static float total_time = 0.0f;
    total_time += time_diff(t0);
    if ((++cnt) % 50 == 0) {
        // std::string str;
        // std::vector<int> res(bp.k);
        // cudaMemcpy(res.data(), d_knn_res.data().get(), bp.k * sizeof(int),
        // cudaMemcpyDeviceToHost); for (int i : res) str += TOS(i) + " ";
        fo.print("knn compute faiss avg time: " + TOS(total_time / cnt) + "s");
    }
    return d_knn_res.data().get();
}

void GPUFuncs::knn_compute_faiss(const float* queries, int* knn_res, int batch, float& time_s) {
    thrust::device_ptr<int> knn_res_ptr(knn_res);
    myfaiss->Search(queries, knn_res_faiss.data().get(), knn_dists_res.data().get(), batch, bp.k);
    thrust::transform(knn_res_faiss.begin(), knn_res_faiss.begin() + batch * bp.k, knn_res_ptr,
                      [] __host__ __device__(faiss::idx_t x) { return static_cast<int>(x); });
}

const int* GPUFuncs::knn_compute(const float* queries, int batch, int& index_size) {
    auto s = now_time();

    assert(bp.base_n > GPU_N);
    float iso_ratio;
    const auto* res = ivf_builder->search(queries, batch, true, &iso_ratio);

    static float total_time = 0.0f;
    static int cnt = 0;
    total_time += time_diff(s);
    if ((++cnt) % 50 == 0)
        fo.print("IVF search avg time: " + TOS(total_time / cnt) + "s, over_cnt=" +
                 TOS(ivf_builder->get_over_cnt()) + ", iso_ratio=" + TOS(iso_ratio));

    index_size = ivf_builder->index_size();
    return res;
    // return ivf_builder->calc_queries_knns(queries, batch);
}
// void GPUFuncs::knn_compute(const float* queries, int batch, int* knn_res) {
//     assert(bp.base_n > GPU_N);
//     ivf_builder->search(queries, knn_res, batch);
//     // knn_res = ivf_builder->calc_queries_knns(queries, batch);
// }

void GPUFuncs::ivf_build() {
    if (PC.ivf_rebuild)
        ivf_builder->prepare();
    else if (!ivf_builder->load())
        ivf_builder->prepare();
}

void GPUFuncs::knn_prepare(bool have_gt, int batch) {
    if (have_gt) {
        cudaMalloc(&d_hits, bp.base_n * sizeof(uint32_t));
        cudaMemset(d_hits, 0, bp.base_n * sizeof(uint32_t));
        return;
    }
    d_knn_res.resize(batch * bp.k);
    if (bp.base_n <= GPU_N) {
        myfaiss = new MyFaiss(bp.dim, bp.k, bp.metric);
        myfaiss->Add(h_base, bp.base_n);
        knn_res_faiss.resize(batch * bp.k);
        knn_dists_res.resize(batch * bp.k);

        cudaMalloc(&d_hits, bp.base_n * sizeof(uint32_t));
        cudaMemset(d_hits, 0, bp.base_n * sizeof(uint32_t));
        cudaMallocHost(&h_hits, bp.base_n * sizeof(uint32_t));
        if (PC.knn_rebuild) {
            cudaMalloc(&d_compact_ids, bp.base_n * sizeof(faiss::idx_t));
            cudaMalloc(&d_over_cnt, sizeof(uint32_t));
            cudaMallocHost(&h_compact_ids, bp.base_n * sizeof(faiss::idx_t));
            cudaMallocHost(&h_over_cnt, sizeof(uint32_t));
            cudaMemset(d_over_cnt, 0, sizeof(uint32_t));

            std::vector<faiss::idx_t> arr(bp.base_n);
            std::iota(arr.begin(), arr.end(), 0);
            std::memcpy(h_compact_ids, arr.data(), bp.base_n * sizeof(faiss::idx_t));
            cudaMemcpy(d_compact_ids, h_compact_ids, bp.base_n * sizeof(faiss::idx_t),
                       cudaMemcpyHostToDevice);
            rebuild();
        }
        // cudaMalloc(&d_hot_deleted, ((bp.base_n + 31) / 32) * sizeof(uint32_t));
    } else {
        // hits.resize(bp.base_n);
        // ExactKnnStreamerConfig config;
        // config.dim = bp.dim;
        // config.k = bp.k;
        // config.metric = bp.metric;
        // config.max_qbatch = PC.batch;
        // knn_streamer = new ExactKnnStreamer(h_base, bp.base_n, config);
        cudaMalloc(&d_queries, sizeof(float) * batch * bp.dim);
        auto graph_dir = graph->get_graph_dir();
        ivf_builder = new IVFBuilder(
            IVFBuilder::Config(graph_dir, bp.base_n, bp.dim, bp.k, h_base, base_file),
            std::max(PC.top_hit_num, PC.search_start_nodes_num), "build");
    }
}

void GPUFuncs::knn_free() {
    thrust::device_vector<int>().swap(d_knn_res);
    if (knn_res_faiss.size()) thrust::device_vector<faiss::idx_t>().swap(knn_res_faiss);
    if (knn_dists_res.size()) thrust::device_vector<float>().swap(knn_dists_res);

    if (myfaiss) delete myfaiss;
    if (knn_streamer) delete knn_streamer;
    if (d_queries) cudaFree(d_queries);

    if (d_compact_ids) cudaFree(d_compact_ids);
    if (d_over_cnt) cudaFree(d_over_cnt);
    if (h_compact_ids) cudaFreeHost(h_compact_ids);
    if (h_hits) cudaFreeHost(h_hits);
    if (h_over_cnt) cudaFreeHost(h_over_cnt);
}

}  // namespace efanna2e
