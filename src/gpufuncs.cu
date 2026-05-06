#include <curand_kernel.h>

#include <cfloat>
#include <iostream>
#include <string>
#include <utility>

#include "gpufuncs.cuh"
#include "uni.h"
#include "utils.cuh"

namespace efanna2e {
// using CN = Candidate_Neighbor;

bool cuda_check_last_error(std::string func_name) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fo.eprint("Kernel error in " + func_name + ": " + cudaGetErrorString(err));
        return true;
    }
    return false;
}

GPUFuncs::GPUFuncs(Graph* graph, const fpath& base_file, const float* h_base,
                   const float* h_queries, int base_n, int query_n, int dim, int k, int max_degree,
                   DIST_METRIC m, int beam_capacity, Mode mode, bool have_gt)
    : construct(true),
      graph(graph),
      bp(dim, max_degree, base_n, query_n, k, beam_capacity, m),
      h_base(h_base),
      h_queries(h_queries),
      base_file(base_file),
      mode_(mode) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);              // 获取 GPU 设备
    shared_mem_per_block = prop.sharedMemPerBlock;  // GPU 最大共享内存大小
    // fo.print("Compute Capability: " + std::to_string(prop.major) + " " +
    //          std::to_string(prop.minor));
    fo.print("Max shared memory per block: " + std::to_string(shared_mem_per_block) + " bytes");
    // fo.print("Max shared memory permultiprocessor: " +
    //          std::to_string(prop.sharedMemPerMultiprocessor) + " bytes");
    // if (!prop.deviceOverlap)
    //     fo.eprint("Device does not support overlap, multi-stream may not improve performance.");
    // else
    //     fo.print("Device supports overlap, multi-stream may improve performance.");

    for (auto name : stream_names) {
        streams.emplace(name, MyStream());
        CUDA_CHECK(cudaStreamCreate(&streams[name].stream));
        CUDA_CHECK(cudaEventCreate(&streams[name].start));
        CUDA_CHECK(cudaEventCreate(&streams[name].stop));
    }
    cudaMalloc(&d_total_degree, sizeof(ull));
    cudaMallocHost(&h_total_degree, sizeof(ull));
    cudaMalloc(&d_noniso_num, sizeof(ull));
    cudaMallocHost(&h_noniso_num, sizeof(ull));
    // query_data_kmeans(h_queries);
    if (mode == Mode::CONS) {
        knn_prepare(have_gt);                 // knn 预分配内存
        upd_prepare(construct, PC.batch, k);  // upd 预分配内存
    } else if (mode == Mode::CE) {
        knn_prepare(have_gt);
        ivf_builder->load_hits();
    }

    // search_prepare();        // search 预分配内存
    fo.iprint(bp.str());
}

GPUFuncs::GPUFuncs(Graph* graph, QueryIndex* qindex, const float* h_base, int base_n, int dim,
                   int k, int max_degree, DIST_METRIC m, int beam_capacity, Mode mode)
    : construct(false),
      graph(graph),
      query_index(qindex),
      //   d_b(bp.gpu_n * bp.dim),
      bp(dim, max_degree, base_n, 0, k, beam_capacity, m),
      h_base(h_base),
      mode_(mode) {
    int deviceCount = 0, curDev = -1;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess || deviceCount == 0) return;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);              // 获取 GPU 设备
    shared_mem_per_block = prop.sharedMemPerBlock;  // GPU 最大共享内存大小
    fo.print("Compute Capability: " + std::to_string(prop.major) + " " +
             std::to_string(prop.minor));
    fo.print("Max shared memory per block: " + std::to_string(shared_mem_per_block) + " bytes");
    fo.print("Max shared memory permultiprocessor: " +
             std::to_string(prop.sharedMemPerMultiprocessor) + " bytes");
    if (!prop.deviceOverlap)
        fo.eprint("Device does not support overlap, multi-stream may not improve performance.");
    else
        fo.print("Device supports overlap, multi-stream may improve performance.");

    for (auto name : stream_names) {
        streams.emplace(name, MyStream());
        CUDA_CHECK(cudaStreamCreate(&streams[name].stream));
        // CUDA_CHECK(cudaEventCreate(&streams[name].start));
        // CUDA_CHECK(cudaEventCreate(&streams[name].stop));
    }

    cudaMalloc(&d_total_degree, sizeof(ull));
    cudaMallocHost(&h_total_degree, sizeof(ull));
    cudaMalloc(&d_noniso_num, sizeof(ull));
    cudaMallocHost(&h_noniso_num, sizeof(ull));

    if (mode == Mode::UPD) {
        // upd_prepare(construct, PC.insert_batch, k + 1);  // upd 预分配内存
        update_prepare();
        cudaMemcpy(d_total_degree, graph->get_total_degree(), sizeof(ull), cudaMemcpyHostToDevice);
    } else {
        knn_prepare(false);
        cudaMemset(d_total_degree, 0, sizeof(ull));
    }

    fo.iprint(bp.str());
}

GPUFuncs::~GPUFuncs() {
    if (d_hits) cudaFree(d_hits);
    if (ivf_builder) delete ivf_builder;

    search_free();
    if (mode_ == Mode::UPD) update_free();

    cudaFreeHost(h_total_degree);
    cudaFree(d_total_degree);
    cudaFreeHost(h_noniso_num);
    cudaFree(d_noniso_num);

    for (auto& pair : streams) {
        CUDA_CHECK(cudaStreamDestroy(pair.second.stream));
        CUDA_CHECK(cudaEventDestroy(pair.second.start));
        CUDA_CHECK(cudaEventDestroy(pair.second.stop));
    }
}

Query_KNNs::Query_KNNs(const std::vector<int>& knns_in, int query_size, BP bp)
    : size(query_size), knns(query_size), del_flag(bp.base_n, 0), bp(bp) {
    if (!PC.upd_type) query_of_base.assign(bp.base_n * 2, -1);
    for (int i = 0; i < query_size; ++i) {
        knns[i].assign(knns_in.begin() + i * bp.k, knns_in.begin() + (i + 1) * bp.k);
        if (!PC.upd_type) {
            const int id = knns[i][0];
            if (query_of_base[id] == -1) query_of_base[id] = i;
        }
    }
}

Query_KNNs::Query_KNNs(Cache& cache, int query_size, BP bp) : del_flag(bp.base_n, 0), bp(bp) {
    if (!PC.upd_type) query_of_base.assign(bp.base_n * 2, -1);

    knns.resize(query_size);
    const auto& results = cache.results;
    int qid = 0;
    for (const auto& res : results) {
        assert(res.knn.size() == bp.k * res.batch);
        auto it = res.knn.begin();
        for (int i = 0; i < res.batch; ++i) {
            const int id = qid++;
            knns[id].assign(it, it + bp.k);
            it += bp.k;
            if (!PC.upd_type)
                for (int j = 0; j < bp.k; j++) {
                    const int nid = knns[id][j];
                    if (query_of_base[nid] == -1) query_of_base[nid] = id;
                }
            if (qid == query_size) break;
        }
        if (qid == query_size) break;
    }
}

void Query_KNNs::get(const std::vector<int>& qids, std::vector<int>& knns_out,
                     std::vector<int>& offsets) {
    const int qn = qids.size();
    knns_out.clear();
    knns_out.reserve(qn * bp.k);
    offsets.resize(qn);
    for (int i = 0; i < qn; ++i) {
        const int qid = qids[i];
        knns_out.insert(knns_out.end(), knns[qid].begin(), knns[qid].end());
        offsets[i] = !i ? knns[qid].size() : offsets[i - 1] + knns[qid].size();
    }
}

void Query_KNNs::get(const faiss::idx_t* ids, int* knns_out, int n) {
    for (int i = 0; i < n; ++i) {
        const int id = ids[i];
        assert(knns[id].size() == bp.k);
        std::memcpy(knns_out + i * bp.k, knns[id].data(), bp.k * sizeof(int));
    }
}

void Query_KNNs::get_and_add(const std::vector<int>& qids, const int* new_ids,
                             std::vector<int>& knns_out, std::vector<int>& offsets, int N,
                             int query_per_ins) {
    const size_t max_cnt = bp.k * query_per_ins;
    knns_out.clear();
    knns_out.reserve(N * query_per_ins * bp.k);
    offsets.resize(N + 1);
    offsets[0] = 0;

    std::unique_lock<std::shared_mutex> lock(mtx);

    for (int i = 0; i < N; ++i) {
        std::vector<int> cand_knns;
        cand_knns.reserve(max_cnt * 2);
        for (int j = 0; j < query_per_ins; ++j) {
            const int qid = qids[i * query_per_ins + j];
            std::vector<int> new_knns;
            new_knns.reserve(knns[qid].size());
            for (int id : knns[qid])
                if (!del_flag[id]) {
                    cand_knns.push_back(id);
                    new_knns.push_back(id);
                }
            if (new_knns.size() < knns[qid].size()) knns[qid] = std::move(new_knns);
        }
        std::sort(cand_knns.begin(), cand_knns.end());
        cand_knns.erase(std::unique(cand_knns.begin(), cand_knns.end()), cand_knns.end());
        knns_out.insert(
            knns_out.end(), cand_knns.begin(),
            cand_knns.size() < max_cnt ? cand_knns.end() : cand_knns.begin() + max_cnt);

        offsets[i + 1] = knns_out.size();
    }
    // for (int i = 0; i < N; ++i) {
    //     std::unordered_set<int> cand_knns;
    //     for (int j = 0; j < query_per_ins; ++j) {
    //         const int qid = qids[i * query_per_ins + j];
    //         for (int id : knns[qid])
    //             if (!del_flag[id]) cand_knns.insert(id);
    //     }
    //     size_t cnt = 0;
    //     for (int id : cand_knns) {
    //         if (cnt >= max_cnt) break;
    //         knns_out.push_back(id);
    //         ++cnt;
    //     }
    //     offsets[i + 1] = knns_out.size();
    // }

    for (int i = 0; i < N; ++i) {
        const int new_id = new_ids[i];
        for (int j = 0; j < query_per_ins; ++j)
            knns[qids[i * query_per_ins + j]].push_back(new_id);
    }
}

void Query_KNNs::get_and_add(const std::vector<faiss::idx_t>& qids, const int* new_ids,
                             std::vector<int>& knns_out, size_t* offsets, int N, int query_num) {
    knns_out.clear();
    knns_out.reserve(N * query_num * bp.k);
    offsets[0] = 0;

    std::unique_lock<std::shared_mutex> lock(mtx);

    for (int i = 0; i < N; ++i) {
        const int new_id = new_ids[i];
        std::unordered_set<int> cand_knns;
        for (int j = 0; j < query_num; ++j) {
            const faiss::idx_t qid = qids[i * query_num + j];
            cand_knns.insert(knns[qid].begin(), knns[qid].end());
        }
        knns_out.insert(knns_out.end(), cand_knns.begin(), cand_knns.end());
        offsets[i + 1] = knns_out.size();
    }
    for (int i = 0; i < N; ++i) {
        const int new_id = new_ids[i];
        for (int j = 0; j < query_num; ++j) knns[qids[i * query_num + j]].push_back(new_id);
    }
}

void Query_KNNs::get_and_add(const std::vector<faiss::idx_t>& qids, const int* new_ids,
                             int* knns_out, int N) {
    std::unique_lock<std::shared_mutex> lock(mtx);

    for (int i = 0; i < N; ++i) {
        const int new_id = new_ids[i];
        const faiss::idx_t qid = qids[i];
        std::memcpy(knns_out + i * bp.k, knns[qid].data() + knns[qid].size() - bp.k,
                    bp.k * sizeof(int));
    }
    for (int i = 0; i < N; ++i) knns[qids[i]].push_back(new_ids[i]);
}

void Query_KNNs::get_and_add(const std::vector<faiss::idx_t>& qids, const int* new_ids,
                             std::vector<int>& knns_out, size_t* offsets, int N) {
    knns_out.clear();
    knns_out.reserve(N * bp.k);
    offsets[0] = 0;

    std::unique_lock<std::shared_mutex> lock(mtx);

    for (int i = 0; i < N; ++i) {
        const int new_id = new_ids[i];
        const faiss::idx_t qid = qids[i];
        knns_out.insert(knns_out.end(), knns[qid].begin(), knns[qid].end());
        offsets[i + 1] = knns_out.size();
    }
    // for (int i = 0; i < N; ++i) knns[qids[i]].push_back(new_ids[i]);
}

};  // namespace efanna2e