#include "index_gpu.h"

#include <nvml.h>
#include <omp.h>

#include <bitset>
#include <cfloat>
#include <chrono>
#include <condition_variable>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <queue>
#include <random>
#include <shared_mutex>
#include <string>
#include <thread>

#include "candidates.h"
#include "efanna2e/exceptions.h"
#include "efanna2e/parameters.h"
#include "fileout.h"
#include "test.h"

using namespace efanna2e;

// 构造函数，初始化索引管道
IndexGPU::IndexGPU(const size_t dimension, const size_t n, Metric m, Index* initializer)
    : Index(dimension, n, m), initializer_{initializer} {
    if (m == efanna2e::COSINE) need_normalize = true;
}

IndexGPU::~IndexGPU() {}

void IndexGPU::Build(size_t n, const float* data, const Parameters& parameters) {};

void IndexGPU::Search(const float* query, const float* x, size_t k, const Parameters& parameters,
                      unsigned* indices, float* res_dists) {};

void IndexGPU::BuildGPU(size_t n_sq, float* sq_data, size_t n_bp, float* bp_data,
                        Parameters& parameters) {
    SetParameters(parameters);
    auto s = std::chrono::high_resolution_clock::now();
    int M_pjbp = parameters.Get<int>("M_pjbp");
    k = parameters.Get<int>("k");
    metric = parameters.Get<std::string>("dist") == "l2" ? DIST_METRIC::L2 : DIST_METRIC::IP;
    std::string cache_file = parameters.Get<std::string>("cache_file");
    std::string graph_file = parameters.Get<std::string>("graph_file");
    fo.Init(parameters.Get<std::string>("log_file"));
    omp_set_num_threads(parameters.Get<int>("num_threads"));
    data_bp_ = bp_data;
    data_sq_ = sq_data;
    nd_ = n_bp;
    nd_sq_ = n_sq;
    total_pts_ = nd_ + nd_sq_;
    u32_nd_ = static_cast<int>(nd_);
    u32_nd_sq_ = static_cast<int>(nd_sq_);

#ifndef GIG
    projection_graph_.resize(u32_nd_);
    projection_graph_deg_.resize(u32_nd_, 0);
    projection_graph_dist_.resize(u32_nd_);
#endif

    GPUPrepare();

    Test test(gpufuncs, h_queries, h_base, dimension_, u32_nd_, k, metric, M_pjbp);
    test.Run();

    test_search_prepare(gpufuncs);

    KNN_Queue knn_queue(k, QUEUE_SIZE, gpufuncs);

    Test_Result tres;
    Cache cache(cache_file, graph_file, gpufuncs, projection_graph_, projection_graph_deg_,
                projection_graph_dist_, nonisolated_num, batch_id, total_degree, total_time, tres,
                k, nd_);

    start_time = std::chrono::high_resolution_clock::now();

    std::thread knn_task([this, &knn_queue, &cache] { this->KNNTask(knn_queue, cache); });

    std::thread graph_task([this, &knn_queue, &cache] { this->GraphTask(knn_queue, cache); });

    knn_task.join();
    graph_task.join();

    CUDA_CHECK(cudaDeviceSynchronize());

    fo.print("All done. Copying graph from GPU. Total time: " + TOS(compute_duration_time()));

    GPUFree();

    // stats projection graph degree
    uint64_t total_degree = 0;
    int max_deg = 0;
    int min_deg = std::numeric_limits<int>::max();
    for (int i = 0; i < u32_nd_; ++i) {
        const int deg = projection_graph_deg_[i];
        for (int j = 0; j < deg; ++j)
            assert(projection_graph_[i][j] >= 0 && projection_graph_[i][j] < u32_nd_);

        max_deg = std::max(max_deg, deg);
        min_deg = std::min(min_deg, deg);

        total_degree += deg;
    }
    fo.print("total degree: " + TOS(total_degree) + "/" + TOS(u32_nd_));
    fo.print("After projection, average degree of projection graph: " +
             TOS(total_degree * 1.0 / u32_nd_));
    fo.print("After projection, max degree of projection graph: " + TOS(max_deg));
    fo.print("After projection, min degree of projection graph: " + TOS(min_deg));

    CalculateGraphEP();
}

void IndexGPU::GPUPrepare() {
    auto max_degree = parameters_.Get<int>("M_pjbp");
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess || deviceCount == 0) {
        fo.eprint("No CUDA-capable GPU found or CUDA driver not installed.");
    }
    fo.print("Found " + TOS(deviceCount) + " CUDA-capable device(s).");

    int curDev = -1;
    cudaGetDevice(&curDev);
    fo.print("Current device (cudaGetDevice) = " + TOS(curDev));

    // host 内存分配和传输
    CUDA_CHECK(cudaMallocHost(&h_base, u32_nd_ * dimension_ * sizeof(float)));
    memcpy(h_base, data_bp_, u32_nd_ * dimension_ * sizeof(float));
    CUDA_CHECK(cudaMallocHost(&h_queries, u32_nd_sq_ * dimension_ * sizeof(float)));
    memcpy(h_queries, data_sq_, u32_nd_sq_ * dimension_ * sizeof(float));

    gpufuncs = new GPUFuncs(h_base, h_queries, u32_nd_, u32_nd_sq_, dimension_, k, max_degree,
                            metric, TEST_SEARCH_L);

    fo.print("GPU prepare...");
}

void IndexGPU::KNNTask(KNN_Queue& knn_queue, Cache& cache) {
    processed = 0;
    const int old_batch_id = batch_id;
    for (int i = 0; i < cache.results.size(); i++) {
        auto& res = cache.results[i];

        if (i <= old_batch_id) {
            processed += res.batch;
            continue;
        }

        if (knn_queue.WriteTail(res.knn.data(), res.batch, batch_id, KNNType::host)) {
            processed += res.batch;
            batch_id++;
        } else if (stop_flag.load())
            break;
        else
            i--;
    }

    fo.iprint("Cache loaded with processed: " + TOS(processed / 1000000.0) +
              " M, old_batch_id: " + TOS(old_batch_id) + ". Processing new data...");

    while (!stop_flag.load() && processed < u32_nd_sq_) {
        int batch = std::min<int>(BATCH, u32_nd_sq_ - processed);
        // 计算 batch 查询的 knn
        float time_ms;
        int* d_knn = gpufuncs->knn_compute(batch, processed, time_ms);

        BatchResult res = cache.WriteGTCache(d_knn, batch_id, batch, time_ms);

        while (!stop_flag.load() && !knn_queue.WriteTail(d_knn, batch, batch_id, KNNType::device))
            continue;

        processed += batch;
        batch_id++;
    }
    fo.print("KNNTask finished...");
}

void IndexGPU::GraphTask(KNN_Queue& knn_queue, Cache& cache) {
    auto max_degree = parameters_.Get<int>("M_pjbp");
    std::vector<std::shared_mutex> graph_mtx(u32_nd_);

    std::thread worker(&IndexGPU::update_graph, this, std::ref(cache), std::ref(graph_mtx),
                       std::ref(knn_queue));

    int *d_knn_idxs, *d_old_nbrs, *d_old_nbr_nums;
    float* d_old_nbr_dists;
    CUDA_CHECK(cudaMalloc(&d_knn_idxs, BATCH * k * sizeof(int)));

    int old_batch_id = 0;
    while (!stop_flag.load()) {
        CUDA_CHECK(cudaMemset(d_knn_idxs, -1, BATCH * k * sizeof(int)));

        int batch_id;
        int batch = knn_queue.ReadHead(d_knn_idxs, batch_id);
        while (!stop_flag.load() && batch == -1) batch = knn_queue.ReadHead(d_knn_idxs, batch_id);

        if (stop_flag.load()) break;

#ifndef GIG
        CUDA_CHECK(cudaMalloc(&d_old_nbrs, BATCH * k * max_degree * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_old_nbr_dists, BATCH * k * max_degree * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_old_nbr_nums, BATCH * k * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&h_old_nbr_nums, BATCH * k * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&h_knn_idxs, BATCH * k * sizeof(int)));

        get_old_nbrs(d_knn_idxs, d_old_nbrs, d_old_nbr_dists, d_old_nbr_nums, batch, max_degree,
                     graph_mtx);
        auto res = gpufuncs->handle_knn_updates(d_knn_idxs, d_old_nbrs, d_old_nbr_dists,
                                                d_old_nbr_nums, batch);
        Graph_Update_Info gui(d_knn_idxs, res.first, res.second, k, batch, batch_id, max_degree);
        {
            std::unique_lock<std::shared_mutex> lock(queue_mtx);
            graph_update_queue.push(std::move(gui));
        }
#else
        GSD gs = gpufuncs->handle_knn_updates(d_knn_idxs, batch);
        fo.print("Graph Updated with noniso_ratio: " + TOS(1 - (float)gs.iso_num / u32_nd_) +
                 ", avg_degree: " + TOS((float)gs.total_degree / u32_nd_) +
                 ", batch_id: " + TOS(batch_id));
        if (batch_id - old_batch_id < 100) continue;
        {
            std::unique_lock<std::shared_mutex> lock(queue_mtx);
            old_batch_id = batch_id;
            gui.update(batch_id, gs);
        }
#endif
        cv.notify_one();
    }

#ifndef GIG
    if (worker.joinable()) worker.join();
    CUDA_CHECK(cudaFree(d_old_nbrs));
    CUDA_CHECK(cudaFree(d_old_nbr_dists));
    CUDA_CHECK(cudaFree(d_old_nbr_nums));
    CUDA_CHECK(cudaFreeHost(h_old_nbr_nums));
    CUDA_CHECK(cudaFreeHost(h_knn_idxs));
#endif

    CUDA_CHECK(cudaFree(d_knn_idxs));

    fo.print("GraphTask finished...");
}

#ifndef GIG
void IndexGPU::get_old_nbrs(int* d_knn_idxs, int* d_old_nbrs, float* d_old_nbr_dists,
                            int* d_old_nbr_nums, const int batch, const int max_degree,
                            std::vector<std::shared_mutex>& graph_mtx) {
    CUDA_CHECK(
        cudaMemcpy(h_knn_idxs, d_knn_idxs, batch * k * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemset(d_old_nbrs, -1, batch * k * max_degree * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_old_nbr_nums, 0, batch * k * sizeof(int)));

    int* h_old_nbrs;
    float* h_old_nbr_dists;
    cudaMallocHost(&h_old_nbrs, batch * k * max_degree * sizeof(int));
    cudaMallocHost(&h_old_nbr_dists, batch * k * max_degree * sizeof(float));

    for (int i = 0; i < batch * k; i++) {
        const int pivot_id = h_knn_idxs[i];
        if (pivot_id < 0 || pivot_id >= u32_nd_) fo.eprint("Invalid pivot_id: " + TOS(pivot_id));

        std::shared_lock<std::shared_mutex> lock(graph_mtx[pivot_id]);
        int size = projection_graph_deg_[pivot_id];
        h_old_nbr_nums[i] = size;
        if (!size) continue;

        for (int j = 0; j < size; j++) {
            if (projection_graph_[pivot_id][j] < 0)
                fo.eprint("Invalid projection_graph_: " + TOS(projection_graph_[pivot_id][j]) +
                          " pivot_id: " + TOS(pivot_id) + " j: " + TOS(j) + " size: " + TOS(size));

            h_old_nbrs[i * max_degree + j] = projection_graph_[pivot_id][j];
            h_old_nbr_dists[i * max_degree + j] = projection_graph_dist_[pivot_id][j];
        }
    }
    CUDA_CHECK(cudaMemcpy(d_old_nbrs, h_old_nbrs, batch * k * max_degree * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_old_nbr_dists, h_old_nbr_dists, batch * k * max_degree * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_old_nbr_nums, h_old_nbr_nums, batch * k * sizeof(int),
                          cudaMemcpyHostToDevice));
}
#endif

void IndexGPU::GPUFree() {
    CUDA_CHECK(cudaFreeHost(h_base));
    CUDA_CHECK(cudaFreeHost(h_queries));
    delete gpufuncs;
    fo.print("GPU free...");
}

float IndexGPU::compute_duration_time() {
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::high_resolution_clock::now() - start_time);
    return duration.count() + total_time;
}

std::pair<int, int> IndexGPU::SearchPipe(const float* query, size_t k, size_t& qid,
                                         const Parameters& parameters, int* indices,
                                         std::vector<float>& res_dists) {
    const int L_pq = parameters.Get<int>("L_pq");

    NeighborPriorityQueue search_queue(L_pq);

    prefetch_vector((char*)(data_bp_ + projection_ep_ * dimension_), dimension_);
    VisitedList* vl = visited_list_pool_->getFreeVisitedList();
    vl_type* visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    float distance =
        distance_->compare(data_bp_ + projection_ep_ * dimension_, query, (unsigned)dimension_);
    search_queue.insert(Neighbor(projection_ep_, distance, false));

    int cmps = 0;
    int hops = 0;
    while (search_queue.has_unexpanded_node()) {
        auto cur_id = search_queue.closest_unexpanded().id;

        int* cur_nbrs = projection_graph_[cur_id].data();
        ++hops;
        // get neighbors' neighbors, first
        for (size_t j = 0; j < projection_graph_deg_[cur_id];
             ++j) {  // current check node's neighbors
            int nbr = projection_graph_[cur_id][j];
            _mm_prefetch((char*)(visited_array + *(cur_nbrs + j + 1)), _MM_HINT_T0);
            _mm_prefetch((char*)(data_bp_ + *(cur_nbrs + j + 1) * dimension_), _MM_HINT_T0);
            if (visited_array[nbr] != visited_array_tag) {
                visited_array[nbr] = visited_array_tag;
                if (nbr < 0 || nbr >= nd_) fo.eprint("Error: nbr out of range: " + TOS(nbr));

                float distance =
                    distance_->compare(data_bp_ + nbr * dimension_, query, (unsigned)dimension_);

                ++cmps;
                search_queue.insert({nbr, distance, false});
            }
        }
    }

    visited_list_pool_->releaseVisitedList(vl);

    if (unlikely(search_queue.size() < k)) {
        std::stringstream ss;
        ss << "not enough results: " << search_queue.size() << ", expected: " << k;
        throw std::runtime_error(ss.str());
    }

    for (size_t i = 0; i < k; ++i) {
        indices[i] = search_queue[i].id;
        res_dists[i] = search_queue[i].distance;
    }
    return std::make_pair(cmps, hops);
}

void IndexGPU::CalculateGraphEP() {
    float* center = new float[dimension_]();
    memset(center, 0, sizeof(float) * dimension_);
    // calculate centroid in base point
    for (size_t i = 0; i < nd_; ++i) {
        for (size_t d = 0; d < dimension_; ++d) {
            center[d] += data_bp_[i * dimension_ + d];
        }
    }

    for (size_t d = 0; d < dimension_; ++d) {
        center[d] /= (float)nd_;
    }

    float* distances = new float[nd_]();
    memset(distances, 0, sizeof(float) * nd_);
#pragma omp parallel for
    for (size_t i = 0; i < nd_; ++i) {
        const float* cur_data = data_bp_ + i * dimension_;
        float diff = 0;
        for (size_t j = 0; j < dimension_; ++j) {
            diff += ((center[j] - cur_data[j]) * (center[j] - cur_data[j]));
        }
        distances[i] = diff;
    }

    int closest = 0;
    for (size_t i = 1; i < nd_; ++i) {
        if (projection_graph_[i].size() > 0 &&
            distances[i] < distances[closest]) {  // 孤立节点不可为质心
            closest = static_cast<int>(i);
        }
    }
    projection_ep_ = closest;
    delete[] center;
    delete[] distances;
}

void IndexGPU::SaveIndex(const char* filename) {
    std::ofstream out(filename, std::ios::binary | std::ios::out);
    if (!out.is_open()) {
        throw std::runtime_error("cannot open file");
    }
    out.write((char*)&projection_ep_, sizeof(int));
    out.write((char*)&u32_nd_, sizeof(int));
    for (int i = 0; i < u32_nd_; ++i) {
        int nbr_size = projection_graph_deg_[i];
        out.write((char*)&nbr_size, sizeof(int));
        out.write((char*)projection_graph_[i].data(), sizeof(int) * nbr_size);
    }
    out.close();
}

void IndexGPU::LoadVectorData(const char* base_file, const char* sampled_query_file) {
    int base_num = 0, sq_num = 0, base_dim = 0, q_dim = 0;

    load_meta<float>(base_file, base_num, base_dim);
    if (strlen(sampled_query_file) != 0) {
        load_meta<float>(sampled_query_file, sq_num, q_dim);
        if (base_dim != q_dim) {
            throw std::runtime_error("base and query dimension mismatch");
        }
    }
    float* base_data = nullptr;
    float* sampled_query_data = nullptr;
    load_data<float>(base_file, base_num, base_dim, base_data);

    if (need_normalize) {
        for (size_t i = 0; i < base_num; ++i) {
            normalize<float>(base_data + i * (uint64_t)base_dim, (uint64_t)base_dim);
        }
    }

    data_bp_ = data_align(base_data, base_num, base_dim);

    nd_ = base_num;
    nd_sq_ = sq_num;
    total_pts_ = nd_ + nd_sq_;
    u32_nd_ = static_cast<int>(nd_);
    u32_nd_sq_ = static_cast<int>(nd_sq_);
}

void IndexGPU::LoadGraph(const char* filename) {
    // load graph to projection graph
    std::ifstream in(filename, std::ios::binary);
    int npts;
    in.read((char*)&projection_ep_, sizeof(int));
    in.read((char*)&npts, sizeof(npts));
    projection_graph_.resize(npts);
    projection_graph_deg_.resize(npts);
    int out_degree = 0;
    for (int i = 0; i < npts; i++) {
        int nbr_size;
        in.read((char*)&nbr_size, sizeof(nbr_size));
        out_degree += nbr_size;
        projection_graph_deg_[i] = nbr_size;
        projection_graph_[i].resize(nbr_size);
        in.read((char*)projection_graph_[i].data(), nbr_size * sizeof(int));
    }
    fo.print("Projection graph, avg_degree: " + TOS(out_degree * 1.0 / npts));
    in.close();
}