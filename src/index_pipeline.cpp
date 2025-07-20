#include <omp.h>
#include <bitset>
#include <chrono>
#include <ctime>
#include <thread>
#include <condition_variable>
#include <iostream>
#include <queue>
#include <string>
#include <shared_mutex>

#include "efanna2e/exceptions.h"
#include "efanna2e/parameters.h"
#include "candidates.h"
#include "fileout.h"
#include "index_pipeline.h"

// define likely unlikely
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#define PROJECTION_SLACK 2
#define STREAMS 4

using namespace efanna2e;

// 构造函数，初始化索引管道
IndexPipeline::IndexPipeline(const size_t dimension, const size_t n, Metric m, Index *initializer)
    : Index(dimension, n, m), initializer_{initializer}
{
    if (m == efanna2e::COSINE)
        need_normalize = true;
}

IndexPipeline::~IndexPipeline() {}

void IndexPipeline::Build(size_t n, const float *data, const Parameters &parameters) {};

void IndexPipeline::Search(const float *query, const float *x, size_t k, const Parameters &parameters, unsigned *indices, float *res_dists) {};

void IndexPipeline::BuildPipeline(size_t n_sq, float *sq_data, size_t n_bp, float *bp_data,
                                  Parameters &parameters)
{
    std::cout << "__cplusplus: " << __cplusplus << std::endl;
    SetParameters(parameters);
    auto s = std::chrono::high_resolution_clock::now();
    uint32_t M_pjbp = parameters.Get<uint32_t>("M_pjbp");
    k = parameters.Get<uint32_t>("k");
    std::string cache_file = parameters.Get<std::string>("cache_file");
    fo.Init(parameters.Get<std::string>("log_file"));
    omp_set_num_threads(parameters.Get<uint32_t>("num_threads"));
    // omp_set_nested(1);
    // 新代码（允许2层嵌套）
    // omp_set_max_active_levels(2);
    data_bp_ = bp_data;
    data_sq_ = sq_data;
    nd_ = n_bp;
    nd_sq_ = n_sq;
    total_pts_ = nd_ + nd_sq_;
    u32_nd_ = static_cast<uint32_t>(nd_);
    u32_nd_sq_ = static_cast<uint32_t>(nd_sq_);
    learn_base_knn_.resize(u32_nd_sq_);

    ProjectionReserveSpace();

    GPUPrepare();

    KNN_Queue knn_queue(k, 10);
    GTCache cache(cache_file, k);
    std::thread knn_task([this, &knn_queue, &cache]
                         { this->KNNTask(knn_queue, cache); });
    std::thread graph_task([this, &knn_queue]
                           { this->GraphTask(knn_queue); });
    knn_task.join();
    graph_task.join();

    CUDA_CHECK(cudaDeviceSynchronize());
    fo.print("All done. Copying graph from GPU...");

    GPUFree();
    // stats projection graph degree
    float avg_degree = 0;
    uint64_t total_degree = 0;
    uint32_t max_degree = 0;
    uint32_t min_degree = std::numeric_limits<uint32_t>::max();
    for (uint32_t i = 0; i < u32_nd_; ++i)
    {
        if (projection_graph_[i].size() > max_degree)
        {
            max_degree = projection_graph_[i].size();
        }
        if (projection_graph_[i].size() < min_degree)
        {
            min_degree = projection_graph_[i].size();
        }
        avg_degree += static_cast<float>(projection_graph_[i].size());
        total_degree += projection_graph_[i].size();
    }
    fo.print("total degree: " + std::to_string(total_degree));
    avg_degree /= (float)u32_nd_;
    fo.print("After projection, average degree of projection graph: " + std::to_string(avg_degree));
    fo.print("After projection, max degree of projection graph: " + std::to_string(max_degree));
    fo.print("After projection, min degree of projection graph: " + std::to_string(min_degree));

    CalculateProjectionep();
}

void IndexPipeline::GPUPrepare()
{
    auto max_degree = parameters_.Get<uint32_t>("M_pjbp");
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess || deviceCount == 0)
    {
        fo.eprint("No CUDA-capable GPU found or CUDA driver not installed.");
    }
    fo.print("Found " + std::to_string(deviceCount) + " CUDA-capable device(s).");

    // host 内存分配和传输
    CUDA_CHECK(cudaMallocHost(&h_base, u32_nd_ * dimension_ * sizeof(float)));
    memcpy(h_base, data_bp_, u32_nd_ * dimension_ * sizeof(float));
    CUDA_CHECK(cudaMallocHost(&h_queries, u32_nd_sq_ * dimension_ * sizeof(float)));
    memcpy(h_queries, data_sq_, u32_nd_sq_ * dimension_ * sizeof(float));

    gpufuncs = new GPUFuncs(h_base, u32_nd_, dimension_, k, max_degree);
}

void IndexPipeline::KNNTask(KNN_Queue &knn_queue, GTCache &cache)
{
    uint32_t processed = 0;
    for (auto res : cache.results)
    {
        knn_queue.WriteTail(res.knn.data(), res.batch, KNNType::host);
        processed += BATCH;
    }
    while (processed < u32_nd_sq_)
    {
        {
            std::shared_lock<std::shared_mutex> lk(mtx);
            if (stop_flag)
                break;
        }
        fo.print("Processed: " + std::to_string(processed) + "/" + std::to_string(u32_nd_sq_));
        int batch = std::min<uint32_t>(BATCH, u32_nd_sq_ - processed);
        // 计算 batch 查询的 knn
        float time_ms;
        uint32_t *d_knn = gpufuncs->knn_compute(h_queries + processed * dimension_, batch, time_ms);
        cache.WriteCache(d_knn, batch, time_ms);
        knn_queue.WriteTail(d_knn, batch, KNNType::device);

        processed += batch;
    }
}

void IndexPipeline::GraphTask(KNN_Queue &knn_queue)
{
    auto max_degree = parameters_.Get<uint32_t>("M_pjbp");

    std::thread worker(&IndexPipeline::update_graph, this);

    uint32_t *d_knn_idxs;
    CUDA_CHECK(cudaMalloc(&d_knn_idxs, BATCH * k * sizeof(int)));
    while (true)
    {
        {
            std::shared_lock<std::shared_mutex> lk(mtx);
            if (stop_flag)
                break;
        }
        CUDA_CHECK(cudaMemset(d_knn_idxs, -1, BATCH * k * sizeof(int)));
        const int batch = knn_queue.ReadHead(d_knn_idxs);

        uint32_t *d_new_nbr_ids = gpufuncs->handle_knn_updates(d_knn_idxs, batch);
        Graph_Update_Info gui(d_knn_idxs, d_new_nbr_ids, k, batch, max_degree);
        {
            std::unique_lock<std::shared_mutex> lock(mtx);
            graph_update_queue.push(std::move(gui));
        }
        cv.notify_one();
    }
    {
        std::unique_lock<std::shared_mutex> lock(mtx);
        stop_flag = true;
    }
    cv.notify_one();

    if (worker.joinable())
        worker.join();

    CUDA_CHECK(cudaFree(d_knn_idxs));
}

void IndexPipeline::update_graph()
{
    static int nonisolated_num = 0;
    auto iso_thres = parameters_.Get<float>("iso_thres");
    auto max_degree = parameters_.Get<uint32_t>("M_pjbp");
    while (true)
    {
        Graph_Update_Info gui;
        {
            std::unique_lock<std::shared_mutex> lock(mtx);
            cv.wait(lock, [this]
                    { return !graph_update_queue.empty() || stop_flag; });

            if (stop_flag && graph_update_queue.empty())
                break;

            if (graph_update_queue.empty())
                continue;

            gui = std::move(graph_update_queue.front());
            graph_update_queue.pop();
        }

        for (int i = 0; i < gui.batch * k; i++)
        {
            const uint32_t pivot_id = gui.knn_ids[i];
            if (projection_graph_[pivot_id].size() == 0)
            {
                nonisolated_num++;
                projection_graph_[pivot_id].resize(max_degree);
            }
            memcpy(
                projection_graph_[pivot_id].data(),
                gui.new_nbr_ids.get() + i * max_degree,
                max_degree * sizeof(uint32_t));
            projection_graph_[pivot_id].erase(
                std::remove(
                    projection_graph_[pivot_id].begin(),
                    projection_graph_[pivot_id].end(),
                    -1),
                projection_graph_[pivot_id].end());
        }

        float iso_ratio = 100 * (float)nonisolated_num / (float)u32_nd_;
        fo.iprint("Graph updated with isolated ratio = " + std::to_string(iso_ratio) + "%");
        if (iso_ratio >= iso_thres)
        {
            std::unique_lock<std::shared_mutex> lock(mtx);
            stop_flag = true;
            fo.iprint("Isolated ratio below threshold, stopping.");
            return;
        }
    }
}

void IndexPipeline::GPUFree()
{
    CUDA_CHECK(cudaFreeHost(h_base));
    CUDA_CHECK(cudaFreeHost(h_queries));
    delete gpufuncs;
}

std::pair<uint32_t, uint32_t> IndexPipeline::SearchPipe(const float *query, size_t k, size_t &qid, const Parameters &parameters, unsigned *indices, std::vector<float> &res_dists)
{
    uint32_t L_pq = parameters.Get<uint32_t>("L_pq");
    NeighborPriorityQueue search_queue(L_pq);
    std::vector<uint32_t> init_ids;
    init_ids.push_back(projection_ep_);
    prefetch_vector((char *)(data_bp_ + projection_ep_ * dimension_), dimension_);
    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    for (auto &id : init_ids)
    {
        float distance = distance_->compare(data_bp_ + id * dimension_, query, (unsigned)dimension_);
        search_queue.insert(Neighbor(id, distance, false));
    }

    uint32_t cmps = 0;
    uint32_t hops = 0;
    while (search_queue.has_unexpanded_node())
    {
        auto cur_id = search_queue.closest_unexpanded().id;
        uint32_t *cur_nbrs = projection_graph_[cur_id].data();
        ++hops;
        // get neighbors' neighbors, first
        for (size_t j = 0; j < projection_graph_[cur_id].size(); ++j)
        { // current check node's neighbors
            uint32_t nbr = *(cur_nbrs + j);
            _mm_prefetch((char *)(visited_array + *(cur_nbrs + j + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(data_bp_ + *(cur_nbrs + j + 1) * dimension_), _MM_HINT_T0);
            if (visited_array[nbr] != visited_array_tag)
            {
                visited_array[nbr] = visited_array_tag;
                float distance = distance_->compare(data_bp_ + nbr * dimension_, query, (unsigned)dimension_);

                ++cmps;
                search_queue.insert({nbr, distance, false});
            }
        }
    }

    visited_list_pool_->releaseVisitedList(vl);

    if (unlikely(search_queue.size() < k))
    {
        std::stringstream ss;
        ss << "not enough results: " << search_queue.size() << ", expected: " << k;
        throw std::runtime_error(ss.str());
    }

    for (size_t i = 0; i < k; ++i)
    {
        indices[i] = search_queue[i].id;
        res_dists[i] = search_queue[i].distance;
    }
    return std::make_pair(cmps, hops);
}

void IndexPipeline::ProjectionReserveSpace()
{
    uint32_t M_pjbp = parameters_.Get<uint32_t>("M_pjbp");
    projection_graph_.resize(u32_nd_);
    for (uint32_t i = 0; i < u32_nd_; ++i)
    {
        projection_graph_[i].reserve(M_pjbp);
    }
}

void IndexPipeline::CalculateProjectionep()
{
    float *center = new float[dimension_]();
    memset(center, 0, sizeof(float) * dimension_);
    // calculate centroid in base point
    for (size_t i = 0; i < nd_; ++i)
    {
        for (size_t d = 0; d < dimension_; ++d)
        {
            center[d] += data_bp_[i * dimension_ + d];
        }
    }

    for (size_t d = 0; d < dimension_; ++d)
    {
        center[d] /= (float)nd_;
    }

    float *distances = new float[nd_]();
    memset(distances, 0, sizeof(float) * nd_);
#pragma omp parallel for
    for (size_t i = 0; i < nd_; ++i)
    {
        const float *cur_data = data_bp_ + i * dimension_;
        float diff = 0;
        for (size_t j = 0; j < dimension_; ++j)
        {
            diff += ((center[j] - cur_data[j]) * (center[j] - cur_data[j]));
        }
        distances[i] = diff;
    }

    uint32_t closest = 0;
    for (size_t i = 1; i < nd_; ++i)
    {
        if (projection_graph_[i].size() > 0 && distances[i] < distances[closest])
        { // 孤立节点不可为质心
            closest = static_cast<uint32_t>(i);
        }
    }
    projection_ep_ = closest;
    delete[] center;
    delete[] distances;
}

void IndexPipeline::SaveIndex(const char *filename)
{
    std::ofstream out(filename, std::ios::binary | std::ios::out);
    if (!out.is_open())
    {
        throw std::runtime_error("cannot open file");
    }
    out.write((char *)&projection_ep_, sizeof(uint32_t));
    out.write((char *)&u32_nd_, sizeof(uint32_t));
    for (uint32_t i = 0; i < u32_nd_; ++i)
    {
        uint32_t nbr_size = projection_graph_[i].size();
        out.write((char *)&nbr_size, sizeof(uint32_t));
        out.write((char *)projection_graph_[i].data(), sizeof(uint32_t) * nbr_size);
    }
    out.close();
}

void IndexPipeline::Save(const char *filename)
{
    // write graph
    std::ofstream out(filename, std::ios::binary | std::ios::out);
    uint32_t npts = static_cast<uint32_t>(total_pts_);
    out.write((char *)&npts, sizeof(npts));
    for (uint32_t i = 0; i < total_pts_; i++)
    {
        uint32_t nbr_size = static_cast<uint32_t>(bipartite_graph_[i].size());
        out.write((char *)&nbr_size, sizeof(nbr_size));
        out.write((char *)bipartite_graph_[i].data(), nbr_size * sizeof(uint32_t));
    }
    out.close();
}

void IndexPipeline::Load(const char *filename)
{
    // load graph to bipartite_graph
    std::ifstream in(filename, std::ios::binary);
    uint32_t npts;
    in.read((char *)&npts, sizeof(npts));
    bipartite_graph_.resize(npts);
    for (uint32_t i = 0; i < npts; i++)
    {
        uint32_t nbr_size;
        in.read((char *)&nbr_size, sizeof(nbr_size));
        bipartite_graph_[i].resize(nbr_size);
        in.read((char *)bipartite_graph_[i].data(), nbr_size * sizeof(uint32_t));
    }
    in.close();
}

void IndexPipeline::LoadVectorData(const char *base_file, const char *sampled_query_file)
{
    uint32_t base_num = 0, sq_num = 0, base_dim = 0, q_dim = 0;

    load_meta<float>(base_file, base_num, base_dim);
    if (strlen(sampled_query_file) != 0)
    {
        load_meta<float>(sampled_query_file, sq_num, q_dim);
        if (base_dim != q_dim)
        {
            throw std::runtime_error("base and query dimension mismatch");
        }
    }
    float *base_data = nullptr;
    float *sampled_query_data = nullptr;
    load_data<float>(base_file, base_num, base_dim, base_data);
    // load_data<float>(sampled_query_file, sq_num, q_dim, sampled_query_data);

    if (need_normalize)
    {
        for (size_t i = 0; i < base_num; ++i)
        {
            normalize<float>(base_data + i * (uint64_t)base_dim, (uint64_t)base_dim);
        }
    }

    data_bp_ = data_align(base_data, base_num, base_dim);
    // data_sq_ = data_align(sampled_query_data, sq_num, q_dim);

    nd_ = base_num;
    nd_sq_ = sq_num;
    total_pts_ = nd_ + nd_sq_;
    u32_nd_ = static_cast<uint32_t>(nd_);
    u32_nd_sq_ = static_cast<uint32_t>(nd_sq_);
}

void IndexPipeline::LoadGraph(const char *filename)
{
    // load graph to projection graph
    std::ifstream in(filename, std::ios::binary);
    uint32_t npts;
    in.read((char *)&projection_ep_, sizeof(uint32_t));
    in.read((char *)&npts, sizeof(npts));
    projection_graph_.resize(npts);
    float out_degree = 0.0;
    for (uint32_t i = 0; i < npts; i++)
    {
        uint32_t nbr_size;
        in.read((char *)&nbr_size, sizeof(nbr_size));
        out_degree += static_cast<float>(nbr_size);
        projection_graph_[i].resize(nbr_size);
        in.read((char *)projection_graph_[i].data(), nbr_size * sizeof(uint32_t));
    }
    fo.print("Projection graph, avg_degree: " + std::to_string(out_degree / npts));
    in.close();
}