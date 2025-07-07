#include "index_pipeline.h"

#include <omp.h>
#include <cblas.h>

#include <bitset>
#include <boost/dynamic_bitset.hpp>
#include <chrono>
#include <ctime>
#include <random>
#include <atomic>
#include <utility>

#include "efanna2e/exceptions.h"
#include "efanna2e/parameters.h"
#include <queue>

#include "gt_cache.h"
#include "candidates.h"
#include "knn.h"

// define likely unlikely
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#define PROJECTION_SLACK    2
#define BATCH               1024
#define STREAMS             4
namespace efanna2e {

IndexPipeline::IndexPipeline(const size_t dimension, const size_t n, Metric m, Index *initializer)
    : Index(dimension, n, m), initializer_{initializer}, total_pts_const_(n) {
    l2_distance_ = new DistanceL2();
    if (m == efanna2e::COSINE) {
        need_normalize = true;
    }
}

IndexPipeline::~IndexPipeline() {}

void IndexPipeline::Build(size_t n, const float *data, const Parameters &parameters){};

void IndexPipeline::Search(const float *query, const float *x, size_t k, const Parameters &parameters, unsigned *indices, float *res_dists) {};

void IndexPipeline::BuildPipeline(size_t n_sq, float *sq_data, size_t n_bp, float *bp_data,
                                  Parameters &parameters) {
    auto s = std::chrono::high_resolution_clock::now();
    uint32_t CE = parameters.Get<uint32_t>("CE");
    uint32_t k = parameters.Get<uint32_t>("k");
    float iso_thres = parameters.Get<float>("iso_thres");
    omp_set_num_threads(parameters.Get<uint32_t>("num_threads"));
    // omp_set_nested(1);
    // 新代码（允许2层嵌套）
    omp_set_max_active_levels(2);
    std::cout << "CE: " << CE << std::endl;
    if (CE == 0) {
        std::cout << "start build index without connectivity enhancement" << std::endl;
    } else {
        std::cout << "start build index with connectivity enhancement" << std::endl;
    }
    data_bp_ = bp_data;
    data_sq_ = sq_data;
    nd_ = n_bp;
    nd_sq_ = n_sq;
    total_pts_ = nd_ + nd_sq_;
    u32_nd_ = static_cast<uint32_t>(nd_);
    u32_nd_sq_ = static_cast<uint32_t>(nd_sq_);
    u32_total_pts_ = static_cast<uint32_t>(total_pts_);
    locks_ = std::vector<std::mutex>(total_pts_);
    learn_base_knn_.resize(u32_nd_sq_);
    SetParameters(parameters);

    if (need_normalize) {
        std::cout << "normalizing base data" << std::endl;
        for (size_t i = 0; i < nd_; ++i) {
            float *data = const_cast<float *>(data_bp_);
            normalize(data + i * dimension_, dimension_);
        }
    }

    ProjectionReserveSpace();

    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess || deviceCount == 0) {
        std::cout << "No CUDA-capable GPU found or CUDA driver not installed." << std::endl;
        exit(0);
    }
    std::cout << "Found " << deviceCount << " CUDA-capable device(s)." << std::endl;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Compute Capability: " << prop.major << " " << prop.minor << std::endl;
    if (!prop.deviceOverlap) std::cout << "Device does not support overlap, multi-stream may not improve performance.\n";
    else std::cout << "Device supports overlap, multi-stream may improve performance.\n";

    // 1. Pinned memory 分配
    float *h_base, *h_queries;
    CUDA_CHECK(cudaMallocHost(&h_base, nd_ * dimension_ * sizeof(float)));
    memcpy(h_base, data_bp_, nd_ * dimension_ * sizeof(float));
    // free(data_bp_);
    CUDA_CHECK(cudaMallocHost(&h_queries, nd_sq_ * dimension_ * sizeof(float)));
    memcpy(h_queries, data_sq_, nd_sq_ * dimension_ * sizeof(float));
    // free(data_sq_);
    // GPU 内存分配
    float *d_queries, *d_base;
    int *d_idx; float *d_dist;
    CUDA_CHECK(cudaMalloc(&d_base, nd_ * dimension_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_queries, BATCH * dimension_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_idx, BATCH * k * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dist, BATCH * k * sizeof(float)));
    // 初始化 streams 和 events，分块上传基础向量
    cudaStream_t streams[STREAMS];
    // 创建时间记录事件
    cudaEvent_t start_knn[STREAMS], stop_knn[STREAMS];
    cudaEvent_t start_upd[STREAMS], stop_upd[STREAMS];
    // cudaEvent_t events[STREAMS];
    size_t total_elements = (size_t)nd_ * dimension_;
    size_t chunk_size = (total_elements + STREAMS - 1) / STREAMS;
    for(int i = 0; i < STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        CUDA_CHECK(cudaEventCreate(&start_knn[i]));
        CUDA_CHECK(cudaEventCreate(&stop_knn[i]));
        CUDA_CHECK(cudaEventCreate(&start_upd[i]));
        CUDA_CHECK(cudaEventCreate(&stop_upd[i]));
    }
    for(int i = 0; i < STREAMS; i++){
        size_t offset = i * chunk_size;
        size_t remain = total_elements - offset;
        size_t chunk  = remain < chunk_size ? remain : chunk_size;
        CUDA_CHECK(cudaMemcpyAsync(
            d_base + offset,
            h_base + offset,                // 前两个参数按元素偏移
            chunk * sizeof(float),     // 字节数
            cudaMemcpyHostToDevice,
            streams[i]
        ));
        // CUDA_CHECK(cudaEventRecord(events[i], streams[i]));
    }
    // 同步等待
    for (int i = 0; i < STREAMS; i++) CUDA_CHECK(cudaStreamSynchronize(streams[i]));

    gpu_graph_create(u32_nd_, k);

    int stream_id = 0;
    for (uint32_t processed = 0; processed < nd_sq_;) {
        std::cout << "Processed: " << processed << "/" << nd_sq_ << std::endl;
        int batch = std::min<uint32_t>(BATCH, u32_nd_sq_ - processed);
        int sid = stream_id % STREAMS;

        if (stream_id >= STREAMS) {
            CUDA_CHECK(cudaEventSynchronize(stop_upd[sid]));
            // 计算时间（毫秒）
            float ms_knn = 0.f, ms_upd = 0.f;
            CUDA_CHECK(cudaEventElapsedTime(&ms_knn, start_knn[sid], stop_knn[sid]));
            CUDA_CHECK(cudaEventElapsedTime(&ms_upd, start_upd[sid], stop_upd[sid]));

            std::cout << "[Stream " << sid << "] Batch " << processed << ": "
                    << "knn_compute = " << ms_knn << " ms, "
                    << "graph_update = " << ms_upd << " ms\n";
        }
        // 传输 batch 查询
        CUDA_CHECK(cudaMemcpyAsync(
            d_queries, 
            h_queries + processed * dimension_,
            batch * dimension_ * sizeof(float),
            cudaMemcpyHostToDevice, streams[sid]));
        // k-NN 计算
        CUDA_CHECK(cudaEventRecord(start_knn[sid], streams[sid]));   
        gpu_knn_compute_async(d_queries, d_base,
                              nd_, dimension_, batch, k,
                              d_idx, d_dist,
                              streams[sid]);
        CUDA_CHECK(cudaEventRecord(stop_knn[sid], streams[sid]));
        // 将 idx/dist 回传 GPU 用于图更新（无需拷回 CPU）
        CUDA_CHECK(cudaEventRecord(start_upd[sid], streams[sid]));
        gpu_handle_knn_updates_async(nd_, d_idx, d_dist,
                                     batch, k, dimension_,
                                     streams[sid]);
        CUDA_CHECK(cudaEventRecord(stop_upd[sid], streams[sid]));
        
        processed += batch;
        stream_id++;
        // 等待更新完成，以查询孤立节点比例
        if (sid != STREAMS - 1) continue;
        CUDA_CHECK(cudaStreamSynchronize(streams[sid]));
        float iso_ratio = gpu_get_isolated_ratio(u32_nd_);
        std::cout << "Processed " << processed << " queries, isolated ratio = " << iso_ratio << "\n";
        if(1 - iso_ratio >= iso_thres){
            std::cout << "Isolated ratio below threshold, stopping.\n";
            break;
        }    
    }
    // 释放事件资源
    // CUDA_CHECK(cudaEventDestroy(start_knn));
    // CUDA_CHECK(cudaEventDestroy(stop_knn));
    // CUDA_CHECK(cudaEventDestroy(start_upd));
    // CUDA_CHECK(cudaEventDestroy(stop_upd));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFreeHost(h_base));
    CUDA_CHECK(cudaFreeHost(h_queries));
    std::cout << "All done. Copying graph from GPU...\n";

    // 第一步：获取设备符号的地址
    int *d_in_deg_ptr = nullptr;
    int *d_adj_ptr     = nullptr;
    float *d_adj_dist_ptr = nullptr;
    CUDA_CHECK(cudaGetSymbolAddress((void**)&d_in_deg_ptr,   g.in_deg));
    CUDA_CHECK(cudaGetSymbolAddress((void**)&d_adj_ptr,      g.adj));
    CUDA_CHECK(cudaGetSymbolAddress((void**)&d_adj_dist_ptr, g.adj_dist));
    // 第二步：拷贝数据回 host
    std::vector<int> h_in_deg(u32_nd_);
    std::vector<int> h_adj(u32_nd_ * k);
    CUDA_CHECK(cudaMemcpy(h_in_deg.data(), d_in_deg_ptr, u32_nd_ * sizeof(int),     cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_adj.data(),    d_adj_ptr,    u32_nd_ * k * sizeof(int), cudaMemcpyDeviceToHost));

    projection_graph_.resize(u32_nd_);
    for(int i = 0; i < u32_nd_; i++){
        int deg = h_in_deg[i] <= k ? h_in_deg[i] : k;
        projection_graph_[i].reserve(deg);
        for(int j = 0; j < deg; j++){
            int nei = h_adj[i * k + j];
            if(nei >= 0) projection_graph_[i].push_back((uint32_t)nei);
        }
    }
    gpu_graph_destroy();

    // LinkProjection();

    // stats projection graph degree
    float avg_degree = 0;
    uint64_t total_degree = 0;
    uint32_t max_degree = 0;
    uint32_t min_degree = std::numeric_limits<uint32_t>::max();
    for (uint32_t i = 0; i < u32_nd_; ++i) {
        if (projection_graph_[i].size() > max_degree) {
            max_degree = projection_graph_[i].size();
        }
        if (projection_graph_[i].size() < min_degree) {
            min_degree = projection_graph_[i].size();
        }
        avg_degree += static_cast<float>(projection_graph_[i].size());
        total_degree += projection_graph_[i].size();
    }
    std::cout << "total degree: " << total_degree << std::endl;
    avg_degree /= (float)u32_nd_;
    std::cout << "After projection, average degree of projection graph: " << avg_degree << std::endl;
    std::cout << "After projection, max degree of projection graph: " << max_degree << std::endl;
    std::cout << "After projection, min degree of projection graph: " << min_degree << std::endl;

    std::cout << std::endl;

    CalculateProjectionep();
}

void IndexPipeline::LinkProjection() {
    uint32_t M_pjbp = parameters_.Get<uint32_t>("M_pjbp");
    float iso_thres = parameters_.Get<float>("iso_thres");
    std::string cache_file = parameters_.Get<std::string>("cache_file");
    uint32_t k = parameters_.Get<uint32_t>("k");
    uint32_t Nq = parameters_.Get<uint32_t>("M_sq");

    // std::vector<std::vector<Neighbor>> candicate_nbrs(u32_nd_);
    // std::vector<std::vector<uint32_t>> candicate_nbr_ids(u32_nd_);
    // std::vector<uint32_t> candicate_nbrs_size(u32_nd_);
    // std::vector<std::mutex> candicate_nbrs_locks(u32_nd_);
    std::vector<Candidates> candidates(u32_nd_);
#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < u32_nd_; ++i)
        candidates[i].Set(M_pjbp, i);

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    std::cout << "start pipe build with iso_thres: " << iso_thres << std::endl;
    uint32_t query_count = 0, visit_count = 0;
    std::vector<std::atomic<uint32_t>> visit_flags(u32_nd_);
    // visit_flags = std::move(visit_flags_);
    GTCache gt_cache(cache_file, k, u32_nd_, data_sq_, data_bp_, dimension_, metric_);

    std::vector<double> compute_GT_times(u32_nd_sq_);
    // int ratio_flag = 1;
#pragma omp parallel for schedule(dynamic, 64)
    for (uint32_t it_sq = 0; it_sq < u32_nd_sq_; ++it_sq) {
        if (visit_count >= u32_nd_ * iso_thres) continue;
        Result res = gt_cache.GetGT(it_sq);
        compute_GT_times[it_sq] = res.time_ms;

        #pragma omp atomic
        query_count++;

        learn_base_knn_[it_sq].assign(res.closest_points, res.closest_points + k);
        if (learn_base_knn_[it_sq].size() > Nq) {
            learn_base_knn_[it_sq].resize(Nq);
            learn_base_knn_[it_sq].shrink_to_fit();
        }
        uint32_t knn_size = learn_base_knn_[it_sq].size();

        std::vector<bool> visited(knn_size, true);
        for (uint32_t it = 0; it < knn_size; ++it) {
            if (visit_flags[learn_base_knn_[it_sq][it]].exchange(1) == 0) {
                visited[it] = false;
                #pragma omp atomic
                visit_count++;
            }
        }

        for (uint32_t it_first = 0; it_first < knn_size && visit_count < u32_nd_ * iso_thres; ++it_first) {
            int64_t first_node = learn_base_knn_[it_sq][it_first];
            for (size_t it_second = it_first + 1; it_second < knn_size; ++it_second) {
                // if (visited[it_first] && visited[it_second]) continue;
                int64_t second_node = learn_base_knn_[it_sq][it_second];
                float dist = distance_->compare(data_bp_ + dimension_ * (uint64_t)first_node, 
                                                data_bp_ + dimension_ * (uint64_t)second_node,
                                                (unsigned)dimension_);
                // if (!visited[it_first]) 
                    candidates[first_node].AddNbr(second_node, dist);
                // if (!visited[it_second]) 
                    candidates[second_node].AddNbr(first_node, dist);
            }

            // candidates[first_node].FilterNbrs(data_bp_, dimension_, distance_, 0.2, projection_graph_[first_node]);
        }

        if (omp_get_thread_num() == 0) {
            double ratio = visit_count * 100.0 / u32_nd_;
            std::cout << "Non-isolated node ratio: " << ratio << "%; Query count: " << query_count << std::endl;
            // if (ratio > ratio_flag) {
            //     // gt_cache.SaveCacheOld(new_results);
            //     gt_cache.SaveCacheSafely();
            //     ratio_flag += 2;
            // }
        }
    }

#pragma omp parallel for schedule(dynamic, 64)
    for (uint32_t i = 0; i < u32_nd_; i++) {
        candidates[i].FilterNbrs(data_bp_, dimension_, distance_, projection_graph_[i]);
    }

    double total_compute_GT_time = 0;
    for (auto &t: compute_GT_times) 
        total_compute_GT_time += t/1000;
    std::cout << "Total GT calculation time(s): " << total_compute_GT_time << std::endl;

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

    // save t2 - t1 in seconds in projection time
    auto projection_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    
    std::atomic<uint32_t> degree_cnt(0);
    std::atomic<uint32_t> zero_cnt(0);
#pragma omp parallel for schedule(static, 100)
    for (uint32_t i = 0; i < u32_nd_; ++i) {
        if (projection_graph_[i].size() < M_pjbp) {
            // std::cout << "Warning: projection graph node " << node << " has less than M_pjbp neighbors." << std::endl;
            degree_cnt.fetch_add(1);
            if (projection_graph_[i].size() == 0) {
                zero_cnt.fetch_add(1);
            }
        }
    }
    std::cout << "Projection time: " << projection_time << std::endl;
    std::cout << "Warning: " << degree_cnt.load() << " nodes have less than M_pjbp neighbors." << std::endl;
    std::cout << "Warning: " << zero_cnt.load() << " nodes have no neighbors." << std::endl;
}

std::pair<uint32_t, uint32_t> IndexPipeline::SearchPipe(const float *query, size_t k, size_t &qid, const Parameters &parameters, unsigned *indices, std::vector<float>& res_dists) {
    uint32_t L_pq = parameters.Get<uint32_t>("L_pq");
    NeighborPriorityQueue search_queue(L_pq);
    std::vector<uint32_t> init_ids;
    init_ids.push_back(projection_ep_);
    prefetch_vector((char *)(data_bp_ + projection_ep_ * dimension_), dimension_);
    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    for (auto &id : init_ids) {
        float distance = distance_->compare(data_bp_ + id * dimension_, query, (unsigned)dimension_);
        search_queue.insert(Neighbor(id, distance, false));
    }

    uint32_t cmps = 0;
    uint32_t hops = 0;
    while (search_queue.has_unexpanded_node()) {
        auto cur_id = search_queue.closest_unexpanded().id;
        uint32_t *cur_nbrs = projection_graph_[cur_id].data();
        ++hops;
        // get neighbors' neighbors, first
        for (size_t j = 0; j < projection_graph_[cur_id].size(); ++j) {  // current check node's neighbors
            uint32_t nbr = *(cur_nbrs + j);
            _mm_prefetch((char *)(visited_array + *(cur_nbrs + j + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(data_bp_ + *(cur_nbrs + j + 1) * dimension_), _MM_HINT_T0);
            if (visited_array[nbr] != visited_array_tag) {
                visited_array[nbr] = visited_array_tag;
                float distance = distance_->compare(data_bp_ + nbr * dimension_, query, (unsigned)dimension_);
                
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


void IndexPipeline::ProjectionReserveSpace() {
    std::cout << "begin projection graph init" << std::endl;
    uint32_t M_pjbp = parameters_.Get<uint32_t>("M_pjbp");
    projection_graph_.resize(u32_nd_);
    for (uint32_t i = 0; i < u32_nd_; ++i) {
        projection_graph_[i].reserve(M_pjbp * PROJECTION_SLACK);
    }
}

void IndexPipeline::CalculateProjectionep() {
    float *center = new float[dimension_]();
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

    float *distances = new float[nd_]();
    memset(distances, 0, sizeof(float) * nd_);
#pragma omp parallel for
    for (size_t i = 0; i < nd_; ++i) {
        const float *cur_data = data_bp_ + i * dimension_;
        float diff = 0;
        for (size_t j = 0; j < dimension_; ++j) {
            diff += ((center[j] - cur_data[j]) * (center[j] - cur_data[j]));
        }
        distances[i] = diff;
    }

    uint32_t closest = 0;
    for (size_t i = 1; i < nd_; ++i) {
        if (projection_graph_[i].size() > 0 && distances[i] < distances[closest]) { // 孤立节点不可为质心
            closest = static_cast<uint32_t>(i);
        }
    }
    projection_ep_ = closest;
    delete[] center;
    delete[] distances;
}

void IndexPipeline::SaveIndex(const char *filename) {
    std::ofstream out(filename, std::ios::binary | std::ios::out);
    if (!out.is_open()) {
        throw std::runtime_error("cannot open file");
    }
    out.write((char *)&projection_ep_, sizeof(uint32_t));
    out.write((char *)&u32_nd_, sizeof(uint32_t));
    for (uint32_t i = 0; i < u32_nd_; ++i) {
        uint32_t nbr_size = projection_graph_[i].size();
        out.write((char *)&nbr_size, sizeof(uint32_t));
        out.write((char *)projection_graph_[i].data(), sizeof(uint32_t) * nbr_size);
    }
    out.close();
}

void IndexPipeline::Save(const char *filename) {
    // write graph
    std::ofstream out(filename, std::ios::binary | std::ios::out);
    uint32_t npts = static_cast<uint32_t>(total_pts_);
    out.write((char *)&npts, sizeof(npts));
    for (uint32_t i = 0; i < total_pts_; i++) {
        uint32_t nbr_size = static_cast<uint32_t>(bipartite_graph_[i].size());
        out.write((char *)&nbr_size, sizeof(nbr_size));
        out.write((char *)bipartite_graph_[i].data(), nbr_size * sizeof(uint32_t));
    }
    out.close();
}

void IndexPipeline::Load(const char *filename) {
    // load graph to bipartite_graph
    std::ifstream in(filename, std::ios::binary);
    uint32_t npts;
    in.read((char *)&npts, sizeof(npts));
    bipartite_graph_.resize(npts);
    for (uint32_t i = 0; i < npts; i++) {
        uint32_t nbr_size;
        in.read((char *)&nbr_size, sizeof(nbr_size));
        bipartite_graph_[i].resize(nbr_size);
        in.read((char *)bipartite_graph_[i].data(), nbr_size * sizeof(uint32_t));
    }
    in.close();
}

void IndexPipeline::LoadVectorData(const char *base_file, const char *sampled_query_file) {
    uint32_t base_num = 0, sq_num = 0, base_dim = 0, q_dim = 0;

    load_meta<float>(base_file, base_num, base_dim);
    if (strlen(sampled_query_file) != 0) {
        load_meta<float>(sampled_query_file, sq_num, q_dim);
        if (base_dim != q_dim) {
            throw std::runtime_error("base and query dimension mismatch");
        }
    }
    float *base_data = nullptr;
    float *sampled_query_data = nullptr;
    load_data<float>(base_file, base_num, base_dim, base_data);
    // load_data<float>(sampled_query_file, sq_num, q_dim, sampled_query_data);

    if (need_normalize) {
        std::cout << "Normalizing base data" << std::endl;
        for (size_t i = 0; i < base_num; ++i) {
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
    u32_total_pts_ = static_cast<uint32_t>(total_pts_);
}

void IndexPipeline::LoadGraph(const char *filename) {
    // load graph to projection graph
    std::ifstream in(filename, std::ios::binary);
    uint32_t npts;
    in.read((char *)&projection_ep_, sizeof(uint32_t));
    std::cout << "Projection graph, "
              << "ep: " << projection_ep_ << std::endl;
    in.read((char *)&npts, sizeof(npts));
    projection_graph_.resize(npts);
    float out_degree = 0.0;
    for (uint32_t i = 0; i < npts; i++) {
        uint32_t nbr_size;
        in.read((char *)&nbr_size, sizeof(nbr_size));
        out_degree += static_cast<float>(nbr_size);
        projection_graph_[i].resize(nbr_size);
        in.read((char *)projection_graph_[i].data(), nbr_size * sizeof(uint32_t));
    }
    std::cout << "Projection graph, "
              << "avg_degree: " << out_degree / npts << std::endl;
    in.close();
}

}

