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
#include "json.hpp"
// #include "test.h"

#define QUEUE_SIZE 32

using namespace efanna2e;

// 构造函数，初始化索引管道
IndexGPU::IndexGPU(const size_t dimension, const size_t n, Metric m, Index* initializer)
    : Index(dimension, n, m), initializer_{initializer} {
    if (m == efanna2e::COSINE) need_normalize = true;
}

IndexGPU::~IndexGPU() {
    // if (h_base) CUDA_CHECK(cudaFreeHost(h_base));
}

void IndexGPU::load_config(const std::string& config_file) {
    std::ifstream f;
    if (config_file == "default") {
        fo.print("Loading config from std");
        f.open("../config.json");
    } else
        f.open(config_file);

    if (!f.is_open()) fo.iprint("Cannot open config file");

    nlohmann::json j;
    f >> j;

    PC.dim = j.value("dim", 200);
    PC.k = j.value("k", 100);
    PC.max_degree = j.value("max_degree", 32);
    PC.beam_capacity = j.value("beam_capacity", 1024);
    PC.enhance_nap_batch = j.value("enhance_nap_batch", 4096);
    PC.batch = j.value("batch", 1024);
    PC.enhance_search_batch = j.value("enhance_search_batch", 524288);
    PC.enhance_beam_capacity = j.value("enhance_beam_capacity", 100);
    PC.enhance_query_num = j.value("enhance_query_num", 1);
    PC.insert_batch = j.value("insert_batch", 2048);
    PC.insert_query_k = j.value("insert_query_k", 1);
    PC.test_beam_capacity = j.value("test_beam_capacity", 512);
    PC.use_knn_cache = j.value("use_knn_cache", 0);
    PC.search_start_nodes_num = j.value("search_start_nodes_num", 16);
    PC.knn_max_hit = j.value("knn_max_hit", 10);
    PC.need_discard = j.value("need_discard", 1);
    PC.enhance_mode = j.value("enhance_mode", 0);
    PC.knn_rebuild_thres = j.value("knn_rebuild_thres", 0.1);
    PC.ivf_rebuild = j.value("ivf_rebuild", 0);
    PC.ivf_nlist = j.value("ivf_nlist", 4096);
    PC.ivf_nprobe = j.value("ivf_nprobe", 16);
    PC.test_search_frequency = j.value("test_search_frequency", 100);
    PC.test_search = j.value("test_search", 1);
    PC.ivf_points_per_centroid = j.value("ivf_points_per_centroid", 256);
    PC.knn_rebuild = j.value("knn_rebuild", 0);
    PC.top_hit_num = j.value("top_hit_num", 0);
    PC.calc_hit = j.value("calc_hit", 0);
    PC.upd_mode = j.value("upd_mode", 0);
    PC.upd_build = j.value("upd_build", 1);
    PC.upd_type = j.value("upd_type", 1);
    PC.test_beam_capacity = j.value("test_beam_capacity", 512);
    PC.insert_query_k = j.value("insert_query_k", 1);
    PC.search_in_cpu = j.value("search_in_cpu", 0);
    PC.knn_rebuild_min_cnt = j.value("knn_rebuild_min_cnt", 2000000);
}

Test_Result IndexGPU::Search_and_Verify(const float* queries, const int* gts, int num,
                                        int beam_capacity) {
    Test_Result tr;
    float* d_queries;
    int *d_ids_res, *d_ids_gt;
    cudaMalloc(&d_queries, num * dimension_ * sizeof(float));
    cudaMalloc(&d_ids_res, num * k * sizeof(int));
    cudaMalloc(&d_ids_gt, num * k * sizeof(int));
    cudaMemcpy(d_queries, queries, num * dimension_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ids_gt, gts, num * k * sizeof(int), cudaMemcpyHostToDevice);

    static bool init = false;
    if (!init) {
        if (beam_capacity > 0)
            gpufuncs->search_prepare(num, beam_capacity);
        else
            gpufuncs->search_prepare(num);
        init = true;
    }
    tr = gpufuncs->Search(d_queries, d_ids_res, nullptr, num, graph->get_start_ids());
    gpufuncs->search_verify(d_ids_res, d_ids_gt, tr, num, k);

    cudaFree(d_queries);
    cudaFree(d_ids_res);
    cudaFree(d_ids_gt);
    return tr;
}

Test_Result IndexGPU::Search(const float* queries, int num, int beam_capacity, int* ids_res,
                             int M) {
    if (!M) M = PC.search_start_nodes_num;
    float *d_queries, time;
    int* d_ids_res;
    cudaMalloc(&d_ids_res, num * k * sizeof(int));
    cudaMalloc(&d_queries, num * dimension_ * sizeof(float));
    cudaMemcpy(d_queries, queries, num * dimension_ * sizeof(float), cudaMemcpyHostToDevice);

    gpufuncs->search_prepare(num, beam_capacity);
    auto res = gpufuncs->Search(d_queries, d_ids_res, nullptr, num, graph->get_start_ids(), k);
    cudaMemcpy(ids_res, d_ids_res, num * k * sizeof(int), cudaMemcpyDeviceToHost);
    // const int* faiss_res = gpufuncs->knn_compute_faiss(d_queries, num, time);
    // Test_Result faiss_tr(res);
    // gpufuncs->search_verify(d_ids_res, faiss_res, faiss_tr, num, k);
    // fo.print("Faiss check: recall=" + TOS(faiss_tr.calc_recall(num * k)) + "%");
    cudaFree(d_queries);
    cudaFree(d_ids_res);
    return res;
}

void IndexGPU::Delete(const std::vector<int> del_ids) {
    auto s = std::chrono::high_resolution_clock::now();
    auto future = std::async(std::launch::async,
                             [this, &del_ids]() { query_index->get_knns().del(del_ids); });

    const int N = del_ids.size();
    std::vector<int> offsets;

    gpufuncs->del_node_mark(del_ids);
    if (!PC.upd_type) {
        auto fut = gpufuncs->executor.submit_with_future(
            [this, &del_ids]() { query_index->get_knns().del(del_ids); });
        fut.get();
        return;
    }

    auto fut = gpufuncs->executor.submit_with_future(
        [this, &del_ids, &offsets]() { graph->get_del_innbrs(del_ids, del_innbrs, offsets); });
    fut.get();

    gpufuncs->executor.submit([this, &del_ids]() { query_index->get_knns().del(del_ids); });

    // graph->get_innbrs(del_ids, del_innbrs, del_ids_for_innbr, offsets);
    std::vector<int> batch_innbr_offsets(INNBR_BATCH / 32 + 1024, 0);
    const int *batch_del_ids = del_ids.data(), *batch_del_innbrs = del_innbrs.data();
    // del_old_nbrs.resize(N * max_degree);
    // del_new_nbrs.resize(offsets[N] * max_degree);
    // del_new_degs.resize(offsets[N]);
    // batch_innbr_offsets.reserve(INNBR_BATCH / 32 + 1);
    // batch_innbr_offsets.push_back(0);
    // batch_del_ids_for_innbr.reserve(INNBR_BATCH);
    int batch_num = 0, del_num = 0;
    float time = 0;
    for (int i = 0; i < N; i++) {
        const int innbrs_start = offsets[i], innbrs_end = offsets[i + 1],
                  innbr_num = innbrs_end - innbrs_start;
        if (batch_num + innbr_num > INNBR_BATCH) {
            time += gpufuncs->vector_delete(batch_del_ids, batch_innbr_offsets, batch_del_innbrs,
                                            del_num, batch_num);

            // fo.print("Delete nodes(" + TOS(time) + "s): " + TOS(i) + "/" + TOS(N) +
            //          " with innbrs=" + TOS(batch_num));
            batch_del_ids = del_ids.data() + i;
            batch_del_innbrs = del_innbrs.data() + innbrs_start;
            // batch_innbr_offsets.assign(1, 0);
            del_num = batch_num = 0;
        }
        batch_num += innbr_num;
        // batch_innbr_offsets.push_back(batch_num);
        batch_innbr_offsets[++del_num] = batch_num;
    }
    if (batch_num > 0) {
        time += gpufuncs->vector_delete(batch_del_ids, batch_innbr_offsets, batch_del_innbrs,
                                        del_num, batch_num);
        // fo.print("Delete nodes(" + TOS(time) + "s): " + TOS(N) + "/" + TOS(N) +
        //          " with innbrs=" + TOS(batch_num));
    }
    // del_node_fut = std::async(std::launch::async, [this, del_ids]() {
    //     graph->delete_nodes(del_ids, del_innbrs, del_old_nbrs, del_new_nbrs, del_new_degs);
    // });
    // print_times();
    // if (future.valid()) future.wait();
    gpufuncs->executor.finish();
    const float total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::high_resolution_clock::now() - s)
                                 .count() /
                             1000.0f;
    fo.print("Delete nodes(" + TOS(total_time) + "s) finished!");
}
void IndexGPU::Insert(const float* ins_vectors, int ins_num) {
    auto s = std::chrono::high_resolution_clock::now();
    std::vector<int> ins_ids(ins_num);
    graph->insert_nodes(ins_vectors, ins_ids, ins_num);
    gpufuncs->search_prepare(PC.insert_batch);
    for (int i = 0; i < ins_num; i += PC.insert_batch) {
        const int batch = std::min(PC.insert_batch, ins_num - i);
        gpufuncs->vector_insert(ins_vectors + i * dimension_, ins_ids.data() + i, batch);
        // fo.print("Insert vecs(" + TOS(time) + "s): " + TOS(i + batch) + "/" + TOS(ins_num));
    }
    // print_times();
    gpufuncs->executor.finish();
    const float total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::high_resolution_clock::now() - s)
                                 .count() /
                             1000.0f;
    fo.print("Insert nodes(" + TOS(total_time) + "s) finished!");
}

void IndexGPU::RunPrepare(Parameters& parameters, Mode mode, size_t n_bp, float* bp_data) {
    SetParameters(parameters);
    load_config(parameters.Get<std::string>("config_file"));
    fpath base_file = fpath(parameters_.Get<std::string>("base_file"));
    graph_file = fpath(parameters.Get<std::string>("graph_file"));
    fo.Init(parameters.Get<std::string>("log_file"));

    int deviceCount = 0, curDev = -1;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess || deviceCount == 0)
        fo.print("No CUDA-capable GPU found or CUDA driver not installed.");
    else {
        fo.print("Found " + TOS(deviceCount) + " CUDA-capable device(s).");
        cudaGetDevice(&curDev);
        fo.print("Current device (cudaGetDevice) = " + TOS(curDev));
    }

    k = parameters.Get<int>("k");
    max_degree = parameters.Get<int>("max_degree");
    metric = parameters.Get<std::string>("dist") == "l2" ? DIST_METRIC::L2_ : DIST_METRIC::IP_;
    create_dir(graph_file);
    if (n_bp > 0) {
        h_base = bp_data;
        nd_ = n_bp;
        u32_nd_ = static_cast<int>(nd_);
        u32_nd_sq_ = static_cast<int>(nd_sq_);
    }
    if (mode != Mode::SEARCH)
        query_index = new QueryIndex(parameters.Get<std::string>("query_file"), u32_nd_, k,
                                     dimension_, metric);
    GraphType gt =
        u32_nd_ <= GPU_N ? GraphType::GPU : (u32_nd_ <= CPU_N ? GraphType::CPU : GraphType::DISK);
    graph = new Graph(base_file.string(), graph_file, h_base, u32_nd_, max_degree, dimension_, k,
                      metric, mode, gt);
    gpufuncs = new GPUFuncs(graph, query_index, h_base, u32_nd_, dimension_, k, max_degree, metric,
                            PC.test_beam_capacity, mode);
}

void IndexGPU::BuildPrepare(bool only_CE) {
    max_degree = parameters_.Get<int>("max_degree");
    fpath base_file = fpath(parameters_.Get<std::string>("base_file"));
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess || deviceCount == 0) {
        fo.eprint("No CUDA-capable GPU found or CUDA driver not installed.");
    }
    fo.print("Found " + TOS(deviceCount) + " CUDA-capable device(s).");

    int curDev = -1;
    cudaGetDevice(&curDev);
    fo.print("Current device (cudaGetDevice) = " + TOS(curDev));

    // CUDA_CHECK(cudaMallocHost(&h_queries, u32_nd_sq_ * dimension_ * sizeof(float)));
    // memcpy(h_queries, data_sq_, u32_nd_sq_ * dimension_ * sizeof(float));
    // h_queries = data_sq_;

    fo.print("BuildPrepare: base_n-" + TOS(u32_nd_) + " query_n-" + TOS(u32_nd_sq_));
    if (u32_nd_ <= CPU_N) {
        GraphType gt = u32_nd_ <= GPU_N ? GraphType::GPU : GraphType::CPU;
        graph = new Graph("", graph_file, h_base, u32_nd_, max_degree, dimension_, k, metric,
                          only_CE ? Mode::CE : Mode::CONS, gt);
        fo.print("Create GPUFuncs...");
        gpufuncs = new GPUFuncs(graph, base_file, h_base, data_sq_, u32_nd_, u32_nd_sq_,
                                dimension_, k, max_degree, metric, PC.test_beam_capacity,
                                only_CE ? Mode::CE : Mode::CONS, !train_gt_file.empty());
    } else {
        graph = new Graph(base_file.string(), graph_file, h_base, u32_nd_, max_degree, dimension_,
                          k, metric, only_CE ? Mode::CE : Mode::CONS, GraphType::DISK);
        fo.print("Create GPUFuncs...");
        gpufuncs = new GPUFuncs(graph, base_file, nullptr, data_sq_, u32_nd_, u32_nd_sq_,
                                dimension_, k, max_degree, metric, PC.test_beam_capacity,
                                only_CE ? Mode::CE : Mode::CONS, !train_gt_file.empty());
    }
    fo.print("GPU prepare finished");
}

void IndexGPU::RunFree() {
    if (graph) delete graph;
    if (query_index) delete query_index;
    if (gpufuncs) delete gpufuncs;
}

void IndexGPU::BuildGPU(size_t n_sq, float* sq_data, size_t n_bp, float* bp_data,
                        Parameters& parameters) {
    SetParameters(parameters);
    load_config(parameters.Get<std::string>("config_file"));
    int CE = parameters.Get<int>("CE");  // 0: no CE, 1: CE, 2: only CE
    max_degree = parameters.Get<int>("max_degree");
    k = parameters.Get<int>("k");
    metric = parameters.Get<std::string>("dist") == "l2" ? DIST_METRIC::L2_ : DIST_METRIC::IP_;
    cache_file = fpath(parameters.Get<std::string>("cache_file"));
    graph_file = fpath(parameters.Get<std::string>("graph_file"));
    std::string train_gt_file_str = parameters_.Get<std::string>("train_gt_file");
    if (train_gt_file_str != "None") train_gt_file = fpath(train_gt_file_str);
    create_dir(graph_file);
    fo.Init(parameters.Get<std::string>("log_file"));
    h_base = bp_data;
    data_sq_ = sq_data;
    nd_ = n_bp;
    nd_sq_ = n_sq;
    total_pts_ = nd_ + nd_sq_;
    u32_nd_ = static_cast<int>(nd_);
    u32_nd_sq_ = static_cast<int>(nd_sq_);

    // query_sampler = new Query_Sampler(h_base, u32_nd_, data_sq_, u32_nd_sq_, dimension_, 1);

    bool only_CE = (CE == 2);
    BuildPrepare(only_CE);

    // Test test(gpufuncs, h_queries, h_base, dimension_, u32_nd_, k, metric, M_pjbp);
    // test.Run();
    start_time = now_time();

    if (!only_CE && u32_nd_ > GPU_N) gpufuncs->ivf_build();

    if (PC.test_search) test_search_prepare(gpufuncs);

    Cache cache(cache_file, batch_id, k, nd_);

    // test_search(true);

    if (!only_CE) {
        KNN_Queue knn_queue(k, QUEUE_SIZE, gpufuncs);

        std::thread knn_task([this, &knn_queue, &cache] {
            if (train_gt_file.empty()) this->KNNTask(knn_queue, cache);
        });

        std::thread graph_task([this, &knn_queue, &cache] { this->GraphTask(knn_queue, cache); });

        if (knn_task.joinable()) knn_task.join();
        if (graph_task.joinable()) graph_task.join();

        fo.print("NAP done. Copying graph from GPU. Total time: " + TOS(time_diff(start_time)) +
                 "s");
        graph->save_build();
        gpufuncs->save_hits();
    }

    gpufuncs->Build_Free();

    if (CE) {
        if (PC.enhance_mode == 1)
            query_index =
                new QueryIndex(cache, data_sq_, processed, u32_nd_, k, dimension_, metric);
        gpufuncs->connect_enhance(query_index);
        graph->save_build();
    }
    fo.print("All done. Copying graph from GPU. Total time: " + TOS(time_diff(start_time)) + "s");

    if (CE && PC.test_search) {
        gpufuncs->test_search(graph->get_start_ids());
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    BuildFree(cache);
}

void IndexGPU::KNNTask(KNN_Queue& knn_queue, Cache& cache) {
    processed = 0;
    const int old_batch_id = batch_id;
    std::vector<int> batch_knns;
    int* d_batch_knns = nullptr;
    cudaStream_t copy_st;
    if (u32_nd_ <= GPU_N) {
        CUDA_CHECK(cudaMalloc(&d_batch_knns, PC.batch * k * sizeof(int)));
        cudaStreamCreate(&copy_st);
    } else
        batch_knns.resize(PC.batch * k);
    // std::vector<bool> query_used(u32_nd_sq_, false);
    // std::vector<bool> base_visited(u32_nd_, false);
    int batch_num = 0;
    // int base_total_visited = 0;
    for (int i = 0; i < cache.results.size() && !stop_flag; i++) {
        auto& res = cache.results[i];
        if (i <= old_batch_id) {
            processed += res.batch;
            continue;
        }
        int idx = 0;
        while (idx < res.batch) {
            const int cpy_batch = std::min(res.batch - idx, PC.batch - batch_num);
            if (u32_nd_ <= GPU_N)
                CUDA_CHECK(cudaMemcpyAsync(d_batch_knns + batch_num * k, res.knn.data() + idx * k,
                                           cpy_batch * k * sizeof(int), cudaMemcpyHostToDevice,
                                           copy_st));
            else
                memcpy(batch_knns.data() + batch_num * k, res.knn.data() + idx * k,
                       cpy_batch * k * sizeof(int));
            batch_num += cpy_batch;
            idx += cpy_batch;

            if (batch_num == PC.batch) {
                if (u32_nd_ <= GPU_N) {
                    cudaStreamSynchronize(copy_st);
                    knn_queue.WriteTail(d_batch_knns, PC.batch, batch_id);
                } else
                    knn_queue.WriteTailHost(batch_knns.data(), PC.batch, batch_id);
                processed += PC.batch;
                batch_id++;
                batch_num = 0;
            }
        }
    }
    if (batch_num > 0) {
        if (u32_nd_ <= GPU_N) {
            cudaStreamSynchronize(copy_st);
            knn_queue.WriteTail(d_batch_knns, batch_num, batch_id);
        } else
            knn_queue.WriteTailHost(batch_knns.data(), batch_num, batch_id);
        processed += batch_num;
        batch_id++;
        batch_num = 0;
    }

    fo.iprint("Cache loaded with processed: " + TOS(processed / 1000000.0) +
              " M, old_batch_id: " + TOS(old_batch_id) + ". Processing new data...");

    // query_sampler->Init(query_used);
    // query_sampler->generate_query_order(query_used);
    // std::vector<float> sample_queries;
    // auto& sample_qids = query_sampler->OrderSample(sample_queries);
    // std::vector<int> sample_qids(PC.batch);
    while (!stop_flag.load() && processed < u32_nd_sq_) {
        int batch = std::min<int>(PC.batch, u32_nd_sq_ - processed);
        // 计算 batch 查询的 knn
        // query_sampler->Sample(sample_queries, sample_qids, batch);
        if (u32_nd_ <= GPU_N) {
            float time_s;
            int* d_knns = PC.knn_rebuild ? gpufuncs->knn_compute_faiss_hd(
                                               data_sq_ + processed * dimension_, batch, time_s)
                                         : gpufuncs->knn_compute_faiss(
                                               data_sq_ + processed * dimension_, batch, time_s);
            // gpufuncs->knn_compute_faiss(data_sq_ + processed * dimension_, batch, time_s);
            cache.WriteGTCache(d_knns, nullptr, batch_id, batch, time_s);
            knn_queue.WriteTail(d_knns, batch, batch_id);
        } else {
            int index_size;
            const int* knns =
                gpufuncs->knn_compute(data_sq_ + processed * dimension_, batch, index_size);
            cache.WriteGTCacheHost(knns, batch_id, batch);
            knn_queue.WriteTailHost(knns, batch, batch_id);
            if ((float)index_size / u32_nd_ <= 0.1) break;
        }
        processed += batch;
        batch_id++;
        // const int old = base_total_visited;
        // for (int i = 0; i < batch * k; i++) {
        //     const int id = knns[i];
        //     if (!base_visited[id]) {
        //         base_visited[id] = true;
        //         base_total_visited++;
        //     }
        // }
        // const int delta = base_total_visited - old;
        // fo.print(TOS(batch) + " query samples: " + TOS(delta) + "/" + TOS(base_total_visited) +
        //          " " + TOS(delta * 100.0 / (batch * k)) + "%");

        // cache.WriteGTCache(d_knns, sample_qids.data() + query_idx, batch_id, batch, time_s);
    }
    // delete query_sampler;
    if (d_batch_knns) {
        CUDA_CHECK(cudaFree(d_batch_knns));
        cudaStreamDestroy(copy_st);
    }
    if (!stop_flag.load()) {
        stop_flag.store(true);
        knn_queue.Stop();
        gpufuncs->NAP_Stop();
        cv.notify_one();
    }
    fo.print("KNNTask finished...");
}

void IndexGPU::GraphTask(KNN_Queue& knn_queue, Cache& cache) {
    std::thread worker(&IndexGPU::update_graph, this, std::ref(cache), std::ref(knn_queue));

    int *d_knn_ids = nullptr, *h_knn_ids = nullptr;
    if (!train_gt_file.empty()) {
        std::vector<int> query_knns_gt;
        int query_num = 0, gt_k;
        std::ifstream fin(train_gt_file, std::ios::binary);
        fin.read(reinterpret_cast<char*>(&query_num), sizeof(int));
        fin.read(reinterpret_cast<char*>(&gt_k), sizeof(int));
        fo.print("load query gt data from " + train_gt_file.string() +
                 " with nq=" + TOS(query_num) + ", k=" + TOS(gt_k));
        assert(gt_k == k);
        query_knns_gt.resize(query_num * gt_k);
        fin.read(reinterpret_cast<char*>(query_knns_gt.data()), query_num * gt_k * sizeof(int));
        CUDA_CHECK(cudaMalloc(&d_knn_ids, PC.batch * k * sizeof(int)));
        int batch_id = 0;
        for (processed = 0; processed < query_num && !stop_flag.load();
             processed += PC.batch, batch_id++) {
            const int batch = std::min<int>(PC.batch, query_num - processed);
            cudaMemcpy(d_knn_ids, query_knns_gt.data() + processed * k, batch * k * sizeof(int),
                       cudaMemcpyHostToDevice);
            gpufuncs->count_hits(d_knn_ids, batch);
            gpufuncs->graph_update(d_knn_ids, batch_id, batch);
            if (batch_id > 0 && batch_id % PC.test_search_frequency == 0) {
                {
                    std::unique_lock<std::shared_mutex> lock(queue_mtx);
                    gui.update(batch_id, processed + batch);
                }
                cv.notify_one();
            }
        }
        if (!stop_flag.load()) {
            stop_flag.store(true);
            gpufuncs->NAP_Stop();
            cv.notify_one();
        }

    } else {
        if (u32_nd_ <= GPU_N)
            CUDA_CHECK(cudaMalloc(&d_knn_ids, PC.batch * k * sizeof(int)));
        else
            CUDA_CHECK(cudaMallocHost(&h_knn_ids, PC.batch * k * sizeof(int)));
        int total_num = 0;
        while (true) {
            auto s = std::chrono::high_resolution_clock::now();

            int batch_id;
            int batch = u32_nd_ <= GPU_N ? knn_queue.ReadHead(batch_id, d_knn_ids)
                                         : knn_queue.ReadHeadHost(batch_id, h_knn_ids);
            if (batch == -1) break;

            gpufuncs->graph_update(u32_nd_ <= GPU_N ? d_knn_ids : h_knn_ids, batch_id, batch);
            total_num += batch;

            // #ifdef INFO_PRINT
            //         fo.print("handle_knn_updates(" + TOS(load_time) + "s, " + TOS(handle_time) +
            //                  "s): batch_id-" + TOS(batch_id));
            // #endif
            if (batch_id > 0 && batch_id % PC.test_search_frequency == 0) {
                if (u32_nd_ <= CPU_N) {
                    {
                        std::unique_lock<std::shared_mutex> lock(queue_mtx);
                        gui.update(batch_id, total_num);
                    }
                    cv.notify_one();
                } else {
                    auto start_id =
                        gpufuncs->get_search_start_nodes_ivf(PC.search_start_nodes_num, 1);
                    gpufuncs->test_search(start_id);
                }
            }
        }
    }

    if (d_knn_ids) CUDA_CHECK(cudaFree(d_knn_ids));
    if (h_knn_ids) CUDA_CHECK(cudaFreeHost(h_knn_ids));
    if (worker.joinable()) worker.join();
    fo.print("GraphTask finished.");
}

void IndexGPU::BuildFree(Cache& cache) {
    fpath query_file(parameters_.Get<std::string>("query_file"));
    std::filesystem::create_directories(query_file.parent_path());
    std::ofstream ofs(query_file, std::ios::binary);
    if (!ofs.is_open()) fo.eprint("Open query file " + query_file.string() + " failed");

    const int query_size = processed;
    assert(query_size <= cache.get_query_num());
    ofs.write((char*)reinterpret_cast<const char*>(&query_size), sizeof(int));

    int total_batch = 0;
    // std::vector<float> sample_queries(query_size * dimension_);
    for (BatchResult& res : cache.results) {
        const int batch = std::min(res.batch, query_size - total_batch);
        total_batch += batch;
        ofs.write((char*)reinterpret_cast<const char*>(res.knn.data()), batch * k * sizeof(int));
        if (total_batch >= query_size) break;
        //         const int* qids = res.qids.data();
        // #pragma omp parallel for
        //         for (int j = 0; j < res.batch; ++j) {
        //             memcpy(sample_queries.data() + (start + j) * dimension_,
        //                    data_sq_ + qids[j] * dimension_, dimension_ * sizeof(float));
        //         }
    }
    fo.iprint("Save " + TOS(query_size) + " query data to " + query_file.string() +
              ", total_batch=" + TOS(total_batch) + ", query_size=" + TOS(query_size));
    ofs.write((char*)reinterpret_cast<const char*>(  // sample_queries.data()
                  data_sq_),
              query_size * dimension_ * sizeof(float));

    if (graph) delete graph;
    if (gpufuncs) delete gpufuncs;
    // if (h_queries) CUDA_CHECK(cudaFreeHost(h_queries));

    fo.print("GPU build free.");
}

float IndexGPU::compute_duration_time() {
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::high_resolution_clock::now() - start_time);
    return duration.count();
}

void IndexGPU::LoadVectorData(const char* base_file, const char* sampled_query_file,
                              DIST_METRIC m) {
    size_t base_num = 0, sq_num = 0;
    int base_dim = 0, q_dim = 0;

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

    if (m == DIST_METRIC::IP_) normalize_L2_omp(base_data, base_num, base_dim);

    h_base = data_bp_ = base_data;  // data_align(base_data, base_num, base_dim);

    nd_ = base_num;
    nd_sq_ = sq_num;
    total_pts_ = nd_ + nd_sq_;
    u32_nd_ = static_cast<int>(nd_);
    u32_nd_sq_ = static_cast<int>(nd_sq_);
}

void IndexGPU::Build(size_t n, const float* data, const Parameters& parameters) {}
void IndexGPU::Search(const float* query, const float* x, size_t k, const Parameters& parameters,
                      unsigned* indices, float* res_dists) {}

void IndexGPU::LoadRoarGraph(std::string filename) { graph->LoadRoarGraph(filename.c_str()); }

// void IndexGPU::LoadGraph(const char* filename) {
//     // load graph to projection graph
//     std::ifstream in(filename, std::ios::binary);
//     int npts;
//     in.read((char*)&projection_ep_, sizeof(int));
//     in.read((char*)&npts, sizeof(npts));
//     projection_graph_.resize(npts);
//     projection_graph_deg_.resize(npts);
//     int out_degree = 0;
//     for (int i = 0; i < npts; i++) {
//         int nbr_size;
//         in.read((char*)&nbr_size, sizeof(nbr_size));
//         out_degree += nbr_size;
//         projection_graph_deg_[i] = nbr_size;
//         projection_graph_[i].resize(nbr_size);
//         in.read((char*)projection_graph_[i].data(), nbr_size * sizeof(int));
//     }
//     fo.print("Projection graph, avg_degree: " + TOS(out_degree * 1.0 / npts));
//     in.close();
// }

// void IndexGPU::Build(size_t n, const float* data, const Parameters& parameters) {}

// void IndexGPU::CalculateGraphEP() {
//     float* center = new float[dimension_]();
//     memset(center, 0, sizeof(float) * dimension_);
//     // calculate centroid in base point
//     for (size_t i = 0; i < nd_; ++i) {
//         for (size_t d = 0; d < dimension_; ++d) {
//             center[d] += data_bp_[i * dimension_ + d];
//         }
//     }

//     for (size_t d = 0; d < dimension_; ++d) {
//         center[d] /= (float)nd_;
//     }

//     float* distances = new float[nd_]();
//     memset(distances, 0, sizeof(float) * nd_);
// #pragma omp parallel for
//     for (size_t i = 0; i < nd_; ++i) {
//         const float* cur_data = data_bp_ + i * dimension_;
//         float diff = 0;
//         for (size_t j = 0; j < dimension_; ++j) {
//             diff += ((center[j] - cur_data[j]) * (center[j] - cur_data[j]));
//         }
//         distances[i] = diff;
//     }

//     int closest = 0;
//     for (size_t i = 1; i < nd_; ++i) {
//         if (projection_graph_[i].size() > 0 &&
//             distances[i] < distances[closest]) {  // 孤立节点不可为质心
//             closest = static_cast<int>(i);
//         }
//     }
//     projection_ep_ = closest;
//     delete[] center;
//     delete[] distances;
// }

// void IndexGPU::SaveIndex(const char* filename) {
//     std::ofstream out(filename, std::ios::binary | std::ios::out);
//     if (!out.is_open()) {
//         throw std::runtime_error("cannot open file");
//     }
//     out.write((char*)&projection_ep_, sizeof(int));
//     out.write((char*)&u32_nd_, sizeof(int));
//     for (int i = 0; i < u32_nd_; ++i) {
//         int nbr_size = projection_graph_deg_[i];
//         out.write((char*)&nbr_size, sizeof(int));
//         out.write((char*)projection_graph_[i].data(), sizeof(int) * nbr_size);
//     }
//     out.close();
// }

// std::pair<int, int> IndexGPU::SearchPipe(const float* query, size_t k, size_t& qid,
//                                          const Parameters& parameters, int* indices,
//                                          std::vector<float>& res_dists) {
//     const int L_pq = parameters.Get<int>("L_pq");

//     NeighborPriorityQueue search_queue(L_pq);

//     prefetch_vector((char*)(data_bp_ + projection_ep_ * dimension_), dimension_);
//     VisitedList* vl = visited_list_pool_->getFreeVisitedList();
//     vl_type* visited_array = vl->mass;
//     vl_type visited_array_tag = vl->curV;

//     float distance =
//         distance_->compare(data_bp_ + projection_ep_ * dimension_, query, (unsigned)dimension_);
//     search_queue.insert(Neighbor(projection_ep_, distance, false));

//     int cmps = 0;
//     int hops = 0;
//     while (search_queue.has_unexpanded_node()) {
//         auto cur_id = search_queue.closest_unexpanded().id;

//         int* cur_nbrs = projection_graph_[cur_id].data();
//         ++hops;
//         // get neighbors' neighbors, first
//         for (size_t j = 0; j < projection_graph_deg_[cur_id];
//              ++j) {  // current check node's neighbors
//             int nbr = projection_graph_[cur_id][j];
//             _mm_prefetch((char*)(visited_array + *(cur_nbrs + j + 1)), _MM_HINT_T0);
//             _mm_prefetch((char*)(data_bp_ + *(cur_nbrs + j + 1) * dimension_), _MM_HINT_T0);
//             if (visited_array[nbr] != visited_array_tag) {
//                 visited_array[nbr] = visited_array_tag;
//                 if (nbr < 0 || nbr >= nd_) fo.eprint("Error: nbr out of range: " + TOS(nbr));

//                 float distance =
//                     distance_->compare(data_bp_ + nbr * dimension_, query,
//                     (unsigned)dimension_);

//                 ++cmps;
//                 search_queue.insert({nbr, distance, false});
//             }
//         }
//     }

//     visited_list_pool_->releaseVisitedList(vl);

//     if (unlikely(search_queue.size() < k)) {
//         std::stringstream ss;
//         ss << "not enough results: " << search_queue.size() << ", expected: " << k;
//         throw std::runtime_error(ss.str());
//     }

//     for (size_t i = 0; i < k; ++i) {
//         indices[i] = search_queue[i].id;
//         res_dists[i] = search_queue[i].distance;
//     }
//     return std::make_pair(cmps, hops);
// }