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

#include "cache.h"
#include "candidates.h"
#include "efanna2e/exceptions.h"
#include "efanna2e/parameters.h"
#include "fileout.h"
#include "index_gpu.h"
#include "test.h"

using namespace efanna2e;

void IndexGPU::update_graph(Cache &cache, std::vector<std::shared_mutex> &graph_mtx,
                            KNN_Queue &knn_queue) {
    const auto iso_thres = parameters_.Get<float>("iso_thres");
    const auto deg_thres = parameters_.Get<float>("deg_thres");
    const auto recall_thres = parameters_.Get<float>("recall_thres");
    const auto query_thres = parameters_.Get<float>("query_thres");
    const auto max_degree = parameters_.Get<int>("M_pjbp");
    const auto recall_file = parameters_.Get<std::string>("recall_file");
    int old_processed = processed, update_id = 0, iso_num = 0;
    float recall, cost_time = 0.0f, avg_hops, query_data_size;
    while (true) {
#ifdef GIG
        {
            std::unique_lock<std::shared_mutex> lock(queue_mtx);
            cv.wait(lock, [this] { return gui.vaild(); });
            if (!gui.vaild()) continue;
        }
        auto s = std::chrono::high_resolution_clock::now();

        GDD gdd = gpufuncs->get_graph();
        Test_Result tr;
        {
            std::shared_lock<std::shared_mutex> lock(gdd.mtx);
            for (int i = 0; i < TEST_SEARCH_QUERY_SIZE; i += BATCH)
                tr.accumulate(gpufuncs->beam_search_and_verify(i, BATCH));
        }

        auto duration = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::high_resolution_clock::now() - s);
        cost_time = duration.count();
        recall = tr.calc_recall(TEST_SEARCH_QUERY_SIZE * k);
        avg_hops = tr.hops / TEST_SEARCH_QUERY_SIZE;
        query_data_size = processed / 1000000.0;
        GSD gsd = gui.get_GSD();
        iso_num = gsd.iso_num;

        fo.iprint("Graph test with batch_id: " + TOS(gui.get_batch_id()) +
                  ", recall: " + TOS(recall) + "%, avg_hops: " + TOS(avg_hops) +
                  ", iso_num: " + TOS(iso_num) + ", total_degree: " + TOS(gsd.total_degree) +
                  ", cost_time: " + TOS(cost_time) + "s");
        gui.reset();

        /* CPU test
        {
            gdd = gpufuncs->get_graph();    // 重新获取 ep
            std::shared_lock<std::shared_mutex> lock(gdd.mtx);
            test_search(gdd.graph, gdd.graph_deg, gdd.ep);
        }*/
#else
        Graph_Update_Info gui;
        {
            std::unique_lock<std::shared_mutex> lock(queue_mtx);
            cv.wait(lock, [this] { return !graph_update_queue.empty(); });
            if (graph_update_queue.empty()) continue;

            gui = std::move(graph_update_queue.front());
            graph_update_queue.pop();
        }

        const int old_total_degree = total_degree, old_noniso_num = nonisolated_num;
        std::vector<uint8_t> visited(u32_nd_, 0);
        int visited_num = 0;
        for (int i = 0; i < gui.batch * k; i++) {
            const int pivot_id = gui.knn_ids[i];

            if (visited[pivot_id] !=
                0)  // 在同一批次中某个节点出现了多次，则只处理第一次对应的图更新
                continue;

            visited[pivot_id] = 1;
            visited_num++;
            if (pivot_id < 0 || pivot_id >= u32_nd_)
                fo.eprint("Invalid pivot_id: " + TOS(pivot_id));

            std::unique_lock<std::shared_mutex> lock(graph_mtx[pivot_id]);

            int old_size = projection_graph_deg_[pivot_id];
            if (old_size == 0) {
                projection_graph_[pivot_id].reserve(max_degree);
                nonisolated_num++;
            }

            if (old_size < max_degree) {
                projection_graph_[pivot_id].assign(max_degree, -1);
                projection_graph_dist_[pivot_id].resize(max_degree);
            }
            memcpy(projection_graph_[pivot_id].data(), gui.new_nbr_ids.get() + i * max_degree,
                   max_degree * sizeof(int));
            memcpy(projection_graph_dist_[pivot_id].data(),
                   gui.new_nbr_dists.get() + i * max_degree, max_degree * sizeof(float));

            int true_size = 0;
            while (true_size < max_degree && projection_graph_[pivot_id][true_size] >= 0)
                true_size++;

            if (true_size == 0)
                fo.eprint("Invalid true_size with pivot_id: " + TOS(pivot_id) +
                          " batch_id: " + TOS(gui.batch_id));

            projection_graph_deg_[pivot_id] = true_size;

            total_degree += true_size - old_size;
        }

        const int delta_degree = total_degree - old_total_degree;
        const int delta_iso_num = nonisolated_num - old_noniso_num;
        const float avg_degree = total_degree * 1.0 / u32_nd_;
        std::string iso_ratio_str = TOS(100.0 * nonisolated_num / u32_nd_);
        const int processed = gui.batch_id * BATCH;
        query_data_size = processed / 1000000.0;
        fo.print(TOS(update_id++) + ". Graph updated with non-isolated ratio = " +
                 iso_ratio_str.substr(0, iso_ratio_str.find(".") + 3) +
                 "%, processed: " + TOS(query_data_size) + " M, avg_degree: " + TOS(avg_degree) +
                 ", delta_degree: " + TOS(delta_degree) +
                 ", delta_iso_num: " + TOS(delta_iso_num) + "/" + TOS(visited_num));

        if (processed - old_processed < 10000) continue;
        old_processed = processed;

        int batch_id = gui.batch_id, n = nd_;
        float duration_time = compute_duration_time();
        auto res = test_search();
        recall = res.get_recall();
        avg_hops = res.get_hops();
        cost_time = res.get_time();
        iso_num = u32_nd_ - nonisolated_num;

        cache.SaveGraphCache(projection_graph_, projection_graph_dist_, projection_graph_deg_,
                             graph_mtx, batch_id, duration_time, res, n);
#endif
        gpufuncs->print_times();

        writeCSV(
            {{TOS(query_data_size), TOS(recall), TOS(avg_hops), TOS(cost_time), TOS(iso_num)}},
            recall_file);

        if (recall >= recall_thres) {
            stop_flag.store(true);
            knn_queue.Stop();
            fo.print("recall > threshold, stopping.");
            return;
        }
        if (query_data_size >= query_thres) {
            stop_flag.store(true);
            knn_queue.Stop();
            fo.print("query_data_size > threshold, stopping.");
            return;
        }
    }
}

void IndexGPU::test_search_prepare(GPUFuncs *gpufuncs) {
    // 随机采样获取 TEST_SEARCH_QUERY_SIZE 个查询用于测试：
    const int step = nd_sq_ / TEST_SEARCH_QUERY_SIZE;
    std::fstream file;
    std::string test_example_str;
    const auto recall_file = parameters_.Get<std::string>("recall_file");
    test_query_data.resize(TEST_SEARCH_QUERY_SIZE * dimension_);
    if (std::filesystem::exists(recall_file)) {
        fo.print("Read test query data...");
        file.open(recall_file, std::ios::in);
        if (file.is_open()) {
            std::string line;
            std::getline(file, line);
            fo.print("Test examples: " + line);
            std::istringstream iss(line);
            for (int i = 0; i < TEST_SEARCH_QUERY_SIZE; i++) {
                std::string idx;
                std::getline(iss, idx, ',');
                memcpy(test_query_data.data() + i * dimension_,
                       data_sq_ + std::stoi(idx) * dimension_, dimension_ * sizeof(float));
                test_example_str += idx + " ";
            }
        } else
            fo.eprint("Failed to open recall file!");
    } else {
        fo.print("Generate test query data...");
        file.open(recall_file, std::ios::out);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(0, step - 1);
        for (int i = 0; i < TEST_SEARCH_QUERY_SIZE; i++) {
            const int sample_idx = i * step + dist(gen);
            if (sample_idx >= nd_sq_)
                fo.eprint("Error: sample_idx out of range: " + TOS(sample_idx) +
                          " \nnd_sq:" + TOS(nd_sq_) + ", step:" + TOS(step) + ", i:" + TOS(i));
            file << sample_idx << ",";
            memcpy(test_query_data.data() + i * dimension_, data_sq_ + sample_idx * dimension_,
                   dimension_ * sizeof(float));
            test_example_str += TOS(sample_idx) + " ";
        }
        file << "\n";
    }
    fo.print("test_example:\n" + test_example_str);

    // 计算查询测试的 gt knns：
    float time;
    test_query_knns.resize(TEST_SEARCH_QUERY_SIZE * k);
    for (int batch_i = 0; batch_i < TEST_SEARCH_QUERY_SIZE / BATCH; batch_i++)
        CUDA_CHECK(
            cudaMemcpy(test_query_knns.data() + batch_i * BATCH * k,
                       gpufuncs->knn_compute(test_query_data.data() + batch_i * BATCH * dimension_,
                                             BATCH, time),
                       BATCH * k * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < TEST_SEARCH_QUERY_SIZE * k; i++) {
        if (test_query_knns[i] < 0 || test_query_knns[i] >= nd_)
            fo.eprint("Error: test_query_knns out of range: " + TOS(test_query_knns[i]) +
                      " with i = " + TOS(i));
    }

    std::string example_str;
    for (int i = 0; i < TEST_SEARCH_QUERY_SIZE; i++)
        std::sort(test_query_knns.begin() + i * k, test_query_knns.begin() + (i + 1) * k);

    for (int i = 0; i < k; i++) example_str += TOS(test_query_knns[i]) + " ";
    fo.print("Example query_knns:\n" + example_str);
#ifdef GIG
    gpufuncs->test_query_data_prepare(test_query_data.data(), test_query_knns.data(),
                                      TEST_SEARCH_QUERY_SIZE);
#endif
}

void IndexGPU::test_search(int *d_graph, int *d_graph_deg, int *d_graph_ep) {
    auto s = std::chrono::high_resolution_clock::now();
    const uint32_t L_pq = TEST_SEARCH_L;
    const auto max_degree = parameters_.Get<int>("M_pjbp");
    int hit_num = 0, cmps = 0, hops = 0, graph_ep;
    std::vector<uint8_t> visited_array(nd_);
    std::vector<int> test_graph(nd_ * max_degree);
    std::vector<int> test_graph_deg(nd_);
    CUDA_CHECK(cudaMemcpy(test_graph.data(), d_graph, nd_ * max_degree * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(
        cudaMemcpy(test_graph_deg.data(), d_graph_deg, nd_ * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&graph_ep, d_graph_ep, sizeof(int), cudaMemcpyDeviceToHost));
    prefetch_vector((char *)(data_bp_ + graph_ep * dimension_), dimension_);

    for (int qid = 0; qid < TEST_SEARCH_QUERY_SIZE; qid++) {
        float *query = test_query_data.data() + qid * dimension_;
        memset(visited_array.data(), 0, nd_ * sizeof(uint8_t));
        NeighborPriorityQueue search_queue(L_pq);
        search_queue.insert(Neighbor(
            graph_ep,
            distance_->compare(data_bp_ + graph_ep * dimension_, query, (unsigned)dimension_),
            false));

        while (search_queue.has_unexpanded_node()) {
            auto cur_id = search_queue.closest_unexpanded().id;
            int *cur_nbrs = test_graph.data() + cur_id * max_degree;
            ++hops;
            // get neighbors' neighbors, first
            for (size_t j = 0; j < test_graph_deg[cur_id];
                 ++j) {  // current check node's neighbors
                int nbr = test_graph[cur_id * max_degree + j];
                assert(nbr >= 0 && nbr < nd_);
                _mm_prefetch(
                    (char *)(data_bp_ + test_graph[cur_id * max_degree + j + 1] * dimension_),
                    _MM_HINT_T0);

                if (visited_array[nbr] == 0) {
                    visited_array[nbr] = 1;
                    ++cmps;
                    search_queue.insert({nbr,
                                         distance_->compare(data_bp_ + nbr * dimension_, query,
                                                            (unsigned)dimension_),
                                         false});
                }
            }
        }
        if (unlikely(search_queue.size() < k))
            fo.eprint("not enough results for test_search: " + TOS(search_queue.size()) +
                      ", expected: " + TOS(k));

        for (int i = 0; i < k; i++) {
            int res = search_queue[i].id;
            for (int j = 0; j < k; j++) {
                if (res == test_query_knns[qid * k + j]) {
                    hit_num++;
                    break;
                }
            }
        }
    }

    auto duration = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::high_resolution_clock::now() - s);
    float recall = hit_num * 100.0 / (TEST_SEARCH_QUERY_SIZE * k);
    float avg_hops = hops * 1.0 / TEST_SEARCH_QUERY_SIZE;
    fo.iprint("CPU test_search done\nrecall: " + TOS(recall) + " %, graph_ep: " + TOS(graph_ep) +
              ", hops: " + TOS(avg_hops) + ", cmps: " + TOS(cmps * 1.0 / TEST_SEARCH_QUERY_SIZE) +
              ", test time: " + TOS(duration.count()) + "s");
}

Test_Result IndexGPU::test_search() {
    CalculateGraphEP();
    auto s = std::chrono::high_resolution_clock::now();

    const uint32_t L_pq = TEST_SEARCH_L;
    prefetch_vector((char *)(data_bp_ + projection_ep_ * dimension_), dimension_);
    std::vector<uint8_t> visited_array(nd_);

    std::vector<std::vector<int>> &test_graph = projection_graph_;
    std::vector<int> test_graph_deg = projection_graph_deg_;

    int hit_num = 0, cmps = 0, hops = 0;
    for (int qid = 0; qid < TEST_SEARCH_QUERY_SIZE; qid++) {
        float *query = test_query_data.data() + qid * dimension_;
        memset(visited_array.data(), 0, nd_ * sizeof(uint8_t));
        NeighborPriorityQueue search_queue(L_pq);
        search_queue.insert(Neighbor(projection_ep_,
                                     distance_->compare(data_bp_ + projection_ep_ * dimension_,
                                                        query, (unsigned)dimension_),
                                     false));

        while (search_queue.has_unexpanded_node()) {
            auto cur_id = search_queue.closest_unexpanded().id;
            int *cur_nbrs = test_graph[cur_id].data();
            ++hops;
            // get neighbors' neighbors, first
            for (size_t j = 0; j < test_graph_deg[cur_id];
                 ++j) {  // current check node's neighbors
                int nbr = test_graph[cur_id][j];
                if (nbr < 0 || nbr >= nd_) fo.eprint("Error: nbr out of range: " + TOS(nbr));

                _mm_prefetch((char *)(data_bp_ + test_graph[cur_id][j + 1] * dimension_),
                             _MM_HINT_T0);

                if (visited_array[nbr] == 0) {
                    visited_array[nbr] = 1;

                    ++cmps;

                    search_queue.insert({nbr,
                                         distance_->compare(data_bp_ + nbr * dimension_, query,
                                                            (unsigned)dimension_),
                                         false});
                }
            }
        }

        if (unlikely(search_queue.size() < k))
            fo.iprint("not enough results for test_search: " + TOS(search_queue.size()) +
                      ", expected: " + TOS(k));

        for (int i = 0; i < k; i++) {
            int res = search_queue[i].id;
            for (int j = 0; j < k; j++) {
                if (res == test_query_knns[qid * k + j]) {
                    hit_num++;
                    break;
                }
            }
        }
    }

    auto duration = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::high_resolution_clock::now() - s);
    float recall = hit_num * 100.0 / (TEST_SEARCH_QUERY_SIZE * k);
    float avg_hops = hops * 1.0 / TEST_SEARCH_QUERY_SIZE;
    fo.iprint("test_search done\nrecall: " + TOS(recall) +
              " %, projection_ep_: " + TOS(projection_ep_) + ", hops: " + TOS(avg_hops) +
              ", cmps: " + TOS(cmps * 1.0 / TEST_SEARCH_QUERY_SIZE) +
              ", test time: " + TOS(duration.count()) + "s");

    return Test_Result(recall, avg_hops, duration.count());
}
