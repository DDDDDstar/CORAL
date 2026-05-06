#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <boost/dynamic_bitset.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "efanna2e/distance.h"
#include "efanna2e/neighbor.h"
#include "efanna2e/parameters.h"
#include "efanna2e/util.h"
#include "fileout.h"
#include "index_gpu.h"
#include "uni.h"

namespace po = boost::program_options;
using namespace efanna2e;

typedef faiss::gpu::StandardGpuResources GPUResources;
typedef faiss::gpu::GpuIndexFlatL2 GPUFlatL2;

int main(int argc, char** argv) {
    std::string base_data_file, sampled_query_data_file, update_gt_file, query_file, recall_file,
        config_file;
    std::string data_type;
    std::string dist, dataset;
    int max_degree;
    int k;
    std::string graph_file, query_cache_file, log_file, cache_file;
    float recall_thres, query_thres;
    // std::string evaluation_save_file;

    po::options_description desc{"Arguments"};
    try {
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("dataset", po::value<std::string>(&dataset)->required(),
                           "dataset <t2i>");
        desc.add_options()("base_data_path", po::value<std::string>(&base_data_file)->required(),
                           "Input data file in bin format");
        desc.add_options()("update_gt_file", po::value<std::string>(&update_gt_file)->required(),
                           "update gt file in bin format");
        desc.add_options()("sampled_query_data_path",
                           po::value<std::string>(&sampled_query_data_file)->required(),
                           "Sampled query file in bin format");
        desc.add_options()("dist", po::value<std::string>(&dist)->default_value("l2")->required(),
                           "distance function <l2/ip>");
        desc.add_options()("data_type",
                           po::value<std::string>(&data_type)->default_value("float")->required(),
                           "data type <int8/uint8/float>");
        desc.add_options()("recall_thres", po::value<float>(&recall_thres)->default_value(80),
                           "The threshold of the recall of index");
        desc.add_options()("query_thres", po::value<float>(&query_thres)->default_value(10.0),
                           "The threshold of the size of query data");
        desc.add_options()("max_degree", po::value<int>(&max_degree)->default_value(2)->required(),
                           "Number of neighbors for graph");
        desc.add_options()("k", po::value<int>(&k)->default_value(1)->required(),
                           "k nearest neighbors");
        desc.add_options()("query_path", po::value<std::string>(&query_file)->required(),
                           "Query file in bin format");

        desc.add_options()("config_file",
                           po::value<std::string>(&config_file)->default_value("default"),
                           "config file");
        // desc.add_options()("log_file", po::value<std::string>(&log_file)->default_value(""),
        //                    "log file path");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return -1;
    }
    size_t base_num, sq_num, query_num;
    int base_dim, sq_dim, query_dim;
    efanna2e::load_meta<float>(base_data_file.c_str(), base_num, base_dim);
    efanna2e::load_meta<float>(sampled_query_data_file.c_str(), sq_num, sq_dim);
    efanna2e::Metric dist_metric = efanna2e::INNER_PRODUCT;
    if (dist == "l2") {
        dist_metric = efanna2e::L2;
        std::cout << "Using l2 as distance metric" << std::endl;
    } else if (dist == "ip") {
        dist_metric = efanna2e::INNER_PRODUCT;
        std::cout << "Using inner product as distance metric" << std::endl;
    } else if (dist == "cosine") {
        dist_metric = efanna2e::COSINE;
        std::cout << "Using cosine as distance metric" << std::endl;
    } else {
        std::cout << "Unknown distance type: " << dist << std::endl;
        return -1;
    }

    float *data_bp = nullptr, *data_sq = nullptr, *aligned_data_bp = nullptr,
          *aligned_data_sq = nullptr;
    efanna2e::Parameters parameters;
    efanna2e::load_data<float>(base_data_file.c_str(), base_num, base_dim, data_bp);
    efanna2e::load_data<float>(sampled_query_data_file.c_str(), sq_num, sq_dim, data_sq);
    aligned_data_bp = efanna2e::data_align(data_bp, base_num, base_dim);
    aligned_data_sq = efanna2e::data_align(data_sq, sq_num, sq_dim);

    efanna2e::load_meta<float>(query_file.c_str(), query_num, query_dim);
    float* query_data = nullptr;
    efanna2e::load_data<float>(query_file.c_str(), query_num, query_dim, query_data);
    float* aligned_query_data = efanna2e::data_align(query_data, query_num, query_dim);

    // if (dist_metric == efanna2e::INNER_PRODUCT) {
    //     efanna2e::ip_normalize(aligned_data_bp, base_dim);
    //     dist_metric = efanna2e::L2;
    // }
    const size_t total_num = base_num, upd_num = base_num = base_num / 2;
    efanna2e::IndexGPU index(base_dim, base_num + sq_num, dist_metric, nullptr);
    index.load_config(config_file);

    std::string date_time = efanna2e::getCurrentDateTimeString();
    const std::string recall_thres_str = std::to_string(int(recall_thres)),
                      dataset_dist = dataset + "_" + dist + "_" + TOS(max_degree) + "_upd",
                      gt_dir = GT_PREFIX + dataset_dist;
    create_dir(fpath(gt_dir));
    graph_file = INDEX_PREFIX + dataset_dist;
    query_cache_file = gt_dir + "/query.cache";
    log_file = UPD_LOG_PREFIX + dataset + "_" + dist + "_" + TOS(PC.upd_mode) + "_" +
               TOS(PC.upd_type) + ".log";
    // evaluation_save_file = UPD_EVALUATION_PREFIX + dataset + "/" + date_time + "_" + dist +
    // ".csv";
    recall_file = RECALL_PREFIX + dataset + "_" + dist + "_upd.csv";
    cache_file = gt_dir + "/knn.cache";

    parameters.Set<int>("max_degree", max_degree);
    parameters.Set<int>("k", k);
    parameters.Set<int>("CE", 1);
    parameters.Set<std::string>("base_file", base_data_file);
    parameters.Set<std::string>("graph_file", graph_file);
    parameters.Set<std::string>("query_file", query_cache_file);
    parameters.Set<std::string>("cache_file", cache_file);
    parameters.Set<std::string>("log_file", log_file);
    parameters.Set<std::string>("recall_file", recall_file);
    parameters.Set<std::string>("config_file", config_file);
    parameters.Set<std::string>("dist", dist);
    parameters.Set<std::string>("dataset", dataset);
    parameters.Set<float>("recall_thres", recall_thres);
    parameters.Set<float>("query_thres", query_thres);

    auto s = std::chrono::high_resolution_clock::now();
    namespace fs = std::filesystem;
    fpath evaluation_save_file =
        fpath(EVALUATION_PREFIX) / "upd" / (TOS(PC.upd_mode) + "_" + TOS(PC.upd_type) + ".csv");

    if (PC.upd_mode == 1)
        update_gt_file += ".del";
    else if (PC.upd_mode == 2)
        update_gt_file += ".ins";
    std::ifstream gt_file(update_gt_file, std::ios::binary);
    std::ofstream eva_file(evaluation_save_file, std::ios::out);
    if (!gt_file.is_open()) {
        std::cout << "Failed to open update_gt_file: " << update_gt_file << std::endl;
        exit(-1);
    }
    if (!eva_file.is_open()) {
        std::cout << "Failed to open evaluation_save_file: " << evaluation_save_file << std::endl;
        exit(-1);
    }
    int round;
    gt_file.read(reinterpret_cast<char*>(&round), sizeof(int));
    std::vector<int> gts(query_num * k * (round + 1));
    gt_file.read(reinterpret_cast<char*>(gts.data()), sizeof(int) * query_num * k * (round + 1));
    eva_file << "round, recall, avg_hops, time, qps" << std::endl;
    // try {
    // index.BuildGPU(sq_num, aligned_data_sq, base_num, aligned_data_bp, parameters);
    // } catch (const std::exception& e) {
    //     std::cerr << "BuildGPU exception: " << e.what() << std::endl;
    //     throw;
    // }
    if (PC.upd_build) {
        fo.iprint("Building initial index...");
        index.BuildGPU(sq_num, aligned_data_sq, base_num, aligned_data_bp, parameters);
    }

    index.RunPrepare(parameters, Mode::UPD, base_num, aligned_data_bp);
    auto tr = index.Search_and_Verify(aligned_query_data, gts.data(), query_num);
    eva_file << "0, " << tr.calc_recall(query_num * k) << ", " << tr.avg_hops(query_num) << ", "
             << tr.get_time() << ", " << query_num / tr.get_time() << std::endl;
    fo.iprint("Search result: recall=" + TOS(tr.calc_recall(query_num * k)) +
              ", avg_hops=" + TOS(tr.avg_hops(query_num)) + ", time=" + TOS(tr.get_time()) + "s" +
              ", qps=" + TOS(query_num / tr.get_time()));
    const size_t batch = (upd_num + round - 1) / round;
    std::vector<int> del_ids;
    if (PC.upd_mode == 0) {
        for (int batch_i = 0; batch_i < round; batch_i++) {
            const size_t upd_n = std::min(batch, upd_num - batch_i * batch),
                         insert_start_id = base_num + batch_i * batch;
            del_ids.resize(upd_n);
            std::iota(del_ids.begin(), del_ids.end(), batch_i * batch);
            index.Delete(del_ids);
            // auto tr = index.Search_and_Verify(
            //     aligned_query_data, gts.data() + query_num * k * (batch_i + 1), query_num);
            index.Insert(aligned_data_bp + insert_start_id * base_dim, upd_n);
            // gt_file.read(reinterpret_cast<char*>(gts.data()), sizeof(int) * query_num * k);
            if ((batch_i + 1) % 4 != 0) continue;

            auto tr = index.Search_and_Verify(
                aligned_query_data, gts.data() + query_num * k * (batch_i + 1), query_num);
            eva_file << batch_i + 1 << ", " << tr.calc_recall(query_num * k) << ", "
                     << tr.avg_hops(query_num) << ", " << tr.get_time() << ", "
                     << query_num / tr.get_time() << std::endl;
            fo.iprint("Batch " + TOS(batch_i) + "/" + TOS(round) +
                      " Search result: recall=" + TOS(tr.calc_recall(query_num * k)) +
                      "%, avg_hops=" + TOS(tr.avg_hops(query_num)) + ", time=" +
                      TOS(tr.get_time()) + "s" + ", qps=" + TOS(query_num / tr.get_time()));
        }
    } else if (PC.upd_mode == 4) {
        for (int batch_i = 0; batch_i < round; batch_i++) {
            const size_t upd_n = std::min(batch, upd_num - batch_i * batch),
                         insert_start_id = base_num + batch_i * batch;
            del_ids.resize(upd_n);
            std::iota(del_ids.begin(), del_ids.end(), batch_i * batch);
            index.Delete(del_ids);
            auto dtr = index.Search_and_Verify(
                aligned_query_data, gts.data() + query_num * k * (batch_i + 1) * 2, query_num);
            fo.iprint(
                "Batch " + TOS(batch_i) + "/" + TOS(round) +
                " Search result after delete: recall=" + TOS(dtr.calc_recall(query_num * k)) +
                "%, avg_hops=" + TOS(dtr.avg_hops(query_num)) + ", time=" + TOS(dtr.get_time()) +
                "s");
            index.Insert(aligned_data_bp + insert_start_id * base_dim, upd_n);
            // gt_file.read(reinterpret_cast<char*>(gts.data()), sizeof(int) * query_num * k);
            auto tr = index.Search_and_Verify(
                aligned_query_data, gts.data() + query_num * k * (batch_i + 1) * 2 + query_num * k,
                query_num);
            eva_file << batch_i + 1 << ", " << tr.calc_recall(query_num * k) << ", "
                     << tr.avg_hops(query_num) << ", " << tr.get_time() / query_num << ", "
                     << query_num / tr.get_time() << "\n";
            fo.iprint("Batch " + TOS(batch_i) + "/" + TOS(round) +
                      " Search result: recall=" + TOS(tr.calc_recall(query_num * k)) +
                      "%, avg_hops=" + TOS(tr.avg_hops(query_num)) +
                      ", time=" + TOS(tr.get_time()) + "s");
        }
    } else if (PC.upd_mode == 1) {
        for (int batch_i = 0; batch_i < round / 2; batch_i++) {
            const size_t upd_n = std::min(batch, upd_num - batch_i * batch);
            del_ids.resize(upd_n);
            std::iota(del_ids.begin(), del_ids.end(), batch_i * batch);
            index.Delete(del_ids);
            // index.Insert(aligned_data_bp + insert_start_id * base_dim, upd_n);
            // gt_file.read(reinterpret_cast<char*>(gts.data()), sizeof(int) * query_num * k);
            auto tr = index.Search_and_Verify(
                aligned_query_data, gts.data() + query_num * k * (batch_i + 1), query_num);
            eva_file << batch_i + 1 << ", " << tr.calc_recall(query_num * k) << ", "
                     << tr.avg_hops(query_num) << ", " << tr.get_time() / query_num << ", "
                     << query_num / tr.get_time() << "\n";
            fo.iprint("Batch " + TOS(batch_i) + "/" + TOS(round / 2) +
                      " Search result: recall=" + TOS(tr.calc_recall(query_num * k)) +
                      "%, avg_hops=" + TOS(tr.avg_hops(query_num)) +
                      ", time=" + TOS(tr.get_time()) + "s");
        }
    } else {
        for (int batch_i = 0; batch_i < round; batch_i++) {
            const size_t upd_n = std::min(batch, upd_num - batch_i * batch),
                         insert_start_id = base_num + batch_i * batch;
            index.Insert(aligned_data_bp + insert_start_id * base_dim, upd_n);
            // gt_file.read(reinterpret_cast<char*>(gts.data()), sizeof(int) * query_num * k);
            auto tr = index.Search_and_Verify(
                aligned_query_data, gts.data() + query_num * k * (batch_i + 1), query_num);
            eva_file << batch_i + 1 << ", " << tr.calc_recall(query_num * k) << ", "
                     << tr.avg_hops(query_num) << ", " << tr.get_time() / query_num << ", "
                     << query_num / tr.get_time() << "\n";
            fo.iprint("Batch " + TOS(batch_i) + "/" + TOS(round) +
                      " Search result: recall=" + TOS(tr.calc_recall(query_num * k)) +
                      "%, avg_hops=" + TOS(tr.avg_hops(query_num)) +
                      ", time=" + TOS(tr.get_time()) + "s");
        }
    }
    std::cout << "Finished\n";

    return 0;
}