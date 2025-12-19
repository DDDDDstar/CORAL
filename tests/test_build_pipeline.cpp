#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <boost/dynamic_bitset.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <cmath>
#include <ctime>
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

int main(int argc, char** argv) {
    std::string base_data_file;
    std::string sampled_query_data_file;
    // std::string query_data_file;
    std::string bipartite_index_save_file, learn_base_nn_file, base_learn_nn_file;
    std::string data_type;
    std::string dist, dataset;
    int M_sq;
    int M_pjbp, L_pjpq;
    // int L_pq;
    int num_threads;
    int CE;
    int k;
    float iso_thres, deg_thres, recall_thres, query_thres;
    std::string cache_file, log_file, graph_file, recall_file, query_file;

    po::options_description desc{"Arguments"};
    try {
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                           "data type <int8/uint8/float>");
        desc.add_options()("dist", po::value<std::string>(&dist)->required(),
                           "distance function <l2/ip>");
        desc.add_options()("dataset", po::value<std::string>(&dataset)->required(),
                           "dataset <t2i>");
        desc.add_options()("base_data_path", po::value<std::string>(&base_data_file)->required(),
                           "Input data file in bin format");
        desc.add_options()("sampled_query_data_path",
                           po::value<std::string>(&sampled_query_data_file)->required(),
                           "Sampled query file in bin format");
        // desc.add_options()("query_data_path",
        //                    po::value<std::string>(&query_data_file)->required(),
        //                    "Query file in bin format");
        // desc.add_options()("bipartite_index_save_path",
        // po::value<std::string>(&bipartite_index_save_file)->required(),
        //    "Path prefix for saving bipartite index file components");
        // desc.add_options()("projection_index_save_path",
        //                    po::value<std::string>(&projection_index_save_file)->required(),
        //                    "Path prefix for saving projetion index file components");
        // desc.add_options()("M_bp", po::value<int>(&M_bp)->default_value(32),
        //    "Number of neighbors for base points to build the bipartite graph");
        desc.add_options()(
            "M_sq", po::value<int>(&M_sq)->default_value(32),
            "Number of neighbors for sampled query points to build the bipartite graph");
        desc.add_options()("M_pjbp", po::value<int>(&M_pjbp)->default_value(32),
                           "Number of neighbors for projection graph");
        desc.add_options()("L_pjpq", po::value<int>(&L_pjpq)->default_value(32),
                           "Priority queue length for projection graph searching");

        // desc.add_options()("L_pq", po::value<int>(&L_pq)->default_value(32),
        //                    "Priority queue length for searching");
        desc.add_options()("num_threads,T",
                           po::value<int>(&num_threads)->default_value(omp_get_num_procs()),
                           "Number of threads used for building index (defaults to "
                           "omp_get_num_procs())");
        // desc.add_options()("learn_base_nn_path",
        // po::value<std::string>(&learn_base_nn_file)->required(),
        //                    "Path of learn-base NN file");
        // desc.add_options()("base_learn_nn_path",
        // po::value<std::string>(&base_learn_nn_file)->required(),
        //                    "Path of base-learn NN file");
        desc.add_options()("CE", po::value<int>(&CE)->default_value(1),
                           "Use connectivity enhancement or not");
        desc.add_options()("k", po::value<int>(&k)->default_value(1)->required(),
                           "k nearest neighbors");
        // desc.add_options()("iso_thres", po::value<float>(&iso_thres)->default_value(0.5),
        //                    "The threshold of the proportion of isolated nodes");
        // desc.add_options()("deg_thres", po::value<float>(&deg_thres)->default_value(5),
        //                    "The threshold of the avg degree of nodes");
        desc.add_options()("recall_thres", po::value<float>(&recall_thres)->default_value(80),
                           "The threshold of the recall of index");
        desc.add_options()("query_thres", po::value<float>(&query_thres)->default_value(10.0),
                           "The threshold of the size of query data");
        // desc.add_options()("cache_file", po::value<std::string>(&cache_file)->required(),
        //                    "file path to cache GT");
        // desc.add_options()("graph_file", po::value<std::string>(&graph_file)->required(),
        //                    "file path to cache graph");
        // desc.add_options()("log_file",
        // po::value<std::string>(&log_file)->default_value("std")->required(),
        //                    "log file path");
        // desc.add_options()("recall_file", po::value<std::string>(&recall_file)->required(),
        //                    "recall record file path");

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
    std::cout << "sampled query: " << sampled_query_data_file << std::endl;
    int base_num, base_dim, sq_num, sq_dim;
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

    float* data_bp = nullptr;
    float* data_sq = nullptr;
    float* aligned_data_bp = nullptr;
    float* aligned_data_sq = nullptr;
    efanna2e::Parameters parameters;
    efanna2e::load_data<float>(base_data_file.c_str(), base_num, base_dim, data_bp);
    efanna2e::load_data<float>(sampled_query_data_file.c_str(), sq_num, sq_dim, data_sq);
    aligned_data_bp = data_bp;
    aligned_data_sq = data_sq;
    // aligned_data_bp = efanna2e::data_align(data_bp, base_num, base_dim);
    // aligned_data_sq = efanna2e::data_align(data_sq, sq_num, sq_dim);

    // if (dist_metric == efanna2e::INNER_PRODUCT) {
    //     efanna2e::ip_normalize(aligned_data_bp, base_dim);
    //     dist_metric = efanna2e::L2;
    // }
    std::string date_time = getCurrentDateTimeString(), version;
#ifdef algo0
    version = "v0";
#endif
#ifdef algo1
    version = "v1";
#endif
#ifdef algo2
#ifdef GIG
    version = "gig";
#else
    version = "v2";
#endif
#endif
    const std::string recall_thres_str = std::to_string(int(recall_thres));
    const std::string gt_dir = GT_PREFIX + dataset + "/" + dist + "_" + recall_thres_str;
    graph_file = INDEX_PREFIX + dataset + "/" + dist + "_" + date_time + "/";
    log_file =
        LOG_PREFIX + dataset + "/" + date_time + "_" + dist + "_" + recall_thres_str + ".log";
    recall_file = RECALL_PREFIX + dataset + "_" + dist + "_" + version + ".csv";
    cache_file = gt_dir + "/knn.cache";
    query_file = gt_dir + "/query.cache";
    efanna2e::fo.print("Index graph save path: " + graph_file + "\nlog path: " + log_file +
                       "\nrecall path: " + recall_file + "\nknn cache path: " + cache_file);

    efanna2e::IndexGPU index(base_dim, base_num + sq_num, dist_metric, nullptr);
    // parameters.Set<int>("M_bp", M_bp);
    parameters.Set<int>("M_sq", M_sq);
    // parameters.Set<int>("L_pq", L_pq);
    parameters.Set<int>("M_pjbp", M_pjbp);
    parameters.Set<int>("L_pjpq", L_pjpq);
    parameters.Set<int>("num_threads", num_threads);
    parameters.Set<int>("CE", CE);
    parameters.Set<int>("k", k);
    parameters.Set<std::string>("cache_file", cache_file);
    parameters.Set<std::string>("base_file", base_data_file);
    parameters.Set<std::string>("graph_file", graph_file);
    parameters.Set<std::string>("query_file", query_file);
    parameters.Set<std::string>("log_file", log_file);
    parameters.Set<std::string>("recall_file", recall_file);
    parameters.Set<std::string>("dist", dist);
    parameters.Set<std::string>("dataset", dataset);
    parameters.Set<float>("iso_thres", iso_thres);
    parameters.Set<float>("deg_thres", deg_thres);
    parameters.Set<float>("recall_thres", recall_thres);
    parameters.Set<float>("query_thres", query_thres);

    omp_set_num_threads(num_threads);
    auto s = std::chrono::high_resolution_clock::now();
    index.BuildGPU(sq_num, aligned_data_sq, base_num, aligned_data_bp, parameters);

    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;

    efanna2e::fo.iprint("indexing time: " + std::to_string(diff.count()));
    // index.SaveIndex(projection_index_save_file.c_str());
    efanna2e::fo.print("Save index to " + graph_file);

    return 0;
}