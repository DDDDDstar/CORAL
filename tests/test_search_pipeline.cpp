#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <boost/dynamic_bitset.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "efanna2e/distance.h"
#include "efanna2e/neighbor.h"
#include "efanna2e/parameters.h"
#include "efanna2e/util.h"
#include "index_gpu.h"
#include "uni.h"

namespace po = boost::program_options;
using namespace efanna2e;

float ComputeRecall(int q_num, int k, int gt_dim, int* res, int* gt) {
    int total_count = 0;
    for (int i = 0; i < q_num; i++) {
        std::vector<int> one_gt(gt + i * gt_dim, gt + i * gt_dim + k);
        std::vector<int> intersection;
        std::vector<int> temp_res(res + i * k, res + i * k + k);
        for (auto p : one_gt) {
            if (std::find(temp_res.begin(), temp_res.end(), p) != temp_res.end())
                intersection.push_back(p);
        }

        total_count += static_cast<int>(intersection.size());
    }
    return static_cast<float>(total_count) / (float)(k * q_num);
}

double ComputeRderr(float* gt_dist, int gt_dim, std::vector<std::vector<float>>& res_dists, int k,
                    efanna2e::Metric metric) {
    double total_err = 0;
    int q_num = res_dists.size();

    for (int i = 0; i < q_num; i++) {
        std::vector<float> one_gt(gt_dist + i * gt_dim, gt_dist + i * gt_dim + k);
        std::vector<float> temp_res(res_dists[i].begin(), res_dists[i].end());
        if (metric == efanna2e::INNER_PRODUCT) {
            for (size_t j = 0; j < k; ++j) {
                temp_res[j] = -1.0 * temp_res[j];
            }
        } else if (metric == efanna2e::COSINE) {
            for (size_t j = 0; j < k; ++j) {
                temp_res[j] = 2.0 * (1.0 - (-1.0 * temp_res[j]));
            }
        }
        double err = 0.0;
        for (int j = 0; j < k; j++) {
            err += std::fabs(temp_res[j] - one_gt[j]) / double(one_gt[j]);
        }
        err = err / static_cast<double>(k);
        total_err = total_err + err;
    }
    return total_err / static_cast<double>(q_num);
}

int main(int argc, char** argv) {
    std::string base_data_file;
    std::string query_file;
    std::string sampled_query_data_file;
    std::string gt_file, config_file, res_file;
    fpath log_file;

    // std::string projection_index_save_file;
    std::string data_type;
    std::string dist;
    std::string dataset;
    std::vector<int> L_vec;
    int num_threads;
    int k, max_degree;
    double max_recall, max_hop;
    float recall_thres, query_thres;
    int CE;
    // int roar, para, cdmg;
    std::string index_type;

    po::options_description desc{"Arguments"};
    try {
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                           "data type <int8/uint8/float>");
        desc.add_options()("dist", po::value<std::string>(&dist)->required(),
                           "distance function <l2/ip>");
        desc.add_options()("base_data_path", po::value<std::string>(&base_data_file)->required(),
                           "Input data file in bin format");
        desc.add_options()("dataset", po::value<std::string>(&dataset)->required(),
                           "dataset <t2i>");
        desc.add_options()("recall_thres", po::value<float>(&recall_thres)->default_value(100),
                           "The threshold of the recall of index");
        desc.add_options()("query_path", po::value<std::string>(&query_file)->required(),
                           "Query file in bin format");
        desc.add_options()("gt_path", po::value<std::string>(&gt_file)->required(),
                           "Groundtruth file in bin format");
        desc.add_options()("L", po::value<std::vector<int>>(&L_vec)->multitoken(),
                           "Priority queue length for searching");
        desc.add_options()("max_recall", po::value<double>(&max_recall)->default_value(0),
                           "Max recall required to reach, must > 0.5");
        desc.add_options()("max_hop", po::value<double>(&max_hop)->default_value(1000000000),
                           "Max limited hops");
        desc.add_options()("k", po::value<int>(&k)->default_value(1)->required(),
                           "k nearest neighbors");
        desc.add_options()("max_degree",
                           po::value<int>(&max_degree)->default_value(32)->required(),
                           "max_degree");
        desc.add_options()("query_thres", po::value<float>(&query_thres)->default_value(1.0),
                           "The threshold of the size of query data");
        desc.add_options()("log_file", po::value<fpath>(&log_file)->default_value(""),
                           "log file path");
        desc.add_options()("res_file", po::value<std::string>(&res_file)->default_value(""),
                           "result file path");
        desc.add_options()("config_file",
                           po::value<std::string>(&config_file)->default_value("default"),
                           "config file path");
        desc.add_options()("CE", po::value<int>(&CE)->default_value(1),
                           "Use connectivity enhancement or not");
        desc.add_options()("index_type", po::value<std::string>(&index_type)->default_value(""),
                           "");
        // desc.add_options()("para", po::value<int>(&para)->default_value(0), "");
        // desc.add_options()("cdmg", po::value<int>(&cdmg)->default_value(0), "");

        // desc.add_options()("evaluation_save_path",
        // po::value<std::string>(&evaluation_save_path),
        //                    "Path prefix for saving evaluation results");
        // desc.add_options()("num_threads,T",
        //                    po::value<int>(&num_threads)->default_value(omp_get_num_procs()),
        //                    "Number of threads used for building index (defaults to "
        //                    "omp_get_num_procs())");

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

    size_t base_num, sq_num = 0;
    int base_dim, sq_dim = 0;
    // efanna2e::load_meta<float>(base_data_file.c_str(), base_num, base_dim);
    if (!sampled_query_data_file.empty()) {
        efanna2e::load_meta<float>(sampled_query_data_file.c_str(), sq_num, sq_dim);
    }

    efanna2e::Parameters parameters;

    fpath graph_file, query_cache_file;
    const std::string recall_thres_str = std::to_string(int(recall_thres)),
                      query_thres_str = std::format("{:.1f}", query_thres),
                      dataset_dist = dataset + "_" + dist + "_" + TOS(max_degree);
    fpath gt_dir = fpath(GT_PREFIX) / dataset_dist;
    fpath evaluation_save_file =
        fpath(EVALUATION_PREFIX) / fpath(res_file.empty() ? "search" : res_file) /
        (index_type == "" ? dataset_dist + "_" + query_thres_str + "_" + TOS(CE) + ".csv"
                          : (index_type + "_" + dataset_dist + ".csv"));

    graph_file =
        fpath(INDEX_PREFIX) / (dataset_dist + "_" + query_thres_str + "_" + TOS(CE ? 1 : 0));
    if (log_file.string() == "")
        log_file = fpath(LOG_PREFIX) / "search" / (dataset_dist + "_" + query_thres_str + ".log");
    else
        log_file = fpath(LOG_PREFIX) / log_file;
    query_cache_file = gt_dir / "query.cache";

    parameters.Set<std::string>("dist", dist);
    parameters.Set<int>("k", k);
    parameters.Set<int>("max_degree", max_degree);
    parameters.Set<int>("CE", CE);
    parameters.Set<std::string>("query_file", query_cache_file.string());
    parameters.Set<std::string>("graph_file", graph_file.string());
    parameters.Set<std::string>("base_file", base_data_file);
    parameters.Set<std::string>("log_file", log_file.string());
    parameters.Set<std::string>("config_file", config_file);
    size_t q_pts;
    int q_dim;
    efanna2e::load_meta<float>(query_file.c_str(), q_pts, q_dim);
    float* query_data = nullptr;
    efanna2e::load_data<float>(query_file.c_str(), q_pts, q_dim, query_data);
    if (dist == "ip") normalize_L2_omp(query_data, q_pts, q_dim);

    // efanna2e::load_data<float>(base_data_file.c_str(), base_num, base_dim, base_data);
    // float* aligned_query_data = efanna2e::data_align(query_data, q_pts, q_dim);
    const int copy_n = 1;
    std::vector<float> aligned_query_data(q_pts * q_dim * copy_n);
    memcpy(aligned_query_data.data(), query_data, q_pts * q_dim * sizeof(float));
    for (int i = 1; i < copy_n; i++)
        aligned_query_data.insert(aligned_query_data.end(), aligned_query_data.begin(),
                                  aligned_query_data.begin() + q_pts * q_dim);

    int gt_pts, gt_dim;
    // std::ifstream gtin(gt_file, std::ios::binary);
    // gtin.seekg(sizeof(int) * 2, std::ios::beg);
    // gtin.read((char*)(gt_ids), q_pts * k * sizeof(int));

    efanna2e::load_gt_meta<int>(gt_file.c_str(), gt_pts, gt_dim);
    int* gt_ids = new int[gt_pts * gt_dim];
    float* gt_dists = nullptr;
    efanna2e::load_gt_data<int>(gt_file.c_str(), gt_pts, gt_dim, gt_ids);
    fo.print("gt_pts: " + TOS(gt_pts) + ", gt_dim: " + TOS(gt_dim));
    // efanna2e::load_gt_data_with_dist<int, float>(gt_file.c_str(), gt_pts, gt_dim, gt_ids,
    //                                              gt_dists);
    efanna2e::Metric dist_metric = efanna2e::INNER_PRODUCT;
    DIST_METRIC metric;
    if (dist == "l2") {
        dist_metric = efanna2e::L2;
        metric = DIST_METRIC::L2_;
        std::cout << "Using l2 as distance metric" << std::endl;
    } else if (dist == "ip") {
        dist_metric = efanna2e::INNER_PRODUCT;
        metric = DIST_METRIC::IP_;
        std::cout << "Using inner product as distance metric" << std::endl;
    }
    // } else if (dist == "cosine") {
    //     dist_metric = efanna2e::COSINE;
    //     std::cout << "Using cosine as distance metric" << std::endl;
    // }
    else {
        std::cout << "Unknown distance type: " << dist << std::endl;
        return -1;
    }

    // if (!std::filesystem::exists(projection_index_save_file.c_str())) {
    //     std::cout << "projection index file does not exist." << std::endl;
    //     return -1;
    // }

    efanna2e::IndexGPU index(q_dim, base_num + sq_num, dist_metric, nullptr);

    index.LoadSearchNeededData(base_data_file.c_str(), sampled_query_data_file.c_str(), metric);

    std::cout << "Load graph index: " << graph_file << std::endl;
    index.RunPrepare(parameters, Mode::SEARCH);
    if (index_type == "roar")
        index.LoadRoarGraph(std::string("../data/") + dataset + "/indexes/roar.index");
    else if (index_type == "para")
        index.LoadRoarGraph(std::string("../data/") + dataset + "/indexes/para-t2i-test-5");
    else if (index_type == "cdmg")
        index.LoadRoarGraph(std::string("../data/") + dataset + "/indexes/cdmg.index");
    // if (index.need_normalize) {
    //     std::cout << "Normalizing query data" << std::endl;
    //     for (int i = 0; i < q_pts; i++) {
    //         efanna2e::normalize<float>(aligned_query_data + i * q_dim, q_dim);
    //     }
    // }
    // index.InitVisitedListPool(num_threads);

    // Search
    std::cout << "k: " << k << std::endl;
    // q_pts = 1;
    int* res = new int[q_pts * k * copy_n];
    memset(res, 0, sizeof(int) * q_pts * k * copy_n);
    // std::vector<std::vector<float>> res_dists(q_pts, std::vector<float>(k, 0.0));
    std::ofstream evaluation_out(evaluation_save_file, std::ios::out);
    if (!evaluation_out.is_open()) {
        std::cout << "Unable to open evaluation file: " << evaluation_save_file << std::endl;
        exit(1);
    }
    // std::cout << "Using thread: " << num_threads << std::endl;
    std::cout << "L" << "\t\tQPS" << "\t\tavg_time" << "\t\trecall@" << k << "\t\tavg_hops"
              << std::endl;
    evaluation_out << "L, QPS, time, recall@" << k << ", avg_hops" << std::endl;
    fo.iprint("L, QPS, time, recall@" + TOS(k) + ", avg_hops");
    float recall;
    if (L_vec.size() == 0 && max_recall > 0.5) {
        L_vec.push_back(128);
    }

    // for (int i = 0; i < 3; ++i) {
    for (int i = 0; i < L_vec.size(); ++i) {
        int L = L_vec[i];
        if (k > L) {
            std::cout << "L must greater or equal than k" << std::endl;
            exit(1);
        }

        auto tr = index_type == ""
                      ? index.Search(aligned_query_data.data(), q_pts * copy_n, L, res)
                      : index.Search(aligned_query_data.data(), q_pts * copy_n, L, res, 1);

        const float qps = (float)q_pts * copy_n / tr.get_time(), time = tr.get_time();
        recall = ComputeRecall(q_pts, k, gt_dim, res, gt_ids);
        const float avg_hops = tr.avg_hops(q_pts * copy_n);
        std::cout << L << "\t\t" << qps << "\t\t" << time << "\t\t" << recall << "\t\t" << avg_hops
                  << std::endl;

        evaluation_out << L << "," << qps << "," << time << "," << recall << "," << avg_hops
                       << std::endl;
        fo.print(TOS(L) + ", " + TOS(qps) + ", " + TOS(time) + ", " + TOS(recall) + ", " +
                 TOS(avg_hops));

        if (max_recall > 0.5 && recall < max_recall) {
            L_vec.push_back(128 + L_vec[L_vec.size() - 1]);
        }
        if (i > 50) {
            fo.iprint(
                "Queue is too long, judged as unable to meet the recall requirement, "
                "termination");
            break;
        }
    }
    if (evaluation_out.is_open()) {
        evaluation_out.close();
    }

    delete[] res;
    delete[] gt_ids;

    return 0;
}