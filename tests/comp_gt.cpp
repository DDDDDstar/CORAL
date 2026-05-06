#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
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

using namespace efanna2e;

namespace po = boost::program_options;

typedef faiss::gpu::StandardGpuResources GPUResources;
typedef faiss::gpu::GpuIndexFlatL2 GPUFlatL2;

int main(int argc, char** argv) {
    std::string base_data_file;
    std::string query_data_file;
    std::string gt_save_file;
    std::string data_type;
    std::string dist, dataset;
    int M_sq;
    int M_pjbp, L_pjpq;
    // int L_pq;
    int num_threads;
    int CE;
    int k, nq;
    float iso_thres, deg_thres, recall_thres, query_thres;
    std::string cache_file, log_file, graph_file, recall_file, query_file;

    po::options_description desc{"Arguments"};
    try {
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("data_type", po::value<std::string>(&data_type)->default_value("float"),
                           "data type <int8/uint8/float>");
        desc.add_options()("dist", po::value<std::string>(&dist)->default_value("l2"),
                           "distance function <l2/ip>");
        desc.add_options()("base_data_path", po::value<std::string>(&base_data_file)->required(),
                           "Input data file in bin format");
        desc.add_options()("query_data_path", po::value<std::string>(&query_data_file)->required(),
                           "Sampled query file in bin format");
        desc.add_options()("gt_save_file", po::value<std::string>(&gt_save_file)->required(),
                           "GT result file in bin format");
        desc.add_options()("k", po::value<int>(&k)->default_value(10)->required(),
                           "k nearest neighbors");
        desc.add_options()("nq", po::value<int>(&nq)->default_value(0)->required(), "nq");

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
    std::cout << "sampled query: " << query_data_file << std::endl;
    size_t base_num, sq_num;
    int base_dim, sq_dim;
    load_meta<float>(base_data_file.c_str(), base_num, base_dim);
    load_meta<float>(query_data_file.c_str(), sq_num, sq_dim);
    if (nq > 0) sq_num = nq;

    float* data_bp = nullptr;
    float* data_sq = nullptr;
    // float* aligned_data_bp = nullptr;
    // float* aligned_data_sq = nullptr;
    Parameters parameters;
    load_data<float>(query_data_file.c_str(), sq_num, sq_dim, data_sq);
    // if (base_num <= BLOCK_SIZE)
    load_data<float>(base_data_file.c_str(), base_num, base_dim, data_bp);
    if (dist == "ip") {
        normalize_L2_omp(data_bp, base_num, base_dim);
        normalize_L2_omp(data_sq, sq_num, sq_dim);
    }

    Metric dist_metric = INNER_PRODUCT;
    DIST_METRIC m;
    if (dist == "l2") {
        dist_metric = L2;
        m = DIST_METRIC::L2_;
        std::cout << "Using l2 as distance metric" << std::endl;
    } else if (dist == "ip") {
        dist_metric = INNER_PRODUCT;
        m = DIST_METRIC::IP_;
        std::cout << "Using inner product as distance metric" << std::endl;
    } else if (dist == "cosine") {
        dist_metric = COSINE;
        std::cout << "Using cosine as distance metric" << std::endl;
    } else {
        std::cout << "Unknown distance type: " << dist << std::endl;
        return -1;
    }
    std::cout << "base_num: " << base_num << " sq_num: " << sq_num << " dim: " << base_dim
              << std::endl;

    // GPUResources gpu_res;
    // gpu_res.noTempMemory();
    // GPUFlatL2 gpu_index(&gpu_res, base_dim);
    MyFaiss* myfaiss;
    // ExactKnnStreamer* knn_streamer;
    myfaiss = new MyFaiss(base_dim, k, m, base_num <= BLOCK_SIZE);
    // if (base_num <= BLOCK_SIZE)
    //     myfaiss = new MyFaiss(base_dim, k, m);
    // else {
    //     ExactKnnStreamerConfig config;
    //     config.dim = base_dim;
    //     config.k = k;
    //     config.metric = m;
    //     config.useFloat16Storage = false;
    //     config.max_qbatch = 2048;
    //     config.gpu_budget_bytes = size_t(30ull) * 1024 * 1024 * 1024;
    //     knn_streamer = new ExactKnnStreamer(data_bp, base_num, config);
    // }
    std::vector<float> knn_dists(sq_num * k);
    std::vector<faiss::idx_t> knn_ids(sq_num * k);
    std::vector<int> knn_ids_int(sq_num * k);
    std::ofstream gt_out(gt_save_file, std::ios::binary);
    if (!gt_out.is_open()) {
        std::cout << "open gt_save_file error" << std::endl;
        exit(-1);
    }
    gt_out.write(reinterpret_cast<char*>(&sq_num), sizeof(int));
    gt_out.write(reinterpret_cast<char*>(&k), sizeof(int));
    auto s = std::chrono::high_resolution_clock::now();
    myfaiss->Add(data_bp, base_num);

    std::cout << "Faiss search" << std::endl;
    size_t block = (base_num <= BLOCK_SIZE) ? 1000000 : 2048;
    std::string res;
    for (size_t i = 0; i < sq_num; i += block) {
        std::cout << "remaining: " << i << "/" << sq_num << std::endl;
        size_t read = std::min<size_t>(block, sq_num - i);
        // if (base_num <= BLOCK_SIZE)
        myfaiss->Search(data_sq + i * sq_dim, knn_ids.data() + i * k, knn_dists.data() + i * k,
                        read);
        // else
        //     knn_streamer->process_batch_async(data_sq + i * sq_dim, (int)read,
        //                                       knn_ids.data() + i * k);

        // myfaiss->Search(data_bp, base_num, data_sq + i * sq_dim, knn_ids.data() + i * k,
        //                 knn_dists.data() + i * k, read);
        for (int j = 0; j < k; j++) res += std::to_string(knn_ids[j]) + " ";
    }

    std::cout << "Faiss search end: " << res << std::endl;
    float time = std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - s)
                     .count() /
                 1000.0,
          avg_time = time / sq_num;

    std::transform(knn_ids.begin(), knn_ids.end(), knn_ids_int.begin(),
                   [](faiss::idx_t x) { return static_cast<int>(x); });

    gt_out.write(reinterpret_cast<char*>(knn_ids_int.data()), sq_num * k * sizeof(int));
    gt_out.close();
    std::cout << sq_num << " gt comp finished, time: " << time << ", avg_time: " << avg_time
              << std::endl;
    return 0;
}

/*
dataset=./data/t2i-10M
./build/tests/comp_gt \
        --base_data_path ${dataset}/base.fbin  \
        --query_data_path ${dataset}/query.train.fbin  \
        --gt_save_file ${dataset}/train.gt.bin --k 100 --nq 1000000
*/