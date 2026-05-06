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
using namespace efanna2e;

typedef faiss::gpu::StandardGpuResources GPUResources;
typedef faiss::gpu::GpuIndexFlatL2 GPUFlatL2;

int main(int argc, char** argv) {
    std::string base_data_file;
    std::string query_file;
    std::string update_gt_file;
    std::string data_type;
    std::string dist, dataset;
    std::string query_cache_file;
    int k, upd_mode;

    po::options_description desc{"Arguments"};
    try {
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("dist", po::value<std::string>(&dist)->required(),
                           "distance function <l2/ip>");
        desc.add_options()("dataset", po::value<std::string>(&dataset)->required(),
                           "dataset <t2i>");
        desc.add_options()("base_data_path", po::value<std::string>(&base_data_file)->required(),
                           "Input data file in bin format");
        desc.add_options()("query_path", po::value<std::string>(&query_file)->required(),
                           "Query file in bin format");
        desc.add_options()("update_gt_file", po::value<std::string>(&update_gt_file)->required(),
                           "update gt file in bin format");
        desc.add_options()("k", po::value<int>(&k)->required(), "k nearest neighbors");
        desc.add_options()("upd_mode", po::value<int>(&upd_mode)->required(), "update mode");

        desc.add_options()("data_type",
                           po::value<std::string>(&data_type)->default_value("float")->required(),
                           "data type <int8/uint8/float>");

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
    size_t base_num, sq_num;
    int base_dim, sq_dim;
    efanna2e::load_meta<float>(base_data_file.c_str(), base_num, base_dim);
    efanna2e::Metric dist_metric = efanna2e::INNER_PRODUCT;
    DIST_METRIC m = DIST_METRIC::L2_;
    if (dist == "l2") {
        dist_metric = efanna2e::L2;
        std::cout << "Using l2 as distance metric" << std::endl;
    } else if (dist == "ip") {
        m = DIST_METRIC::IP_;
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
    efanna2e::Parameters parameters;
    if (base_num <= BLOCK_SIZE) {
        efanna2e::load_data<float>(base_data_file.c_str(), base_num, base_dim, data_bp);
        data_bp = efanna2e::data_align(data_bp, base_num, base_dim);
    }
    efanna2e::load_meta<float>(query_file.c_str(), sq_num, sq_dim);
    float* query_data = nullptr;
    std::cout << "load query\n";
    efanna2e::load_data<float>(query_file.c_str(), sq_num, sq_dim, query_data);
    query_data = efanna2e::data_align(query_data, sq_num, sq_dim);

    // if (dist_metric == efanna2e::INNER_PRODUCT) {
    //     efanna2e::ip_normalize(aligned_data_bp, base_dim);
    //     dist_metric = efanna2e::L2;
    // }
    const int total_num = base_num;
    base_num = base_num / 2;
    // data_ins = aligned_data_bp + base_num * base_dim;
    // GPUResources gpu_res;
    // GPUFlatL2 gpu_index(&gpu_res, base_dim);
    MyFaiss myfaiss(base_dim, k, m);
    std::vector<float> knn_dists(sq_num * k);
    std::vector<faiss::idx_t> knn_ids(sq_num * k);
    std::vector<int> knn_ids_int(sq_num * k);
    PC.upd_mode = upd_mode;
    if (upd_mode == 1)
        update_gt_file += ".del";
    else if (upd_mode == 2)
        update_gt_file += ".ins";
    std::ofstream gt_file(update_gt_file, std::ios::binary);
    const int round = 100, batch = (base_num + round - 1) / round;
    if (!gt_file.is_open()) {
        std::cout << "Failed to open update_gt_file: " << update_gt_file << std::endl;
        exit(-1);
    }
    gt_file.write(reinterpret_cast<const char*>(&round), sizeof(int));

    myfaiss.Add(data_bp, base_num);
    myfaiss.Search(query_data, knn_ids, knn_dists, sq_num);

    std::transform(knn_ids.begin(), knn_ids.end(), knn_ids_int.begin(),
                   [](faiss::idx_t x) { return static_cast<int>(x); });
    gt_file.write(reinterpret_cast<const char*>(knn_ids_int.data()), sizeof(int) * sq_num * k);

    if (upd_mode == 0) {  // delete and insert:
        for (int batch_i = 0; batch_i < round; batch_i++) {
            std::cout << "Update " << batch_i << " round" << std::endl;
            const size_t upd_n = std::min((size_t)batch, base_num - batch_i * batch);
            const size_t start_id = batch_i * batch + upd_n;
            // gpu_index.reset();
            assert(start_id + upd_n <= total_num);
            myfaiss.Add(data_bp + start_id * base_dim, base_num);
            myfaiss.Search(query_data, knn_ids, knn_dists, sq_num);
            std::transform(knn_ids.begin(), knn_ids.end(), knn_ids_int.begin(),
                           [start_id](faiss::idx_t x) { return static_cast<int>(x) + start_id; });
            gt_file.write(reinterpret_cast<const char*>(knn_ids_int.data()),
                          sizeof(int) * sq_num * k);
        }
    } else if (upd_mode == 4) {  // delete and insert:
        for (int batch_i = 0; batch_i < round; batch_i++) {
            std::cout << "Update " << batch_i << " round" << std::endl;
            const size_t upd_n = std::min((size_t)batch, base_num - batch_i * batch);
            const size_t start_id = batch_i * batch + upd_n,
                         ins_start_id = batch_i * batch + base_num;
            // gpu_index.reset();
            assert(start_id + upd_n <= total_num);
            myfaiss.Add(data_bp + start_id * base_dim, base_num - batch);
            myfaiss.Search(query_data, knn_ids, knn_dists, sq_num);
            std::transform(knn_ids.begin(), knn_ids.end(), knn_ids_int.begin(),
                           [start_id](faiss::idx_t x) { return static_cast<int>(x) + start_id; });
            gt_file.write(reinterpret_cast<const char*>(knn_ids_int.data()),
                          sizeof(int) * sq_num * k);
            myfaiss.Add(data_bp + ins_start_id * base_dim, upd_n, false);
            myfaiss.Search(query_data, knn_ids, knn_dists, sq_num);
            std::transform(knn_ids.begin(), knn_ids.end(), knn_ids_int.begin(),
                           [start_id](faiss::idx_t x) { return static_cast<int>(x) + start_id; });
            gt_file.write(reinterpret_cast<const char*>(knn_ids_int.data()),
                          sizeof(int) * sq_num * k);
        }
    } else if (upd_mode == 1) {  // only delete:
        for (int batch_i = 0; batch_i < round / 2; batch_i++) {
            std::cout << "Update " << batch_i << " round" << std::endl;
            const size_t upd_n = std::min((size_t)batch, base_num - batch_i * batch);
            const size_t start_id = batch_i * batch + upd_n;
            // gpu_index.reset();
            assert(start_id + upd_n <= total_num);
            // memcpy(data, data_bp + start_id * base_dim,
            //        (size_t)base_num * base_dim * sizeof(float));
            // gpu_index.add(base_num, data);
            // gpu_index.search(sq_num, aligned_query_data, k, knn_dists.data(), knn_ids.data());
            myfaiss.Add(data_bp + start_id * base_dim, base_num - start_id);
            myfaiss.Search(query_data, knn_ids, knn_dists, sq_num);
            std::transform(knn_ids.begin(), knn_ids.end(), knn_ids_int.begin(),
                           [start_id](faiss::idx_t x) { return static_cast<int>(x) + start_id; });
            gt_file.write(reinterpret_cast<const char*>(knn_ids_int.data()),
                          sizeof(int) * sq_num * k);
        }
    } else {  // only insert:
        for (int batch_i = 0; batch_i < round; batch_i++) {
            std::cout << "Update " << batch_i << " round" << std::endl;
            const size_t upd_n = std::min((size_t)batch, base_num - batch_i * batch);
            const size_t start_id = batch_i * batch;
            // gpu_index.reset();
            // memcpy(data, data_bp + start_id * base_dim,
            //        (size_t)base_num * base_dim * sizeof(float));
            // gpu_index.add(base_num, data);
            // gpu_index.search(sq_num, aligned_query_data, k, knn_dists.data(), knn_ids.data());
            myfaiss.Add(data_bp + (base_num + start_id) * base_dim, upd_n, false);
            myfaiss.Search(query_data, knn_ids, knn_dists, sq_num);
            std::transform(knn_ids.begin(), knn_ids.end(), knn_ids_int.begin(),
                           [](faiss::idx_t x) { return static_cast<int>(x); });
            gt_file.write(reinterpret_cast<const char*>(knn_ids_int.data()),
                          sizeof(int) * sq_num * k);
        }
    }
    efanna2e::fo.print("Save update gt to " + update_gt_file);

    return 0;
}