#ifndef HYPERPARA_H
#define HYPERPARA_H

#include <faiss/Index.h>
#include <faiss/IndexBinary.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/utils/distances.h>
#include <float.h>
#include <malloc.h>

#include <cassert>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <filesystem>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "fileout.h"
#include "json.hpp"

typedef faiss::gpu::StandardGpuResources GPUResources;
typedef faiss::gpu::GpuIndexIVFFlat GPUIVFFlat;
typedef faiss::IndexFlatL2 CPUFlatL2;
typedef faiss::gpu::GpuIndexFlatL2 GPUFlatL2;
typedef faiss::IndexFlatIP CPUFlatIP;
typedef faiss::gpu::GpuIndexFlatIP GPUFlatIP;

typedef std::chrono::_V2::system_clock::time_point TIME;

#define CONFIG_FILE "../config.json"

// #define VERBOSE 1

#define MEM_MODE 1  // 0 clock, 1 disk

// #define PC.upd_mode 0  // del and ins
// #define PC.upd_mode 1  // only del
// #define PC.upd_mode 2  // only ins
// #define PC.upd_mode 4

#define DATA_ALIGN_FACTOR 4

#define MAX_DEGREE 32
#define INNBR_BATCH (4096 * MAX_DEGREE)

#define TEST_SEARCH_QUERY_SIZE (PC.batch)

#define IP_CLOSEST FLT_MAX
#define IP_FARTHEST (-FLT_MAX)
#define L2_CLOSEST 0.0
#define L2_FARTHEST FLT_MAX

#define GPU_N 10010000
#define CPU_N 100000000
#define DISK_N 1000000000

// #define algo0 "add_reverse"
// #define algo1 "add_all_nbrs"
#define algo2 "add_nbrs_to_non_full_nodes"
#define GIG "graph in gpu"
// #define GPU_TEST "gpu test"
#define INFO_PRINT true

#define LOG_PREFIX "../logs/"
#define UPD_LOG_PREFIX "../logs/upd/"
#define RECALL_PREFIX "../logs/recall/"
#define INDEX_PREFIX "../indexes/"
#define EVALUATION_PREFIX "../evaluation/"
#define UPD_EVALUATION_PREFIX "../evaluation/upd/"
#define GT_PREFIX "../gts/"

#define TOS(x) std::to_string(x)
#define GB(x) (x * (1ULL << 30))
using TIME = std::chrono::_V2::system_clock::time_point;

enum DIST_METRIC { L2_, IP_, COS_ };

namespace efanna2e {

struct ProConfig {
    int dim = 200;
    int k = 100;
    int max_degree = 32;
    int beam_capacity = 1024;
    int enhance_nap_batch = 4096;
    int batch = 1024;
    int enhance_search_batch = 524288;
    int enhance_beam_capacity = 100;
    int enhance_query_num = 1;
    int insert_batch = 2048;
    int insert_query_k = 1;
    int test_search = 1;
    int test_beam_capacity = 512;
    int use_knn_cache = 0;
    int search_start_nodes_num = 16;
    uint32_t knn_max_hit = 10;
    // int tempk_ratio = 2;
    int need_discard = 1;
    int enhance_mode = 0;  // 0 search, 1 knn
    int knn_rebuild = 0;
    float knn_rebuild_thres = 0.1;
    int ivf_rebuild = 0;
    int ivf_nlist = 4096;
    int ivf_nprobe = 16;
    int test_search_frequency = 100;  // batch
    int ivf_points_per_centroid = 256;
    int top_hit_num = 0;
    int calc_hit = 0;
    int upd_mode = 0;  // 0 del and ins, 1 only del, 2 only ins
    int upd_build = 0;
    int upd_type = 1;  // 0 roar, 1 search, 2 query, 3 search + query
    int search_in_cpu = 0;
    int knn_rebuild_min_cnt = 2000000;
};
extern ProConfig PC;

using ull = unsigned long long;
using JSON = nlohmann::json;

typedef std::filesystem::path fpath;

enum STATUS {
    Invalid,
    Initial,    // 初始状态
    Discarded,  // 淘汰状态
    Retained,   // 保留状态
    Repeated    // 节点和前面的节点重复
};

enum Mode { CONS, SEARCH, UPD, CE };

enum GraphType { GPU, CPU, DISK };

// struct Graph_Struct_D {
//     int iso_num;       // 孤立点数量
//     int total_degree;  // 总度数
// };
// using GSD = Graph_Struct_D;

double welford_variance(const uint32_t* v, int N);

void writeCSV(const std::vector<std::vector<std::string>>& data, const std::string& filename,
              const std::ios_base::openmode mode = std::ios::app);

std::string getCurrentDateTimeString();

template <typename T>
inline bool vec_exists(std::vector<T>& vec, const T& val) {
    return std::find(vec.begin(), vec.end(), val) != vec.end();
}

inline float farthest_dist(DIST_METRIC m) {
    return m == DIST_METRIC::L2_ ? L2_FARTHEST : IP_FARTHEST;
}

inline float closest_dist(DIST_METRIC m) {
    return m == DIST_METRIC::L2_ ? L2_CLOSEST : IP_CLOSEST;
}

int numDigits(int n);  // 计算数字 n 的位数

inline bool compare_dist(float d0, float d1, DIST_METRIC m) {
    return m == DIST_METRIC::L2_ ? d0 > d1 : d0 < d1;  // false - d0 更近，true - d1 更近
}

int testAndSetBit(int* bitset, int bit_index);

inline int max_pow2_le(int N) {
    if (N <= 0) return 0;
    N |= (N >> 1);
    N |= (N >> 2);
    N |= (N >> 4);
    N |= (N >> 8);
    N |= (N >> 16);
    return N - (N >> 1);
}

// 返回 <= N 的最大 2 的幂 (N > 0)
inline uint32_t floor_pow2(uint32_t N) {
    N |= N >> 1;
    N |= N >> 2;
    N |= N >> 4;
    N |= N >> 8;
    N |= N >> 16;
    return N - (N >> 1);
}

void print_times();
void record_time(const std::string name, float time_s);
std::string avg_time_serialize();
void avg_time_deserialize(const std::string& str);

template <typename T>
void load_part_of_data(const std::string& filename, uint64_t start, uint32_t points_num, int dim,
                       T*& data) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) fo.eprint("open file error: " + filename);

    in.seekg(sizeof(uint32_t) * 2 + sizeof(T) * dim * start, std::ios::beg);
    uint64_t pts = static_cast<uint64_t>(points_num);
    uint64_t d = static_cast<uint64_t>(dim);
    uint64_t new_dim = (d + DATA_ALIGN_FACTOR - 1) / DATA_ALIGN_FACTOR * DATA_ALIGN_FACTOR;
    if (data == nullptr) {
        std::cout << "allocating data memory for " << pts * new_dim * sizeof(T) << " bytes"
                  << std::endl;
        data = (T*)memalign(DATA_ALIGN_FACTOR * 8, pts * new_dim * sizeof(T));
    }
    if (new_dim == d)
        in.read((char*)data, pts * d * sizeof(T));
    else
        for (size_t i = 0; i < points_num; i++) {
            in.read((char*)(data + i * new_dim), d * sizeof(T));
            memset(data + i * new_dim + d, 0, (new_dim - d) * sizeof(T));
        }
    // cursor position
    std::ios::pos_type ss = in.tellg();
    size_t pos = (pts + start) * d * sizeof(T) + sizeof(uint32_t) * 2;
    if ((size_t)ss != pos)
        fo.eprint("Read file incompleted! filename:" + filename +
                  "\nread pos: " + TOS((size_t)ss) + ", act pos: " + TOS(pos));

    in.close();
}

class TaskExecutor {
   public:
    TaskExecutor() { worker_ = std::thread(&TaskExecutor::worker_loop, this); }
    ~TaskExecutor() { stop(); }
    TaskExecutor(const TaskExecutor&) = delete;
    TaskExecutor& operator=(const TaskExecutor&) = delete;

    // 提交无返回值任务
    void submit(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (stopped_) throw std::runtime_error("submit on stopped TaskExecutor");
            tasks_.emplace_back([task = std::move(task)]() mutable {
                try {
                    task();
                } catch (const std::exception& e) {
                    fo.eprint("[TaskExecutor] task exception: " + std::string(e.what()));
                } catch (...) {
                    fo.eprint("[TaskExecutor] task exception: unknown");
                }
            });
        }
        cv_.notify_one();
    }

    // 提交带返回值任务，调用方可 future.get()
    template <class F>
    auto submit_with_future(F&& f) -> std::future<std::invoke_result_t<F>> {
        using Ret = std::invoke_result_t<F>;
        auto task_ptr = std::make_shared<std::packaged_task<Ret()>>(std::forward<F>(f));
        std::future<Ret> fut = task_ptr->get_future();
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (stopped_) throw std::runtime_error("submit on stopped TaskExecutor");
            tasks_.emplace_back([task_ptr]() mutable { (*task_ptr)(); });
        }
        cv_.notify_one();
        return fut;
    }

    void finish() {
        auto fut = submit_with_future([]() {});
        fut.get();
    }

    void stop() {
        bool need_join = false;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (!stopped_) {
                stopped_ = true;
                need_join = true;
            }
        }
        cv_.notify_all();
        if (need_join && worker_.joinable()) worker_.join();
    }

   private:
    void worker_loop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_.wait(lk, [this]() { return stopped_ || !tasks_.empty(); });
                if (stopped_ && tasks_.empty()) break;
                task = std::move(tasks_.front());
                tasks_.pop_front();
            }
            task();  // 串行执行任务
        }
    }

   private:
    std::mutex mtx_;
    std::condition_variable cv_;
    std::deque<std::function<void()>> tasks_;
    bool stopped_ = false;
    std::thread worker_;
};

class Base_Parameters {
   public:
    size_t query_n, base_n, visit_n, hash_n;
    int dim, max_degree, k, beam_capacity, beam_expand_num, beam_size,
        visited_words_num = PC.batch / 32, gpu_n;
    // uint32_t bloom_u32;
    DIST_METRIC metric;

    Base_Parameters() = default;
    Base_Parameters(int base_n, int dim, DIST_METRIC metric, int k)
        : base_n(base_n), metric(metric), dim(dim), k(k) {};
    Base_Parameters(int dim, int max_degree, int base_n, int query_n, int k, int beam_capacity,
                    DIST_METRIC metric)
        : dim(dim),
          max_degree(max_degree),
          base_n(base_n),
          visit_n((base_n + 31) / 32),
          query_n(query_n),
          k(k),
          beam_capacity(beam_capacity),
          metric(metric),
          gpu_n(std::min(GPU_N, base_n)) {}
    inline std::string str() {
        return std::string("Base_Parameters: dim=") + TOS(dim) +
               ", max_degree=" + TOS(max_degree) + ", base_n=" + TOS(base_n) +
               ", visit_n=" + TOS(visit_n) + ", query_n=" + TOS(query_n) + ", k=" + TOS(k) +
               ", beam_capacity=" + TOS(beam_capacity) +
               ", beam_expand_num=" + TOS(beam_expand_num) + ",beam_size=" + TOS(beam_size) +
               ", metric=" + TOS(metric) + ", gpu_n=" + TOS(gpu_n);
    }
};
using BP = Base_Parameters;

// class Candidate_Neighbor {
//    public:
//     int id;          // 候选邻居向量在基础向量集中的索引
//     int idx;         // 候选邻居向量在其所在 KNN 簇中的索引
//     STATUS status;   // 候选邻居向量淘汰情况（0 待定，1 保留，2 淘汰，-1 全部保留）
//     float dist;      // 候选邻居向量与 pivot 向量的距离
//     float vec[DIM];  // 候选邻居向量
// };
class Candidate_Neighbor {
   public:
    int id;         // 候选邻居向量在基础向量集中的索引
    int idx;        // 候选邻居向量在 GPU 缓存中的索引
    STATUS status;  // 候选邻居向量淘汰情况（0 待定，1 保留，2 淘汰，-1 全部保留）
    float dist;     // 候选邻居向量与 pivot 向量的距离
};
using CN = Candidate_Neighbor;

class BatchResult {
   public:
    std::vector<int> knn, qids;
    int batch;
    double time_s;

    BatchResult() = default;
    // BatchResult(int* d_knn, const int k, const int batch, const double time_s);
    BatchResult(const int* d_knns, const int* qids_, const int k, const int batch,
                const double time_s)
        : batch(batch), time_s(time_s) {
        knn.resize(k * batch);
        cudaMemcpy(knn.data(), d_knns, batch * k * sizeof(int), cudaMemcpyDeviceToHost);
        if (qids_) {
            qids.resize(batch);
            memcpy(qids.data(), qids_, batch * sizeof(int));
        }
        // std::string res;
        // bool right = true;
        // for (int i = 0; i < batch * k; i++) {
        //     res += TOS(knn[i]) + " ";
        //     if (knn[i] < 0) right = false;
        // }
        // if (!right) fo.eprint("knn error: " + res);
    }

    BatchResult(const int* knns, const int k, const int batch) : batch(batch), time_s(0.0f) {
        const size_t total = k * batch;
        knn.resize(total);
        std::memcpy(knn.data(), knns, total * sizeof(int));
    }

    BatchResult(const int k, const int batch) : batch(batch) {
        knn.resize(k * batch);
        qids.resize(batch);
    }
};

class Test_Result {
   public:
    Test_Result() : recall(0.0), hops(0.0), time(0.0), hits(0) {};
    Test_Result(Test_Result& others) {
        hits = others.hits;
        hops = others.hops;
        time = others.time;
        recall = others.recall;
    }
    Test_Result(const int hits, const size_t hops, const float time)
        : hits(hits), hops(hops), time(time) {}
    Test_Result(const int hits, const size_t hops) : hops(hops), hits(hits) {}
    inline float get_recall() const { return recall; }
    inline size_t get_hops() const { return hops; }
    inline float get_time() const { return time; }
    float calc_recall(int total_num) { return recall = 100.0 * hits / total_num; }
    float avg_hops(int search_num) { return 1.0 * hops / search_num; }
    void accumulate(const Test_Result res) {
        hops += res.hops;
        time += res.time;
        hits += res.hits;
    }

    float recall, time;
    int hits;
    size_t hops;
};

class Graph_Update_Info {
   public:
    Graph_Update_Info() : is_update(false) {}
    inline void update(int batch_id_, int total_num_) {
        batch_id = batch_id_;
        total_num = total_num_;
        is_update = true;
    }
    inline void reset() { is_update = false; }
    inline bool vaild() { return is_update; }
    inline int get_batch_id() { return batch_id; }
    inline int get_num() { return total_num; }
    // inline GSD get_GSD() { return gs; }

   private:
    bool is_update;
    int batch_id, total_num;
    // GSD gs;
};
using GUI = Graph_Update_Info;

class Avg_Time {
   private:
    int n;
    float time;

   public:
    Avg_Time() : n(0), time(0) {}
    Avg_Time(const std::string& time_str, const std::string& n_str) {
        time = std::stof(time_str);
        n = std::stoi(n_str);
    }
    void add(float t) {
        n++;
        time += t;
    }
    inline float get() { return time / n; };
    inline float get_total() { return time; };
    inline int get_n() { return n; };
    // 重载 < 运算符，用于排序
    bool operator<(const Avg_Time& other) const {
        return time < other.time;  // 按年龄升序
    }
    std::string serialize() { return std::to_string(time) + "," + std::to_string(n); }
    void deserialize(const std::string& str) {
        std::istringstream iss(str);
        std::string temp;
        std::getline(iss, temp, ',');
        time = std::stof(temp);
        std::getline(iss, temp, ',');
        n = std::stoi(temp);
    }
};

inline void create_dir(const fpath& graph_dir_path) {
    // fo.print("Creating graph directory: " + graph_dir_path.string());
    if (!std::filesystem::exists(graph_dir_path))
        std::filesystem::create_directories(graph_dir_path);  // 递归创建多级目录
}

void normalize_L2_omp(float* x, size_t n, size_t d);

extern std::unordered_map<std::string, Avg_Time> avg_times;
};  // namespace efanna2e

inline TIME now_time() { return std::chrono::high_resolution_clock::now(); }

inline float time_diff(TIME start, TIME end) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
}
inline float time_diff(TIME start) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(now_time() - start).count() /
           1000.0;
}
#endif  // HYPERPARA_H