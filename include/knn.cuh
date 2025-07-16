#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <array>

namespace efanna2e
{

#define BATCH 128
#define MAX_DIM 200

#define CUDA_CHECK(call)                                                         \
    do                                                                           \
    {                                                                            \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess)                                                  \
        {                                                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl;                   \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

#define MAXK 128

    class Stream
    {
    public:
        cudaStream_t stream;
        cudaEvent_t start, stop;

        Stream();
        ~Stream();
    };

    struct Candidate_Neighbor
    {
        uint32_t id;        // 候选邻居向量在基础向量集中的索引
        uint32_t idx;       // 候选邻居向量在其所在 KNN 簇中的索引
        uint8_t status = 0; // 候选邻居向量淘汰情况（0 待定，1 保留，2 淘汰）
        float dist;         // 候选邻居向量与 pivot 向量的距离
    };

    class GPUFuncs
    {
    public:
        GPUFuncs(
            const float *h_base, int base_n, int dim, int k, int max_degree);
        ~GPUFuncs();
        uint32_t *knn_compute(const float *h_queries, const int batch, float &time_ms);
        uint32_t *handle_knn_updates(const uint32_t *d_knn_idxs, const int batch);
        // void get_graph_data(std::vector<int> &h_in_deg, std::vector<int> &h_adj);

    private:
        float *d_q, *d_b;
        int base_n, dim, k, max_degree;
        uint32_t *d_zero_to_k_idxs;

        uint32_t *d_knn_res;     // 存储 knn_compute 每次的结果（一个 batch 的 KNN idxs）
        uint32_t *d_new_nbr_ids; // handle_knn_updates 的结果

        const std::string stream_names[2] = {"knn", "upd"};
        std::unordered_map<std::string, Stream> streams;

        int blocknum_per_query = 5;

        size_t shared_mem_per_block;

        void event_record_time_start(const std::string name);
        float event_record_time_stop(const std::string name, std::string msg);
    };

} // namespace efanna2e