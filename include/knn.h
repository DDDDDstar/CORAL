#include <cuda_runtime.h>
namespace efanna2e {

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err = (call);                                                        \
        if (err != cudaSuccess) {                                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "        \
                      << cudaGetErrorString(err) << std::endl;                           \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    } while (0)

#define MAXK 128

// 图数据结构
struct Graph {
    int *in_deg;       // 当前边数量
    int *adj;          // 邻居索引，大小 base_n * K
    float *adj_dist;   // 对应邻居距离
};

#ifdef __CUDACC__
    extern __device__ Graph g;
#else
    extern Graph g; // 主机端声明
#endif

extern "C"{
    /**
     * @brief GPU异步计算k近邻
     * @param d_q 查询点数组（设备指针）
     * @param d_b 基础数据集数组（设备指针）
     * @param base_n 基础数据集大小
     * @param dim 数据维度
     * @param batch 批处理大小
     * @param K 近邻数
     * @param d_idx 输出：近邻索引（设备指针）
     * @param d_dist 输出：近邻距离（设备指针）
     * @param stream CUDA流
     */
    void gpu_knn_compute_async(
        float* d_q, float* d_b, int base_n, int dim, 
        int batch, int K, int* d_idx, float* d_dist, cudaStream_t stream
    );

    /**
     * @brief 异步处理k-NN结果并更新图结构
     * @param base_n 基础数据集大小
     * @param d_idx 近邻索引（设备指针）
     * @param d_dist 近邻距离（设备指针）
     * @param batch 批处理大小
     * @param K 近邻数
     * @param dim 数据维度
     * @param stream CUDA流
     */
    void gpu_handle_knn_updates_async(
        int base_n, int* d_idx, float* d_dist, int batch, int K, int dim, cudaStream_t stream
    );

    /**
     * @brief 计算图中孤立点的比例
     * @param stream CUDA流
     * @return 孤立点比例（0.0~1.0）
     */
    float gpu_get_isolated_ratio(int base_n);

    void gpu_graph_create(int base_n, int K);
    void gpu_graph_destroy();


}
} // namespace efanna2e