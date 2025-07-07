#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include "knn.h"
#include <cfloat>
#include <utility>
#include <iostream>

namespace efanna2e {

constexpr int TILE_B = 128;      // 每次加载 base 数据块大小

// 设备端全局图结构
__device__ Graph g;

__global__ void knn_tiled_kernel(
    const float* d_q, const float* d_b,
    int base_n, int dim, int* d_idx, float* d_dist, int batch, int K
) {
    extern __shared__ float tile_b[];
    int qi = blockIdx.x;
    if (qi >= batch) return;

    float* buf = &tile_b[0];
    const float* qvec = &d_q[qi * dim];
    // 每个线程在寄存器中维护自己的 top-K 阵列
    float best_dist[100];
    int best_idx[100];
    // 初始化 top-K 为 +inf
    for (int t = 0; t < K; ++t) {
        best_dist[t] = FLT_MAX;
        best_idx[t]  = -1;
    }
    // 分块遍历所有 base 向量
    for (int b_off = 0; b_off < base_n; b_off += TILE_B) {
        int chunk = min(TILE_B, base_n - b_off);
        int tid = threadIdx.x;
        int total_elems = chunk * dim;
        int per = (total_elems + blockDim.x - 1)/blockDim.x;
        // 所有线程协作将这一块 base 数据加载到 shared memory
        for (int e = 0; e < per; ++e) {
            int idx_glob = tid + e * blockDim.x;
            if (idx_glob < total_elems) {
                tile_b[idx_glob] = d_b[b_off*dim + idx_glob];
            }
        }
        __syncthreads();  // 确保 tile_b 已加载完毕 
        // 对这一块数据计算距离并更新寄存器中的 top-K
        for (int bi = 0; bi < chunk; ++bi) {
            // 计算欧氏距离
            float sum = 0.0f;
            for (int d = 0; d < dim; ++d) {
                float diff = qvec[d] - tile_b[bi * dim + d];
                sum += diff * diff;
            }
            // 插入到 best_dist / best_idx 的本地 top-K 中
            // 只需比较最远的第 K-1 个位置
            if (sum < best_dist[K - 1]) {
                // 放到末尾，然后上移
                best_dist[K - 1] = sum;
                best_idx[K - 1]  = b_off + bi;
                // 简单插入排序，将新值上移到合适位置
                for (int x = K - 2; x >= 0 && best_dist[x] > best_dist[x + 1]; --x) {
                    float tmpd = best_dist[x];
                    best_dist[x] = best_dist[x+1];
                    best_dist[x+1] = tmpd;
                    int tmpi = best_idx[x];
                    best_idx[x] = best_idx[x+1];
                    best_idx[x+1] = tmpi;
                }
            }
        }
        __syncthreads();
    }
    // 将寄存器中的结果写回全局内存
    for (int t = 0; t < K; ++t) {
        int out = qi * K + t;
        d_dist[out] = best_dist[t];
        d_idx[out]  = best_idx[t];
    }
}

__global__ void init_graph(int base_n, int K){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < base_n) {
        g.in_deg[i] = 0;
        for (int j = 0; j < K; ++j) {
            g.adj[i * K + j] = -1;
            g.adj_dist[i * K + j] = FLT_MAX;
        }
    }
}

__global__ void graph_update(const int* idx, const float* dist, int batch, int K){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch) return;

    int lane = threadIdx.x % warpSize;
    int base = tid * K;
    for (int j = 0; j < K; ++j) {
        int v = idx[base + j];
        float dv = dist[base + j];
        // warp 聚合增量
        unsigned mask_v = __match_any_sync(__activemask(), v);
        __syncwarp(mask_v);
        int count = __popc(mask_v);
        int leader = __ffs(mask_v) - 1;
        int old_val = 0;
        if (lane == leader) {
            old_val = atomicAdd(&g.in_deg[v], count);
        }
        __syncwarp(mask_v);
        old_val = __shfl_sync(mask_v, old_val, leader);
        int rank = __popc(mask_v & ((1u << lane) - 1));
        int pos = old_val + rank;

        if (pos < K) {
            g.adj[v * K + pos] = idx[base + j];       // self 邻居也可以插入
            g.adj_dist[v * K + pos] = dv;
        } else if (lane == leader) {
            int worst = 0;
            float maxd = g.adj_dist[v * K];
            for (int z = 1; z < K; ++z) {
                float d = g.adj_dist[v * K + z];
                if (d > maxd) { maxd = d; worst = z; }
            }
            if (dv < maxd) {
                g.adj[v * K + worst] = idx[base + j];
                g.adj_dist[v * K + worst] = dv;
            }
        }
    }
}

__global__ void count_isolated(int base_n, int* out_iso) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < base_n && g.in_deg[i] == 0) atomicAdd(out_iso, 1);
}


bool cuda_check_last_error(const char* func_name) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel error in " << func_name << ": " << cudaGetErrorString(err) << std::endl;
        return true;
    }
    return false;
}

extern "C" void gpu_graph_create(int base_n, int K) {
    Graph h_g;
    CUDA_CHECK(cudaMalloc(&h_g.in_deg,   base_n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&h_g.adj,      base_n * K * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&h_g.adj_dist, base_n * K * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(g, &h_g, sizeof(Graph)));

    int threads = 256, blocks = (base_n + threads - 1) / threads;
    init_graph<<<blocks, threads>>>(base_n, K);
    CUDA_CHECK(cudaDeviceSynchronize());  // 等待内核完成
    if (cuda_check_last_error("gpu_graph_create")) exit(EXIT_FAILURE);
    std::cout << "gpu_graph_create success, (base_n, K): " << base_n << ", " << K << std::endl; 
}

// 释放资源的接口
extern "C" void gpu_graph_destroy() {
    Graph h_g;
    CUDA_CHECK(cudaMemcpyFromSymbol(&h_g, g, sizeof(Graph)));  // 获取设备指针
    CUDA_CHECK(cudaFree(h_g.in_deg));
    CUDA_CHECK(cudaFree(h_g.adj));
    CUDA_CHECK(cudaFree(h_g.adj_dist));
}

extern "C" void gpu_handle_knn_updates_async(
    int base_n, int* d_idx, float* d_dist,
    int batch, int K, int dim, cudaStream_t stream
) {
    int threads = 256, blocks = (batch + threads - 1) / threads;
    graph_update<<<blocks, threads, 0, stream>>>(d_idx, d_dist, batch, K);
    if (cuda_check_last_error("gpu_handle_knn_updates_async")) exit(EXIT_FAILURE);
}

extern "C" float gpu_get_isolated_ratio(int base_n) {
    int *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(int)));
    int threads=256, blocks=(base_n + threads - 1) / threads;
    count_isolated<<<blocks, threads>>>(base_n, d_out);
    // 同步等待内核完成
    CUDA_CHECK(cudaDeviceSynchronize());
    if (cuda_check_last_error("gpu_get_isolated_ratio")) {
        CUDA_CHECK(cudaFree(d_out));
        exit(EXIT_FAILURE);
    }
    int iso;
    CUDA_CHECK(cudaMemcpy(&iso, d_out, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));
    std::cout << "gpu_get_isolated_ratio success, (base_n, d_out): " << base_n << ", " << iso << std::endl; 
    return static_cast<float>(iso) / base_n;
}

extern "C" void gpu_knn_compute_async(
    float* d_q, float* d_b,
    int base_n, int dim, int batch, int K,
    int* d_idx, float* d_dist,
    cudaStream_t stream
) {
    int blocks = batch;
    int threads = 256;
    size_t shared_bytes = TILE_B * dim * sizeof(float);

    CUDA_CHECK(cudaFuncSetAttribute(knn_tiled_kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));

    knn_tiled_kernel<<<blocks, threads, shared_bytes, stream>>>(
        d_q, d_b, base_n, dim, d_idx, d_dist, batch, K);
    if (cuda_check_last_error("gpu_knn_compute_async")) {
        exit(EXIT_FAILURE);
    }
}
}