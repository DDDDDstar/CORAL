#include <chrono>
#include <climits>
#include <random>

#include "fileout.h"
#include "graph.cuh"
#include "utils.cuh"

namespace efanna2e {
__global__ void bfs_init(int* hops, int* frontier, int start, int N) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        if (i != start)
            hops[i] = -1;
        else
            hops[i] = 0;
    }
    if (!blockIdx.x && !threadIdx.x) frontier[0] = start;
}

__global__ void update_min_hop_kernel(const int* __restrict__ hops, int* __restrict__ min_hops,
                                      int N) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        const int h = hops[i];
        if (h >= 0) {
            if (h < min_hops[i] || min_hops[i] < 0) min_hops[i] = h;
        } else if (min_hops[i] == INT_MAX)
            min_hops[i] = -1;
    }
}

__global__ void bfs_expand_kernel(const Slot* __restrict__ slots, const int* __restrict__ frontier,
                                  int frontier_size, int* __restrict__ next_frontier,
                                  int* __restrict__ next_size, int* __restrict__ hops, int level) {
    const int tpb = blockDim.x, tid = blockIdx.x * tpb + threadIdx.x, warp = tid / warpSize,
              lane = tid % warpSize, totalwarp = gridDim.x * tpb / warpSize;
    for (int i = warp; i < frontier_size; i += totalwarp) {
        const int u = frontier[i], deg = slots[u].deg, *nbrs = slots[u].nbrs;
        for (int i = lane; i < deg; i += warpSize) {
            const int v = nbrs[i];
            bool valid = (atomicCAS(hops + v, -1, level) == -1);
            const unsigned active = __activemask(), mask = __ballot_sync(active, valid);
            const int cnt = __popc(mask);
            if (cnt) {
                int base = -1;
                if (!lane) base = atomicAdd(next_size, cnt);
                base = __shfl_sync(active, base, 0);
                if (valid) next_frontier[base + __popc(mask & ((1u << lane) - 1))] = v;
            }
        }
    }
}

void Graph::bfs(int start, int* d_hops, int* d_frontier, int* d_next_frontier, int* d_fsize,
                int* d_nsize, int* h_fsize) {
    int level = 1;
    *h_fsize = 1;

    bfs_init<<<1024, 512>>>(d_hops, d_frontier, start, total_node_size);

    while (*h_fsize > 0) {
        cudaMemset(d_nsize, 0, sizeof(int));

        bfs_expand_kernel<<<1024, 512>>>(gpu_slots, d_frontier, *h_fsize, d_next_frontier, d_nsize,
                                         d_hops, level++);

        std::swap(d_frontier, d_next_frontier);
        std::swap(d_fsize, d_nsize);
        cudaMemcpy(h_fsize, d_fsize, sizeof(int), cudaMemcpyDeviceToHost);
    }
}

void Graph::FPS(std::vector<int>& selected, int M) {
    auto s = std::chrono::high_resolution_clock::now();
    int start, old_size = selected.size();
    if (old_size < M) {
        fo.print("Generate new start nodes...");
        std::random_device rd;   // 获取真随机种子
        std::mt19937 gen(rd());  // 使用 Mersenne Twister 引擎
        std::uniform_int_distribution<> dis(0, total_node_size - 1);
        Slot* s;
        cudaMallocHost(&s, sizeof(Slot));
        while (true) {
            start = dis(gen);
            cudaMemcpy(s, gpu_slots + start, sizeof(Slot), cudaMemcpyDeviceToHost);
            if (s->deg > 0) break;
        }
        selected.reserve(M);
        selected.push_back(start);
    } else
        return;

    const int tpb = 512, blocks = (total_node_size + tpb) / tpb;
    int *d_min_hops, *d_hops, *d_frontier, *d_next_frontier, *d_fsize, *d_nsize, *h_fsize;
    cudaMalloc(&d_hops, sizeof(int) * total_node_size);
    cudaMalloc(&d_min_hops, sizeof(int) * total_node_size);
    cudaMalloc(&d_frontier, sizeof(int) * total_node_size);
    cudaMalloc(&d_next_frontier, sizeof(int) * total_node_size);
    cudaMalloc(&d_fsize, sizeof(int));
    cudaMalloc(&d_nsize, sizeof(int));
    cudaMallocHost(&h_fsize, sizeof(int));

    thrust::fill(thrust::device_pointer_cast(d_min_hops),
                 thrust::device_pointer_cast(d_min_hops) + total_node_size,
                 std::numeric_limits<int>::max());

    // --- CUB buffers for argmax ---
    size_t temp_bytes = 0;
    cub::DeviceReduce::ArgMax(nullptr, temp_bytes, d_min_hops,
                              (cub::KeyValuePair<int, int>*)nullptr, total_node_size);

    void* d_temp;
    cub::KeyValuePair<int, int>*d_out, *h_out;
    cudaMalloc(&d_temp, temp_bytes);
    cudaMalloc(&d_out, sizeof(cub::KeyValuePair<int, int>));
    cudaMallocHost(&h_out, sizeof(cub::KeyValuePair<int, int>));

    while (selected.size() < M) {
        bfs(start, d_hops, d_frontier, d_next_frontier, d_fsize, d_nsize, h_fsize);

        update_min_hop_kernel<<<blocks, tpb>>>(d_hops, d_min_hops, total_node_size);

        cub::DeviceReduce::ArgMax(d_temp, temp_bytes, d_min_hops, d_out, total_node_size);
        cudaMemcpy(h_out, d_out, sizeof(cub::KeyValuePair<int, int>), cudaMemcpyDeviceToHost);

        start = h_out->key;
        selected.push_back(start);
    }

    cudaFree(d_hops);
    cudaFree(d_min_hops);
    cudaFree(d_frontier);
    cudaFree(d_next_frontier);
    cudaFree(d_fsize);
    cudaFree(d_nsize);
    cudaFree(d_temp);
    cudaFree(d_out);
    cudaFreeHost(h_fsize);
    cudaFreeHost(h_out);

    const float time = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::high_resolution_clock::now() - s)
                           .count() /
                       1000.0f;
    fo.print("FPS(" + TOS(time) + "s): M=" + TOS(M));
}
};  // namespace efanna2e