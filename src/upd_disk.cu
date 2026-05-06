#include <cfloat>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>

#include "fileout.h"
#include "gpufuncs.cuh"
#include "utils.cuh"

namespace efanna2e {
__global__ void pair_dist_compute_kernel(
    // const UpdData data,
    const int* __restrict__ knn_ids, const float* __restrict__ d_vecs,
    const float* __restrict__ d_nbr_vecs, const Slot* __restrict__ d_slots,
    CN* __restrict__ cand_nbrs, float* __restrict__ cand_dists, int batch, int tile_k, BP bp) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x,
              cand_size = bp.k + bp.max_degree;
    extern __shared__ float tile_row_vecs[];  // 存储一个 tile 的基础向量 [tile_k * dim]
    for (int batch_id = bid; batch_id < batch; batch_id += gridDim.x) {
        const int offset = batch_id * bp.k, nbr_idx_start = batch * bp.k + offset * bp.max_degree,
                  *ids = knn_ids + offset;
        const float *vecs = d_vecs + (size_t)offset * bp.dim,
                    *nbr_vecs = d_nbr_vecs + (size_t)offset * bp.max_degree * bp.dim;
        const Slot* slots = d_slots + offset;
        const float farthest = farthest_dist_d(bp);

        for (int row_tile = 0; row_tile < bp.k; row_tile += tile_k) {
            // 加载 tile_k 个 row 向量进共享内存:
            for (int i = tid; i < tile_k * bp.dim; i += tpb) {
                // 线程处理的tile 中第 i / dim 个向量在其所在 kNNs 中的索引
                const int global_idx = row_tile + i / bp.dim;
                if (global_idx >= bp.k) break;
                tile_row_vecs[i] = vecs[global_idx * bp.dim + i % bp.dim];
            }
            __syncthreads();

            // 计算加载进共享内存的 tile_k 个 row 向量和 bp.k 个列向量的距离
            // 每个线程负责 tile_k * bp.k 距离矩阵中一个点的计算
            for (int i = tid; i < tile_k * cand_size; i += tpb) {
                const int local_row = i / cand_size, global_row = row_tile + local_row;
                if (global_row >= bp.k) break;

                const int global_col = i % cand_size,
                          matrix_idx = offset * cand_size + global_row * cand_size + global_col;
                const Slot& row_slot = slots[global_row];
                const int row_deg = row_slot.deg;
                const int row_id = ids[global_row];
                if (row_id < 0 || global_col >= bp.k + row_deg || global_row == global_col) {
                    cand_nbrs[matrix_idx] = {-1, 0, Invalid, farthest};
                    cand_dists[matrix_idx] = farthest;
                    continue;
                }
                if (global_col >= bp.k) {
                    const int nbr_i = global_col - bp.k, nbr_id = row_slot.nbrs[nbr_i];
                    const float dist = cand_dists[matrix_idx] = row_slot.dists[nbr_i];
                    cand_nbrs[matrix_idx] = {row_slot.nbrs[nbr_i],
                                             nbr_idx_start + global_row * bp.max_degree + nbr_i,
                                             Initial, dist};
                } else if (global_row < global_col) {
                    const int col_id = ids[global_col];
                    if (col_id < 0) {
                        cand_nbrs[matrix_idx] = {-1, 0, Invalid, farthest};
                        cand_dists[matrix_idx] = farthest;
                        continue;
                    }
                    const Slot& col_slot = slots[global_col];
                    const int sym_matrix_idx =
                        offset * cand_size + global_col * cand_size + global_row;
                    assert(row_id != col_id);
                    if (row_id == col_id || row_deg > 0 && col_slot.deg > 0) {
                        cand_dists[matrix_idx] = cand_dists[sym_matrix_idx] = farthest;
                        cand_nbrs[matrix_idx] =
                            cand_nbrs[sym_matrix_idx] = {-1, 0, Invalid, farthest};
                    } else {
                        const float dist = calc_distance(tile_row_vecs + local_row * bp.dim,
                                                         vecs + global_col * bp.dim, bp);
                        cand_nbrs[matrix_idx] = {col_id, offset + global_col, Initial, dist};
                        cand_nbrs[sym_matrix_idx] = {row_id, offset + global_row, Initial, dist};
                        cand_dists[matrix_idx] = cand_dists[sym_matrix_idx] = dist;
                    }
                }
            }
            __syncthreads();  // 等待所有线程完成当前 tile 行的计算
        }
    }
}

__device__ __forceinline__ int atomicTestAndSetBit(uint32_t* bitset, int bit_index) {
    const uint32_t mask = 1u << (bit_index & 31);
    uint32_t* addr = bitset + (bit_index >> 5);
    const uint32_t old = atomicOr(addr, mask);
    return (old & mask) != 0;
}
__global__ void update_new_nbrs_kernel(Slot* __restrict__ slots, CN* __restrict__ cand_nbrs,
                                       int pivot_num, int cand_size, BP bp) {
    const int tid = threadIdx.x, pivot_i = blockIdx.x * blockDim.x + tid;
    if (pivot_i < pivot_num) {
        int nbr_num = 0, last_id = -1;
        Slot& slot = slots[pivot_i];
        const CN* cands = cand_nbrs + pivot_i * cand_size;
        int retained = 0;
        for (int i = 0; i < cand_size && nbr_num < bp.max_degree; i++) {
            const CN& cand_nbr = cands[i];
            const auto status = cand_nbr.status;
            if (status == Retained) {
                retained++;
                // const int cand_nbr_id = cand_nbr.id;
                // assert(cand_nbr_id >= 0);
                // assert(cand_nbr_id != last_id);
                // const int idx = nbr_num++;
                // slot.nbrs[idx] = last_id = cand_nbr_id;
                // slot.dists[idx] = cand_nbr.dist;
            } else if (status == Invalid)
                break;
        }
        if (!retained) return;
        int discarded_need = bp.max_degree - retained;
        for (int j = 0; j < cand_size && nbr_num < bp.max_degree; j++) {
            const CN& cand_nbr = cands[j];
            const auto status = cand_nbr.status;
            if (status == Retained) {
                const int cand_nbr_id = cand_nbr.id;
                assert(cand_nbr_id >= 0);
                assert(cand_nbr_id != last_id);
                const int idx = nbr_num++;
                slot.nbrs[idx] = last_id = cand_nbr_id;
                slot.dists[idx] = cand_nbr.dist;
            } else if (status == Discarded && (discarded_need--) > 0) {
                const int cand_nbr_id = cand_nbr.id;
                assert(cand_nbr_id >= 0);
                assert(cand_nbr_id != last_id);
                const int idx = nbr_num++;
                last_id = slot.nbrs[idx] = cand_nbr_id;
                slot.dists[idx] = cand_nbr.dist;
            } else if (cand_nbr.status == Invalid)
                break;
        }
        if (nbr_num > 0) slot.deg = nbr_num;
    }
}

void GPUFuncs::disk_neibor_aware(UpdData* data) {
    const int cand_size = bp.max_degree + bp.k, pivot_num = data->batch * bp.k,
              size = pivot_num * cand_size, tpb = 512, blocks = (pivot_num + tpb - 1) / tpb;

    // 计算每组中节点两两间距，生成候选集（包括旧邻居）
    pair_dist_compute_kernel<<<data->batch, tpb, tile_k * bp.dim * sizeof(float), data->stream>>>(
        // *data,
        data->d_knn_ids, data->d_vecs, data->d_nbr_vecs, data->d_slots, data->cand_nbrs,
        data->cand_dists, data->batch, tile_k, bp);
    CUDA_CHECK(cudaGetLastError());

    // dist_extract_from_CN(data->cand_nbrs, data->cand_dists, data->stream, size, bp);

    auto res = segmented_sort_pairs_nocpy(data->cand_nbrs, data->cand_dists, data->cand_nbrs_sort,
                                          data->cand_dists_sort, data->offsets, data->stream,
                                          pivot_num, cand_size);

    // 2.2. 针对 batch 组 cand_size 近邻数据，
    // 每次迭代线程并行进行 𝑏𝑎𝑡𝑐ℎ * bp.k * (bp.k − 2 − 𝑖) 次淘汰
    // 根据每个 pivot 候选邻居到 pivot 的距离，从第一个开始，
    // 从近到远成为邻居并并行淘汰后面的后续邻居
    candidate_ignore_kernel<<<blocks, tpb, 0, data->stream>>>(data->d_vecs, res.first, pivot_num,
                                                              cand_size, bp);
    CUDA_CHECK(cudaGetLastError());

    // thrust::fill(thrust::cuda::par.on(data->stream), data->updated,
    //              data->updated + (GPU_N + 31) / 32, 0);
    update_new_nbrs_kernel<<<blocks, tpb, 0, data->stream>>>(data->d_slots, res.first, pivot_num,
                                                             cand_size, bp);
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaStreamSynchronize(data->stream));
}

void GPUFuncs::Upd_loop() {
    CUDA_CHECK(cudaSetDevice(0));
    float load_time = 0.0f, nap_time = 0.0f, write_time = 0.0f;
    int cnt = 0;
    while (true) {
        UpdData* data;
        auto s = now_time();
        {
            std::unique_lock<std::shared_mutex> lock(read_mtx);
            read_cv.wait(lock, [this] { return upd_queue->notempty() || upd_queue->stop(); });
        }
        if (upd_queue->empty()) break;
        data = upd_queue->data();
        const int batch_id = data->batch_id;
        auto s1 = now_time();
        if (batch_id > 0) load_time += time_diff(s, s1);
        disk_neibor_aware(data);
        data->need_write = true;
        auto s2 = now_time();
        nap_time += time_diff(s1, s2);
        {
            std::unique_lock<std::shared_mutex> lock(write_mtx);
            upd_queue->read_finish();
        }
        write_cv.notify_one();
        write_time += time_diff(s2);
#ifdef VERBOSE
        if ((++cnt) % 50 == 0)
            fo.print("UPD loop: batch id=" + TOS(batch_id) + ", " + TOS(load_time / cnt) + "s, " +
                     TOS(nap_time / cnt) + "s, " + TOS(write_time / cnt) + "s");
#endif
    }
    for (auto& data : upd_queue->datas) {
        if (data.need_write) {
            data.need_write = false;
            data.write(upd_queue->h_upd_slots);
            graph->write_data(upd_queue->h_upd_slots, data.batch * bp.k);
        }
    }
    fo.print("UPD loop stop");
}

void GPUFuncs::graph_update(const int* knn_ids, int batch_id, int batch) {
    cudaStream_t stream = streams["upd"].stream;
    auto gt = graph_type();
    // fo.print("batch id: " + TOS(batch_id) + " start with graph=" + TOS(int(gt)));
    if (gt == GraphType::GPU) {
        // fo.print("graph update batch " + TOS(batch_id));
        gpu_neibor_aware(knn_ids, batch, bp.k, stream);
#ifdef VERBOSE
        if (batch_id > 0 && batch_id % 50 == 0) {
            cudaMemcpyAsync(h_total_degree, d_total_degree, sizeof(ull), cudaMemcpyDeviceToHost,
                            stream);
            cudaMemcpyAsync(h_noniso_num, d_noniso_num, sizeof(ull), cudaMemcpyDeviceToHost,
                            stream);
            cudaStreamSynchronize(stream);
            fo.print("batch " + TOS(batch_id) +
                     " avg degree: " + TOS(*h_total_degree * 1.0 / bp.base_n) +
                     ", noniso num: " + TOS(*h_noniso_num));
        }
#endif
    } else if (!MEM_MODE) {
        auto gpu_cache = graph->GPUCache();
        NAPData* data;
        {
            std::unique_lock<std::shared_mutex> lock(write_mtx);
            write_cv.wait(lock, [this] { return nap_queue->notfull(); });
        }
        if (nap_queue->stop()) return;
        data = nap_queue->newdata();
        data->load_knns(knn_ids, batch, batch_id);
        auto res = gpu_cache->ensure_cached_from_device_ids(
            data->d_knn_ids, batch * bp.k, (uint32_t)batch_id, true, data->stream);
        int num;
        {
            std::unique_lock<std::shared_mutex> lock(read_mtx);
            num = nap_queue->write_finish();
        }
        read_cv.notify_one();
    } else {
        UpdData* data;
        {
            std::unique_lock<std::shared_mutex> lock(write_mtx);
            write_cv.wait(lock, [this] { return upd_queue->notfull() || upd_queue->stop(); });
        }
        if (upd_queue->stop()) return;

        data = upd_queue->newdata();
        if (data->need_write) {
            data->need_write = false;
            data->write(upd_queue->h_upd_slots);
            graph->write_data(upd_queue->h_upd_slots, data->batch * bp.k);
        }
        data->load_knns(knn_ids, batch, batch_id);
        graph->load_data(data->h_knn_ids, batch * bp.k, upd_queue->h_upd_slots,
                         upd_queue->h_upd_vecs, true);
        data->load(upd_queue->h_upd_slots, upd_queue->h_upd_vecs);
        {
            std::unique_lock<std::shared_mutex> lock(read_mtx);
            upd_queue->write_finish();
        }
        read_cv.notify_one();
    }
}

void GPUFuncs::upd_prepare(bool construct, int batch, int nap_num) {
    auto gt = graph_type();
    if (construct && gt != GraphType::GPU) {
        if (!MEM_MODE) {
            nap_queue = new NAPQueue<NAPData>(bp);
            nap_thread = std::thread(&GPUFuncs::NAP_loop, this);
        } else {
            upd_queue = new NAPQueue<UpdData>(bp);
            nap_thread = std::thread(&GPUFuncs::Upd_loop, this);
        }
    } else {
        const int cand_size = bp.max_degree + nap_num, size = batch * nap_num * cand_size;
        cand_nbrs.resize(size);
        cand_nbrs_sort.resize(size);
        offsets.resize(batch * nap_num + 1);
        cand_dists.resize(size);
        cand_dists_sort.resize(size);
        d_updated.resize((bp.base_n + 31) / 32);
    }

    // 共享内存最大向量存储数量:
    int max_vec_num = (shared_mem_per_block - 2 * bp.k * sizeof(int)) / (bp.dim * sizeof(float));
    // 每个 block 根据共享内存大小限制分块处理对应的 KNN:
    tile_k = std::min(nap_num, max_vec_num);
}

void GPUFuncs::upd_free() {
    if (nap_thread.joinable()) nap_thread.join();

    thrust::device_vector<CN>().swap(cand_nbrs);
    thrust::device_vector<size_t>().swap(offsets);
    thrust::device_vector<float>().swap(cand_dists);
    thrust::device_vector<float>().swap(cand_dists_sort);
    if (nap_queue) delete nap_queue;
    if (upd_queue) delete upd_queue;
}
};  // namespace efanna2e
