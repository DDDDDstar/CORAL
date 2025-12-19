#include <cfloat>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>

#include "fileout.h"
#include "graph.cuh"
#include "utils.cuh"

namespace efanna2e {

// kernel: apply updates to slots
__global__ void update_data_kernel(DeviceSlot* __restrict__ slots,
                                   const int* __restrict__ slot_idxs,
                                   const NBD* __restrict__ upd_data, int* delta_degree,
                                   int* delta_noniso_num, int N, int max_degree) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    extern __shared__ int s_deg[];
    for (int slot_i = bid; slot_i < N; slot_i += gridDim.x) {
        const int slot_idx = slot_idxs[slot_i];
        DeviceSlot* slot = slots + slot_idx;
        const int* nbrs = upd_data->nbrs + slot_i * max_degree;
        const float* dists = upd_data->dists + slot_i * max_degree;
        if (tid == 0) {
            const int old_deg = slot->deg;
            slot->deg = *s_deg = upd_data->degs[slot_i];
            assert(*s_deg > 0);
            if (!old_deg) atomicAdd(delta_noniso_num, 1);
            atomicAdd(delta_degree, *s_deg - old_deg);
        }
        __syncthreads();

        for (int nbr_i = tid; nbr_i < *s_deg; nbr_i += tpb) {
            const int nbr_id = slot->nbrs[nbr_i] = nbrs[nbr_i];
            slot->dists[nbr_i] = dists[nbr_i];
        }
        __syncthreads();
    }
}

__global__ void load_data_kernel(const DeviceSlot* __restrict__ slots,
                                 const VecSlot* __restrict__ vec_slots,
                                 const int* __restrict__ slot_idxs,
                                 const int* __restrict__ vec_slot_idxs,
                                 const int* __restrict__ nbr_vec_slot_idxs,
                                 NBD* __restrict__ load_data_out, int N, int dim, int max_degree) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    extern __shared__ int s_deg[];
    for (int slot_i = bid; slot_i < N; slot_i += gridDim.x) {
        const int slot_idx = slot_idxs[slot_i];
        if (tid == 0)
            load_data_out->degs[slot_i] = *s_deg = slot_idx < 0 ? -1 : slots[slot_idx].deg;
        __syncthreads();

        if (*s_deg >= 0) {
            const DeviceSlot* slot = slots + slot_idx;
            const float* vec = vec_slots[vec_slot_idxs[slot_i]].vec;
            const int offset = slot_i * max_degree, *nbr_vec_slots = nbr_vec_slot_idxs + offset;
            int* nbrs_out = load_data_out->nbrs + offset;
            float *dists_out = load_data_out->dists + offset,
                  *nbrs_vec_out = load_data_out->nbrs_vec + offset * dim;

            for (int d = tid; d < dim; d += tpb) load_data_out->vecs[slot_i * dim + d] = vec[d];

            for (int nbr_i = tid; nbr_i < *s_deg; nbr_i += tpb) {
                nbrs_out[nbr_i] = slot->nbrs[nbr_i];
                dists_out[nbr_i] = slot->dists[nbr_i];
                const float* nbr_vec = vec_slots[nbr_vec_slots[nbr_i]].vec;
                float* nbr_vec_out = nbrs_vec_out + nbr_i * dim;
                for (int d = 0; d < dim; d++) {
                    const float f = nbr_vec[d];
                    nbr_vec_out[d] = f;
                }
            }
        }
    }
}

__global__ void load_data_kernel_v2(const DeviceSlot* __restrict__ slots,
                                    const VecSlot* __restrict__ vec_slots,
                                    const int* __restrict__ slot_idxs,
                                    const int* __restrict__ vec_slot_idxs,
                                    const int* __restrict__ nbr_vec_slot_idxs,
                                    NBD* __restrict__ load_data_out, int N, int dim,
                                    int max_degree) {
    const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
    for (int slot_i = bid * tpb + tid; slot_i < N; slot_i += gridDim.x * tpb) {
        const int slot_idx = slot_idxs[slot_i];
        const int deg = load_data_out->degs[slot_i] = slot_idx < 0 ? -1 : slots[slot_idx].deg;

        if (deg >= 0) {
            const DeviceSlot* slot = slots + slot_idx;
            const float* vec = vec_slots[vec_slot_idxs[slot_i]].vec;
            const int offset = slot_i * max_degree, *nbr_vec_slots = nbr_vec_slot_idxs + offset;
            int* nbrs_out = load_data_out->nbrs + offset;
            float *dists_out = load_data_out->dists + offset,
                  *nbrs_vec_out = load_data_out->nbrs_vec + offset * dim;

            for (int d = 0; d < dim; d++) load_data_out->vecs[slot_i * dim + d] = vec[d];

            for (int nbr_i = 0; nbr_i < deg; nbr_i++) {
                nbrs_out[nbr_i] = slot->nbrs[nbr_i];
                dists_out[nbr_i] = slot->dists[nbr_i];
                const float* nbr_vec = vec_slots[nbr_vec_slots[nbr_i]].vec;
                float* nbr_vec_out = nbrs_vec_out + nbr_i * dim;
                for (int d = 0; d < dim; d++) {
                    const float f = nbr_vec[d];
                    nbr_vec_out[d] = f;
                }
            }
        }
    }
}

// evict_random_slots: choose 'need' slot indices randomly from host_cache_dir,
// for each selected slot that is DIRTY, schedule async D2H writeback; otherwise free immediately.
void Graph::evict_random_slots(int need, cudaStream_t& stream) {
    // std::unordered_set<int> evict_node_ids;
    // evict_node_ids.reserve(need);
    int step = 0, evict_num = 0;
    std::mutex mtx;
    std::condition_variable cv;
    while (evict_num < need) {
        for (auto& kv : gpu_cache_dir) {
            const int slot = kv.second, nid = kv.first;
            if (gpu_slot_states[slot].get_wait_num() == 0) {
                // evict_node_ids.insert(kv.first);
                // if (evict_node_ids.size() == need) break;
                if (gpu_slot_states[slot].is_dirty()) {
                    const int cpu_slot = cpu_cache_dir[nid];
                    CUDA_CHECK(cudaMemcpyAsync(cpu_slots + cpu_slot, gpu_slots + slot,
                                               sizeof(HostSlot), cudaMemcpyDeviceToHost, stream));
                    cpu_slot_states[cpu_slot].mark_dirty();
                }

                gpu_slot_states[slot].free();
                remove_from_bucket(gpu_slot_access_counts[slot].access_count, slot);
                gpu_slot_access_counts[slot].access_count = 0;
                gpu_cache_dir.erase(nid);  // remove mapping now (avoid races)
                gpu_free_slots.push(slot);
                if (++evict_num == need) return;
            }
        }

        if (evict_num < need) {
            fo.print("Step " + TOS(step++) + ": waiting for evicting slots(" + TOS(evict_num) +
                     "/" + TOS(need) + ")");
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait_for(lock, std::chrono::milliseconds(1000));
        }
    }
}

void Graph::write_back_all_gpu_slots() {
    for (auto& kv : gpu_cache_dir) {
        const int slot = kv.second, nid = kv.first;
        if (gpu_slot_states[slot].is_dirty()) {
            const int cpu_slot = cpu_cache_dir[nid];
            CUDA_CHECK(cudaMemcpy(cpu_slots + cpu_slot, gpu_slots + slot, sizeof(HostSlot),
                                  cudaMemcpyDeviceToHost));
            cpu_slot_states[cpu_slot].mark_dirty();
            gpu_slot_states[slot].clean();
        }
    }
}

void Graph::write_back_all_cpu_slots() {
    for (auto& kv : cpu_cache_dir) {
        const int nid = kv.first, slot = kv.second;
        if (cpu_slot_states[slot].is_dirty()) {
            DS.write_node_append(nid, cpu_slots[slot], false);
            cpu_slot_states[slot].clean();
        }
    }
}

void Graph::evict_random_cpu_slots(int need) {
    int step = 0, evict_num = 0;
    std::mutex mtx;
    std::condition_variable cv;
    while (true) {
        for (auto& kv : cpu_cache_dir) {
            const int nid = kv.first, slot = kv.second;

            if (gpu_cache_dir.find(nid) != gpu_cache_dir.end()) {
                const int gpu_slot = gpu_cache_dir[nid];
                if (gpu_slot_states[gpu_slot].get_wait_num() == 0) {
                    if (gpu_slot_states[gpu_slot].is_dirty()) {
                        CUDA_CHECK(cudaMemcpy(cpu_slots + slot, gpu_slots + gpu_slot,
                                              sizeof(HostSlot), cudaMemcpyDeviceToHost));
                        cpu_slot_states[slot].mark_dirty();
                    }
                    gpu_slot_states[gpu_slot].free();
                    gpu_cache_dir.erase(nid);
                    gpu_free_slots.push(gpu_slot);
                    remove_from_bucket(gpu_slot_access_counts[gpu_slot].access_count, gpu_slot);
                    gpu_slot_access_counts[gpu_slot].access_count = 0;
                }
            } else
                cpu_slot_states[slot].free();

            if (cpu_slot_states[slot].get_wait_num() == 0) {
                if (cpu_slot_states[slot].is_dirty()) {
                    DS.write_node_append(nid, cpu_slots[slot], false);
                    cpu_slot_states[slot].clean();
                }
                cpu_cache_dir.erase(nid);  // remove mapping now (avoid races)
                cpu_free_slots.push(slot);
                if (++evict_num == need) return;
            }
        }

        if (evict_num < need) {
            fo.print("Step " + TOS(step++) + ": waiting for evicting cpu slots(" + TOS(evict_num) +
                     "/" + TOS(need) + ")");
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait_for(lock, std::chrono::milliseconds(1000));
        }
    }
}

void Graph::evict_random_vec_slots(int need, cudaStream_t& stream) {
    int step = 0, evict_num = 0;
    std::mutex mtx;
    std::condition_variable cv;
    while (evict_num < need) {
        for (auto& kv : gpu_cache_vec_dir) {
            const int slot = kv.second, nid = kv.first;
            // if (gpu_vec_slot_states[slot].get_wait_num() == 0) {
            // gpu_vec_slot_states[slot].free();
            gpu_cache_vec_dir.erase(nid);
            gpu_free_vec_slots.push(slot);
            if (++evict_num == need) return;
            // }
        }

        if (evict_num < need) {
            fo.print("Step " + TOS(step++) + ": waiting for evicting vec slots(" + TOS(evict_num) +
                     "/" + TOS(need) + ")");
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait_for(lock, std::chrono::milliseconds(1000));
        }
    }
}

void Graph::evict_random_cpu_vec_slots(int need) {
    // std::unordered_set<int> evict_vec_ids;
    // evict_vec_ids.reserve(need);
    int step = 0, evict_num = 0;
    std::mutex mtx;
    std::condition_variable cv;
    while (evict_num < need) {
        for (auto& kv : cpu_cache_vec_dir) {
            const int slot = kv.second, nid = kv.first;
            // if (cpu_vec_slot_states[slot].get_wait_num() == 0) {
            // evict_vec_ids.insert(kv.first);
            // if (evict_vec_ids.size() == need) break;
            // cpu_vec_slot_states[slot].free();
            cpu_cache_vec_dir.erase(nid);
            cpu_free_vec_slots.push(slot);
            if (++evict_num == need) return;
            // }
        }

        if (evict_num < need) {
            fo.print("Step " + TOS(step++) + ": waiting for evicting cpu vec slots(" +
                     TOS(evict_num) + "/" + TOS(need) + ")");
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait_for(lock, std::chrono::milliseconds(1000));
        }
    }
}

void Graph::remove_from_bucket(int old_count, int slot) {
    if (!old_count) return;
    const int prev = gpu_slot_access_counts[slot].prev_slot,
              next = gpu_slot_access_counts[slot].next_slot;
    if (prev != -1)
        gpu_slot_access_counts[prev].next_slot = next;
    else
        access_count_buckets[old_count] = next;

    if (next != -1) gpu_slot_access_counts[next].prev_slot = prev;

    gpu_slot_access_counts[slot].prev_slot = gpu_slot_access_counts[slot].next_slot = -1;
}

void Graph::insert_into_bucket(int count, int slot) {
    while (access_count_buckets.size() <= count) access_count_buckets.push_back(-1);

    const int head = access_count_buckets[count];
    gpu_slot_access_counts[slot].next_slot = head;
    gpu_slot_access_counts[slot].prev_slot = -1;

    if (head != -1) gpu_slot_access_counts[head].prev_slot = slot;

    access_count_buckets[count] = slot;
}

int Graph::prefetch_slot(const int node_id) {
    int slot;
    if (cpu_cache_dir.find(node_id) != cpu_cache_dir.end())
        slot = cpu_cache_dir[node_id];
    else {
        fo.print("prefetch_slot node_id: " + TOS(node_id));
        // assert(cpu_slot_size < total_node_size);
        if (cpu_free_slots.empty()) evict_random_cpu_slots(EVICT_BATCH);

        slot = cpu_free_slots.top();
        cpu_free_slots.pop();
        cpu_cache_dir[node_id] = slot;
        DS.load_node_slot(node_id, cpu_slots[slot]);  // load node data from disk to CPU cache
    }
    cpu_slot_states[slot].wait();
    return slot;
}

VecSlot* Graph::prefetch_vec(const int node_id) {
    int slot;
    if (cpu_cache_vec_dir.find(node_id) != cpu_cache_vec_dir.end())
        slot = cpu_cache_vec_dir[node_id];
    else {
        fo.print("prefetch_vec node_id: " + TOS(node_id));
        if (cpu_free_vec_slots.empty()) evict_random_cpu_vec_slots(EVICT_BATCH);

        slot = cpu_free_vec_slots.top();
        cpu_free_vec_slots.pop();
        cpu_cache_vec_dir[node_id] = slot;
        DS.load_node_vec(node_id, cpu_vec_slots[slot]);  // load vec data from disk to CPU cache
    }
    // No need to wait for vec (without update)
    // cpu_vec_slot_states[slot].wait();
    return cpu_vec_slots + slot;
}

void Graph::init_innbrs() {
    for (int nid = 0; nid < cpu_slot_size; nid++) h_innbrs[nid] = std::vector<int>();

    for (int slot = 0; slot < cpu_slot_size; slot++) {
        for (int nbr_i = 0; nbr_i < cpu_slots[slot].deg; nbr_i++)
            h_innbrs[cpu_slots[slot].nbrs[nbr_i]].push_back(cpu_slots[slot].node_id);
    }
}

void Graph::get_innbrs(const std::vector<int>& node_ids, std::vector<int>& h_innbrs_out,
                       std::vector<int>& h_idxs_out) {
    for (int i = 0; i < node_ids.size(); i++) {
        const int node_id = node_ids[i];
        h_innbrs_out.insert(h_innbrs_out.end(), h_innbrs[node_id].begin(),
                            h_innbrs[node_id].end());
        h_idxs_out.insert(
            h_idxs_out.end(), i,
            h_innbrs[node_id].size());  // 每个入邻居对应的 del_node 在 BATCH 中的 idx
    }
}

void Graph::delete_nodes(const std::vector<int>& del_ids) {
    std::unique_lock<std::shared_mutex> lock(slot_mutex);
    std::unordered_set<int> del_set(del_ids.begin(), del_ids.end());
    // 从其他节点的入邻居中删除 del_id:
    for (auto& [_, nbrs] : h_innbrs)
        nbrs.erase(
            std::remove_if(nbrs.begin(), nbrs.end(), [&](int x) { return del_set.count(x); }),
            nbrs.end());

    std::mutex mtx;
    std::condition_variable cv;

    for (int del_id : del_ids) {
        // TODO: 从 disk 删除节点

        // 从 cpu 内存中移除 del_id:
        if (cpu_cache_dir.find(del_id) != cpu_cache_dir.end()) {
            const int cpu_slot = cpu_cache_dir[del_id];
            std::unique_lock<std::mutex> lock(mtx);
            // while (cpu_slot_states[cpu_slot].get_wait_num() > 0)
            //     cv.wait_for(lock, std::chrono::milliseconds(500));  //
            //     等待被删除节点数据不再被需要
            cpu_free_slots.push(cpu_slot);
            cpu_slot_states[cpu_slot].free();
            cpu_cache_dir.erase(del_id);
        }

        if (cpu_cache_vec_dir.find(del_id) != cpu_cache_vec_dir.end()) {
            const int cpu_vec_slot = cpu_cache_vec_dir[del_id];
            std::unique_lock<std::mutex> lock(mtx);
            // while (cpu_vec_slot_states[cpu_vec_slot].get_wait_num() > 0)
            //     cv.wait_for(lock, std::chrono::milliseconds(500));  //
            //     等待被删除节点数据不再被需要
            cpu_free_vec_slots.push(cpu_vec_slot);
            // cpu_vec_slot_states[cpu_vec_slot].free();
            cpu_cache_vec_dir.erase(del_id);
        }

        // 从 gpu 内存中移除 del_id:
        if (gpu_cache_dir.find(del_id) != gpu_cache_dir.end()) {
            const int gpu_slot = gpu_cache_dir[del_id];
            std::unique_lock<std::mutex> lock(mtx);
            while (gpu_slot_states[gpu_slot].get_wait_num() > 0)
                cv.wait_for(lock, std::chrono::milliseconds(500));  // 等待被删除节点数据不再被需要

            gpu_cache_dir.erase(del_id);
            gpu_free_slots.push(gpu_slot);
            gpu_slot_states[gpu_slot].free();

            remove_from_bucket(gpu_slot_access_counts[gpu_slot].access_count, gpu_slot);
            gpu_slot_access_counts[gpu_slot].access_count = 0;
        }

        if (gpu_cache_vec_dir.find(del_id) != gpu_cache_vec_dir.end()) {
            const int gpu_vec_slot = gpu_cache_vec_dir[del_id];

            std::unique_lock<std::mutex> lock(mtx);
            // while (gpu_vec_slot_states[gpu_vec_slot].get_wait_num() > 0)
            //     cv.wait_for(lock, std::chrono::milliseconds(500));  //
            //     等待被删除节点数据不再被需要

            gpu_cache_vec_dir.erase(del_id);
            gpu_free_vec_slots.push(gpu_vec_slot);
            // gpu_vec_slot_states[gpu_vec_slot].free();
        }
    }
}

void Graph::insert_nodes(const float* h_ins_vecs, std::vector<int>& h_ins_ids_out, int ins_num) {
    // TODO: check expand need

    std::unique_lock<std::shared_mutex> lock(slot_mutex);
    h_ins_ids_out.resize(ins_num);
    for (int i = 0; i < ins_num; i++) {
        const int ins_id = h_ins_ids_out[i] = new_node_id++;
        const int slot = cpu_free_slots.top(), vec_slot = cpu_free_vec_slots.top();
        cpu_free_slots.pop();
        cpu_free_vec_slots.pop();

        cpu_cache_dir[ins_id] = slot;
        cpu_slots[slot].node_id = ins_id;
        cpu_slots[slot].deg = 0;

        cpu_cache_vec_dir[ins_id] = vec_slot;
        cpu_vec_slots[vec_slot].copy(ins_id, h_ins_vecs + i * dim);
    }
}

void Graph::mark_used_graph_data(const std::vector<int>& cpu_slot_idxs,
                                 const std::vector<int>& gpu_slot_idxs) {
    for (int slot : cpu_slot_idxs) cpu_slot_states[slot].use();
    for (int slot : gpu_slot_idxs) gpu_slot_states[slot].use();
}

void Graph::update_graph_data(const std::vector<int>& gpu_slot_idxs, const NBD* upd_data, int N,
                              cudaStream_t& stream) {
    // std::unique_lock<std::shared_mutex> lock(slot_mutex);
    // auto start_time = std::chrono::high_resolution_clock::now();
    thrust::device_vector<int> d_slot_idxs(N), d_delta_degree(1, 0), d_delta_noniso_num(1, 0);
    int delta_degree, delta_noniso_num;

    CUDA_CHECK(cudaMemcpyAsync(d_slot_idxs.data().get(), gpu_slot_idxs.data(), sizeof(int) * N,
                               cudaMemcpyHostToDevice, stream));

    update_data_kernel<<<1024, 32, sizeof(int), stream>>>(
        gpu_slots, d_slot_idxs.data().get(), upd_data, d_delta_degree.data().get(),
        d_delta_noniso_num.data().get(), N, max_degree);

    CUDA_CHECK(cudaMemcpyAsync(&delta_degree, d_delta_degree.data().get(), sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&delta_noniso_num, d_delta_noniso_num.data().get(), sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int slot : gpu_slot_idxs) gpu_slot_states[slot].mark_dirty();

    total_degree.fetch_add(delta_degree);
    noniso_num.fetch_add(delta_noniso_num);

#ifdef INFO_PRINT
    fo.print("Update graph: delta_degree-" + TOS(delta_degree) + ", delta_noniso_num-" +
             TOS(delta_noniso_num) + ", avg degree-" +
             TOS(1.0 * total_degree.load() / total_node_size) + ", noniso_num-" +
             TOS(noniso_num.load()));
#endif
}

// flush_all: writeback all DIRTY slots to host synchronously
void Graph::flush_all_to_disk() {
    std::unique_lock<std::shared_mutex> lock(slot_mutex);
    write_back_all_gpu_slots();
    write_back_all_cpu_slots();
}

int Graph::get_top_M_access_nodes(std::vector<int>& res) {
    const int M = SEARCH_START_NODES_NUM;
    std::string res_str, bucket_num;
    res.reserve(M);

    std::shared_lock<std::shared_mutex> lock(slot_mutex);
    for (int bucket_i = access_count_buckets.size() - 1; bucket_i > 0; bucket_i--) {
        int num = 0;
        for (int slot = access_count_buckets[bucket_i]; slot != -1;
             slot = gpu_slot_access_counts[slot].next_slot) {
            num++;
            if (res.size() < M) {
                // assert(cpu_slots[slot].node_id >= 0);
                res.push_back(cpu_slots[slot].node_id);
                res_str += TOS(cpu_slots[slot].node_id) + " ";
            }
        }
        bucket_num += "[" + TOS(bucket_i) + ": " + TOS(num) + "] ";
    }

    fo.print("get_top_M_access_nodes result: " + res_str + "\nbucket num: " + bucket_num);
    // assert(!res.empty());
    return res.size();
}

void Graph::load_data(const std::vector<int>& node_ids, std::vector<int>& cpu_slots_res,
                      std::vector<int>& gpu_slots_res, NBD* d_load_data, int N,
                      cudaStream_t& stream) {
    std::vector<int> vec_slot_idxs(N, -1), nbr_vec_slot_idxs(N * max_degree, -1);
    cpu_slots_res.assign(N, -1);
    gpu_slots_res.assign(N, -1);

    std::unique_lock<std::shared_mutex> lock(slot_mutex);
    // 1) check host-side directory for presence
    for (int i = 0; i < N; i++) {
        const int nid = node_ids[i];
        if (nid < 0) continue;

        const int cpu_slot = cpu_slots_res[i] = prefetch_slot(nid);  // must call to wait
        HostSlot* node_cpu_slot = cpu_slots + cpu_slot;
        int slot, old_count = 0;
        if (gpu_cache_dir.find(nid) == gpu_cache_dir.end()) {
            if (gpu_free_slots.size() == 0) evict_random_slots(EVICT_BATCH, stream);

            slot = gpu_free_slots.top();
            gpu_free_slots.pop();
            CUDA_CHECK(cudaMemcpyAsync(gpu_slots + slot, node_cpu_slot, sizeof(HostSlot),
                                       cudaMemcpyHostToDevice, stream));
            gpu_cache_dir[nid] = slot;
        } else {
            slot = gpu_cache_dir[nid];
            if (gpu_slot_states[slot].is_dirty()) {
                CUDA_CHECK(cudaMemcpy(node_cpu_slot, gpu_slots + slot, sizeof(HostSlot),
                                      cudaMemcpyDeviceToHost));  // 不能 async！
                gpu_slot_states[slot].clean();
                cpu_slot_states[cpu_slot].mark_dirty();
            }
            old_count = gpu_slot_access_counts[slot].access_count++;
            remove_from_bucket(old_count, slot);
        }
        gpu_slot_states[slot].wait();
        gpu_slots_res[i] = slot;
        insert_into_bucket(old_count + 1, slot);

        if (gpu_cache_vec_dir.find(nid) == gpu_cache_vec_dir.end()) {
            VecSlot* node_cpu_vec_slot = prefetch_vec(nid);
            if (gpu_free_vec_slots.size() == 0) evict_random_vec_slots(EVICT_BATCH, stream);

            gpu_cache_vec_dir[nid] = slot = gpu_free_vec_slots.top();
            gpu_free_vec_slots.pop();
            CUDA_CHECK(cudaMemcpyAsync(gpu_vec_slots + slot, node_cpu_vec_slot, sizeof(HostSlot),
                                       cudaMemcpyHostToDevice, stream));
        } else
            slot = gpu_cache_vec_dir[nid];

        vec_slot_idxs[i] = slot;

        for (int nbr_i = 0; nbr_i < node_cpu_slot->deg; nbr_i++) {
            const int nbrid = node_cpu_slot->nbrs[nbr_i];
            // assert(nbrid >= 0);
            if (gpu_cache_vec_dir.find(nbrid) == gpu_cache_vec_dir.end()) {
                VecSlot* nbr_cpu_vec_slot = prefetch_vec(nbrid);
                if (gpu_free_vec_slots.size() == 0) evict_random_vec_slots(EVICT_BATCH, stream);

                gpu_cache_vec_dir[nid] = slot = gpu_free_vec_slots.top();
                gpu_free_vec_slots.pop();
                CUDA_CHECK(cudaMemcpyAsync(gpu_vec_slots + slot, nbr_cpu_vec_slot,
                                           sizeof(HostSlot), cudaMemcpyHostToDevice, stream));
            } else
                slot = gpu_cache_vec_dir[nbrid];

            nbr_vec_slot_idxs[i * max_degree + nbr_i] = slot;
        }
    }

    thrust::device_vector<int> d_slot_idxs(N), d_vec_slot_idxs(N),
        d_nbr_vec_slot_idxs(N * max_degree);

    CUDA_CHECK(cudaMemcpyAsync(d_slot_idxs.data().get(), gpu_slots_res.data(), sizeof(int) * N,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_vec_slot_idxs.data().get(), vec_slot_idxs.data(), sizeof(int) * N,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_nbr_vec_slot_idxs.data().get(), nbr_vec_slot_idxs.data(),
                               sizeof(int) * N * max_degree, cudaMemcpyHostToDevice, stream));

    int blocks = max_pow2_le(N);
    if (blocks <= 1024)
        load_data_kernel<<<max_pow2_le(N), 32, sizeof(int), stream>>>(
            gpu_slots, gpu_vec_slots, d_slot_idxs.data().get(), d_vec_slot_idxs.data().get(),
            d_nbr_vec_slot_idxs.data().get(), d_load_data, N, dim, max_degree);
    else
        load_data_kernel_v2<<<(N + 256 - 1) / 256, 256, 0, stream>>>(
            gpu_slots, gpu_vec_slots, d_slot_idxs.data().get(), d_vec_slot_idxs.data().get(),
            d_nbr_vec_slot_idxs.data().get(), d_load_data, N, dim, max_degree);

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

Graph::Graph(const std::string& graph_file, float* h_base, int cache_capacity, int num_nodes,
             int max_degree, int dim, int k, bool construct)
    : DS("", graph_file, num_nodes, dim, max_degree, construct),
      cpu_slot_size(num_nodes),
      total_node_size(num_nodes),
      gpu_slot_size(cache_capacity),
      max_degree(max_degree),
      dim(dim),
      k(k),
      construct(construct),
      cpu_slot_states(num_nodes),
      gpu_slot_states(cache_capacity),
      gpu_slot_access_counts(cache_capacity) {
    assert(num_nodes <= CPU_N);

    CUDA_CHECK(cudaMallocHost(&cpu_slots, sizeof(HostSlot) * cpu_slot_size));
    CUDA_CHECK(cudaMallocHost(&cpu_vec_slots, sizeof(VecSlot) * cpu_slot_size));
    CUDA_CHECK(cudaMalloc(&gpu_slots, sizeof(DeviceSlot) * gpu_slot_size));
    CUDA_CHECK(cudaMalloc(&gpu_vec_slots, sizeof(VecSlot) * gpu_slot_size));

    if (construct) {
        for (int i = 0; i < cpu_slot_size; ++i) {  // 初始化 cpu 上的 graph slots 和 vec slots
            HostSlot& s = cpu_slots[i];
            s.node_id = i;
            s.deg = 0;
            for (int j = 0; j < max_degree; ++j) {
                s.nbrs[j] = -1;
                s.dists[j] = 0.f;
            }
            cpu_vec_slots[i].copy(i, h_base + i * dim);

            cpu_cache_dir[i] = i;
            cpu_cache_vec_dir[i] = i;
        }
    } else {
        std::vector<int> node_ids(num_nodes);
        std::iota(node_ids.begin(), node_ids.end(), 0);
        DS.load_batch_slots(node_ids, cpu_slots);  // 从磁盘加载所有的 graph slots

        for (int i = 0; i < cpu_slot_size; ++i) {  // 初始化 cpu vec slots
            cpu_vec_slots[i].copy(i, h_base + i * dim);
            cpu_cache_vec_dir[i] = i;
        }

        init_innbrs();                // 找到每个 base 向量的入邻居向量
        new_node_id = cpu_slot_size;  // 新节点的 id 从 host_num_nodes 开始
    }

    if (num_nodes <= cache_capacity) {  // 如果节点数小于等于 cache 容量，则全部缓存
        fo.iprint("num_nodes <= cache_capacity, all cached!");
        CUDA_CHECK(cudaMemcpy(gpu_slots, cpu_slots, num_nodes * sizeof(HostSlot),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_vec_slots, cpu_vec_slots, num_nodes * sizeof(VecSlot),
                              cudaMemcpyHostToDevice));
        for (int i = 0; i < num_nodes; i++) {
            gpu_cache_dir[i] = i;
            gpu_cache_vec_dir[i] = i;
        }
        for (int i = num_nodes; i < cache_capacity; i++) {
            gpu_free_slots.push(i);
            gpu_free_vec_slots.push(i);
        }
    } else  // 如果节点数大于 cache 容量，则全部置为空闲
        for (int i = 0; i < gpu_slot_size; ++i) {
            gpu_free_slots.push(i);
            gpu_free_vec_slots.push(i);
        }
}

Graph::Graph(const std::string& base_file, const std::string& graph_file, int num_nodes,
             int cache_capacity, int max_degree, int dim, int k, bool construct)
    : DS(base_file, graph_file, num_nodes, dim, max_degree, construct),
      cpu_slot_size(CPU_N),
      gpu_slot_size(cache_capacity),
      total_node_size(num_nodes),
      max_degree(max_degree),
      dim(dim),
      k(k),
      construct(construct),
      cpu_slot_states(cpu_slot_size),
      //   cpu_vec_slot_states(cpu_slot_size),
      gpu_slot_states(cache_capacity),
      //   gpu_vec_slot_states(cache_capacity),
      gpu_slot_access_counts(cache_capacity) {
    assert(num_nodes > CPU_N && num_nodes <= DISK_N);
    CUDA_CHECK(cudaMallocHost(&cpu_slots, sizeof(HostSlot) * cpu_slot_size));
    CUDA_CHECK(cudaMallocHost(&cpu_vec_slots, sizeof(VecSlot) * cpu_slot_size));
    CUDA_CHECK(cudaMalloc(&gpu_slots, sizeof(DeviceSlot) * gpu_slot_size));
    CUDA_CHECK(cudaMalloc(&gpu_vec_slots, sizeof(VecSlot) * gpu_slot_size));

    for (int i = 0; i < cpu_slot_size; ++i) {
        cpu_free_slots.push(i);
        cpu_free_vec_slots.push(i);
    }

    for (int i = 0; i < gpu_slot_size; ++i) {
        gpu_free_slots.push(i);
        gpu_free_vec_slots.push(i);
    }

    if (!construct) {
        init_innbrs();            // 找到每个 base 向量的入邻居向量
        new_node_id = num_nodes;  // 新节点的 id 从 host_num_nodes 开始
    }
}

Graph::~Graph() {
    cudaDeviceSynchronize();

    cudaFree(gpu_slots);
    cudaFree(gpu_vec_slots);
    cudaFreeHost(cpu_slots);
    cudaFreeHost(cpu_vec_slots);
}

Node_Batch_Data::Node_Batch_Data(int n, int max_degree, int dim, cudaStream_t& stream) : n(n) {
    CUDA_CHECK(cudaMallocAsync(&nbrs, n * max_degree * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&degs, n * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&dists, n * max_degree * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&nbrs_vec, n * max_degree * dim * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&vecs, n * dim * sizeof(float), stream));

    CUDA_CHECK(cudaMemsetAsync(nbrs, -1, n * max_degree * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(degs, 0, n * sizeof(int), stream));
}

void Node_Batch_Data::self_copy(int batch, int single_n, int copy_n, int max_degree, int dim,
                                cudaStream_t& stream) {
    for (int t = 1; t < batch; t++) {
        CUDA_CHECK(cudaMemcpyAsync(nbrs + t * single_n * max_degree, nbrs,
                                   copy_n * max_degree * sizeof(int), cudaMemcpyDeviceToDevice,
                                   stream));
        CUDA_CHECK(cudaMemcpyAsync(degs + t * single_n, degs, copy_n * sizeof(int),
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dists + t * single_n * max_degree, dists,
                                   copy_n * max_degree * sizeof(float), cudaMemcpyDeviceToDevice,
                                   stream));
        CUDA_CHECK(cudaMemcpyAsync(nbrs_vec + t * single_n * max_degree * dim, nbrs_vec,
                                   copy_n * max_degree * dim * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(vecs + t * single_n * dim, vecs, copy_n * dim * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));
    }
}

Node_Batch_Data::~Node_Batch_Data() {
    CUDA_CHECK(cudaFree(nbrs));
    CUDA_CHECK(cudaFree(degs));
    CUDA_CHECK(cudaFree(dists));
    CUDA_CHECK(cudaFree(nbrs_vec));
    CUDA_CHECK(cudaFree(vecs));
}

};  // namespace efanna2e

// void Graph::load_data(const std::vector<int>& node_ids, std::vector<int>& cpu_slots_res,
//                       std::vector<int>& gpu_slots_res, NBD* d_load_data, cudaStream_t& stream) {
//     cudaStreamSynchronize(stream);

//     ensure_slots_for_nodes(node_ids, cpu_slots_res, gpu_slots_res, stream);

//     const int n = node_ids.size();
//     std::vector<int> vec_slot_idxs(n), nbr_vec_slot_idxs(n * max_degree);
//     for (int i = 0; i < n; i++) {
//         const int node_id = node_ids[i];
//         vec_slot_idxs[i] = gpu_cache_vec_dir[node_id];

//         for (int nbr_i = 0; nbr_i < cpu_slots[cpu_cache_dir[node_id]].deg; nbr_i++)
//             nbr_vec_slot_idxs[i * max_degree + nbr_i] =
//                 gpu_cache_vec_dir[cpu_slots[cpu_cache_dir[node_id]].nbrs[nbr_i]];
//     }

//     thrust::device_vector<int> d_slot_idxs(n), d_vec_slot_idxs(n),
//         d_nbr_vec_slot_idxs(n * max_degree);

//     CUDA_CHECK(cudaMemcpyAsync(d_slot_idxs.data().get(), gpu_slots_res.data(), sizeof(int) * n,
//                                cudaMemcpyHostToDevice, stream));
//     CUDA_CHECK(cudaMemcpyAsync(d_vec_slot_idxs.data().get(), vec_slot_idxs.data(), sizeof(int) *
//     n,
//                                cudaMemcpyHostToDevice, stream));
//     CUDA_CHECK(cudaMemcpyAsync(d_nbr_vec_slot_idxs.data().get(), nbr_vec_slot_idxs.data(),
//                                sizeof(int) * n * max_degree, cudaMemcpyHostToDevice, stream));

//     {
//         std::shared_lock<std::shared_mutex> lock(data_mutex);
//         const int threads = 32, blocks = 1024,
//                   shared_size = sizeof(DeviceSlot*) + sizeof(VecSlot*) + sizeof(int);
//         load_data_kernel<<<blocks, threads, shared_size, stream>>>(
//             gpu_slots, gpu_vec_slots, d_slot_idxs.data().get(), d_vec_slot_idxs.data().get(),
//             d_nbr_vec_slot_idxs.data().get(), d_load_data, dim, max_degree);
//         CUDA_CHECK(cudaStreamSynchronize(stream));
//     }

//     for (int i = 0; i < n; i++) {
//         const int nid = node_ids[i];
//         gpu_vec_slot_states[vec_slot_idxs[i]].use();
//         cpu_vec_slot_states[cpu_cache_vec_dir[nid]].use();

//         HostSlot& slot = cpu_slots[cpu_slots_res[i]];
//         for (int nbr_i = 0; nbr_i < slot.deg; nbr_i++) {
//             gpu_vec_slot_states[nbr_vec_slot_idxs[i * max_degree + nbr_i]].use();
//             cpu_vec_slot_states[cpu_cache_vec_dir[slot.nbrs[nbr_i]]].use();
//         }
//     }
// }

// __global__ void load_graph_data_kernel(DeviceSlot* slots, const int* slot_idxs, NBD*
// load_data_out,
//                                        int dim, int max_degree) {
//     const int bid = blockIdx.x, tid = threadIdx.x, tpb = blockDim.x;
//     extern __shared__ DeviceSlot* s_slot_ptr[];
//     int *s_deg = (int*)(s_slot_ptr + 1), *s_indeg = (int*)(s_deg + 1);
//     for (int slot_i = bid; slot_i < load_data_out->n;
//          slot_i += gridDim.x) {  // one block for one slot_node
//         if (tid == 0) {
//             *s_slot_ptr = slots + slot_idxs[slot_i];
//             *s_deg = load_data_out->degs[slot_i] = (*s_slot_ptr)->deg;
//         }
//         __syncthreads();

//         // write to neighbors
//         for (int nbr_i = tid; nbr_i < *s_deg; nbr_i += tpb) {
//             load_data_out->nbrs[slot_i * max_degree + nbr_i] = (*s_slot_ptr)->nbrs[nbr_i];
//             load_data_out->dists[slot_i * max_degree + nbr_i] = (*s_slot_ptr)->dists[nbr_i];
//         }
//     }
// }

// ensure_slots_for_nodes: make sure each node in 'nodes' has a slot in GPU cache.
// if missing, allocate slot (evict random if needed) and schedule device H2D copy.
// void Graph::ensure_slots_for_nodes(const std::vector<int>& nodes, std::vector<int>&
// cpu_slots_res,
//                                    std::vector<int>& gpu_slots_res, cudaStream_t& stream) {
//     const int N = nodes.size();
//     cpu_slots_res.resize(N);
//     gpu_slots_res.resize(N);

//     std::unique_lock<std::shared_mutex> lock(slot_mutex);

//     // 1) check host-side directory for presence
//     for (int i = 0; i < N; i++) {
//         const int nid = nodes[i];
//         cpu_slots_res[i] = prefetch_slot(nid);  // must call to wait
//         HostSlot& node_cpu_slot = cpu_slots[cpu_slots_res[i]];
//         int slot, old_count = 0;
//         if (gpu_cache_dir.find(nid) == gpu_cache_dir.end()) {
//             if (gpu_free_slots.size() == 0) evict_random_slots(EVICT_BATCH, stream);

//             slot = gpu_free_slots.top();
//             gpu_free_slots.pop();
//             CUDA_CHECK(cudaMemcpyAsync(gpu_slots + slot, &node_cpu_slot, sizeof(HostSlot),
//                                        cudaMemcpyHostToDevice, stream));
//             gpu_slot_states[slot].wait();
//             gpu_cache_dir[nid] = slot;
//             gpu_slots_res[i] = slot;
//         } else {
//             slot = gpu_cache_dir[nid];
//             gpu_slot_states[slot].wait();
//             gpu_slots_res[i] = slot;
//             old_count = gpu_slot_access_counts[slot].access_count++;
//             remove_from_bucket(old_count, slot);
//         }
//         insert_into_bucket(old_count + 1, slot);

//         if (gpu_cache_vec_dir.find(nid) == gpu_cache_vec_dir.end()) {
//             VecSlot& node_cpu_vec_slot = prefetch_vec(nid);
//             if (gpu_free_vec_slots.size() == 0) evict_random_vec_slots(EVICT_BATCH, stream);

//             const int slot = gpu_free_vec_slots.top();
//             gpu_free_vec_slots.pop();
//             CUDA_CHECK(cudaMemcpyAsync(gpu_vec_slots + slot, &node_cpu_vec_slot,
//             sizeof(HostSlot),
//                                        cudaMemcpyHostToDevice, stream));
//             gpu_cache_vec_dir[nid] = slot;  // host-side dir assign
//             gpu_vec_slot_states[slot].wait();
//         } else
//             gpu_vec_slot_states[gpu_cache_vec_dir[nid]].wait();

//         for (int nbr_i = 0; nbr_i < node_cpu_slot.deg; nbr_i++) {
//             const int nbrid = node_cpu_slot.nbrs[nbr_i];
//             // assert(nbrid >= 0);
//             if (gpu_cache_vec_dir.find(nbrid) == gpu_cache_vec_dir.end()) {
//                 VecSlot& nbr_cpu_vec_slot = prefetch_vec(nbrid);
//                 if (gpu_free_vec_slots.size() == 0) evict_random_vec_slots(EVICT_BATCH, stream);

//                 const int slot = gpu_free_vec_slots.top();
//                 gpu_free_vec_slots.pop();
//                 CUDA_CHECK(cudaMemcpyAsync(gpu_vec_slots + slot, &nbr_cpu_vec_slot,
//                                            sizeof(HostSlot), cudaMemcpyHostToDevice, stream));
//                 gpu_cache_vec_dir[nid] = slot;  // host-side dir assign
//                 gpu_vec_slot_states[slot].wait();
//             } else
//                 gpu_vec_slot_states[gpu_cache_vec_dir[nbrid]].wait();
//         }
//     }
//     CUDA_CHECK(cudaStreamSynchronize(stream));
// }
