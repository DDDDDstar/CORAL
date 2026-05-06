#include "graph.cuh"

#include <cfloat>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>

namespace efanna2e {

void Graph::evict_random_cpu_vec_slots(int need) {
    int step = 0, evict_num = 0;
    std::mutex mtx;
    std::condition_variable cv;
    std::vector<int> evict_slots;
    while (evict_num < need) {
        for (auto& kv : cpu_cache_vec_dir) {
            const int nid = kv.first, slot = kv.second;
            slot_lock(nid);
            if (vec_slot_states[slot].can_evict()) {
                cpu_cache_vec_dir.erase(nid);
                evict_slots.push_back(slot);
                if (++evict_num == need) {
                    slot_unlock(nid);
                    break;
                }
            }
            slot_unlock(nid);
        }

        if (evict_num < need) {
            fo.print("Step " + TOS(step++) + ": waiting for evicting cpu slots(" + TOS(evict_num) +
                     "/" + TOS(need) + ")");
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait_for(lock, std::chrono::milliseconds(1000));
        }
    }
    cpu_free_vec_slots.batch_push(evict_slots);
}

void Graph::evict_random_cpu_slots(int need) {
    int step = 0, evict_num = 0;
    std::mutex mtx;
    std::condition_variable cv;
    std::vector<int> evict_slots;
    while (evict_num < need) {
        for (auto& kv : cpu_cache_dir) {
            const int nid = kv.first, slot = kv.second;
            vec_lock(nid);
            if (slot_states[slot].can_evict()) {
                if (slot_states[slot].is_dirty()) {
                    DS.write_slot_async(nid, cpu_slots[slot]);
                    slot_states[slot].clean();
                }
                cpu_cache_dir.erase(nid);
                evict_slots.push_back(slot);
                if (++evict_num == need) {
                    vec_unlock(nid);
                    break;
                }
            }
            vec_unlock(nid);
        }

        if (evict_num < need) {
            fo.print("Step " + TOS(step++) + ": waiting for evicting cpu slots(" + TOS(evict_num) +
                     "/" + TOS(need) + ")");
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait_for(lock, std::chrono::milliseconds(1000));
        }
    }
    cpu_free_slots.batch_push(evict_slots);
}

const Slot& Graph::load_data_in_cpu(int id) {
    slot_lock(id);
    auto it = cpu_cache_dir.find(id);
    if (it != cpu_cache_dir.end()) {
        const int idx = it->second;
        slot_states[idx].accumulate();
        return cpu_slots[idx];
    }

    if (!cpu_free_slots.size()) evict_random_cpu_slots(1);

    int idx = cpu_free_slots.pop();
    DS.load_node_slot(id, cpu_slots[idx]);
    slot_states[idx].accumulate();
    cpu_cache_dir[id] = idx;
    return cpu_slots[idx];
}

const float* Graph::load_vec_in_cpu(int id) {
    vec_lock(id);
    auto it = cpu_cache_vec_dir.find(id);
    if (it != cpu_cache_vec_dir.end()) {
        const int idx = it->second;
        vec_slot_states[idx].accumulate();
        return cpu_vec_slots.data() + (size_t)idx * dim;
    }

    if (!cpu_free_vec_slots.size()) evict_random_cpu_vec_slots(1);

    int idx = cpu_free_vec_slots.pop();
    float* cpu_vec = cpu_vec_slots.data() + (size_t)idx * dim;
    DS.load_node_vec(id, cpu_vec);
    vec_slot_states[idx].accumulate();
    cpu_cache_vec_dir[id] = idx;
    return cpu_vec;
}

/**
 * 将所有CPU槽位的数据写回存储系统
 * 根据图类型(GraphType)选择不同的写入策略
 */
void Graph::write_back_all_cpu_slots() {
    // 如果图类型是GPU或CPU，则使用并行写入策略
    if (gtype == GraphType::GPU || gtype == GraphType::CPU)
#pragma omp parallel for  // 使用OpenMP并行化循环
        // 遍历所有节点，将每个节点的CPU槽位数据写入存储系统
        for (int i = 0; i < total_node_size; i++) DS.write_node_append(i, cpu_slots[i]);
    else {
        for (auto it : cpu_cache_dir) {
            // 遍历CPU缓存目录中的每个条目
            const int slot = it.second, id = it.first;
            assert(id >= 0);
            // 确保ID和槽位号有效
            assert(slot >= 0);
            if (slot_states[slot].is_dirty()) {
                // 如果槽位状态为"脏"(已被修改)，则异步写入并清理状态
                DS.write_slot_async(id, cpu_slots[slot]);
                slot_states[slot].clean();
            }
        }
    }
}

void Graph::init_innbrs() {
    assert(total_node_size <= GPU_N);
    h_innbrs.resize(total_node_size * 2);
    del_flags.assign(total_node_size * 2, 0);
    // for (int nid = 0; nid < total_node_size; nid++) h_innbrs[nid] = std::vector<int>();

    for (int slot = 0; slot < total_node_size; slot++) {
        const int nid = cpu_slots[slot].node_id, deg = cpu_slots[slot].deg;
        total_degree += deg;
        const int* nbrs = cpu_slots[slot].nbrs;
        for (int nbr_i = 0; nbr_i < deg; nbr_i++) h_innbrs[nbrs[nbr_i]].insert(nid);
    }

    fo.print("init_innbrs: avg_degree=" + TOS(float(total_degree) / total_node_size));
}

void Graph::get_del_innbrs(const std::vector<int>& node_ids, std::vector<int>& innbrs_out,
                           std::vector<int>& offsets) {
    const int N = node_ids.size();
    // fo.print("get innbrs start: node num=" + TOS(N));
    auto s = std::chrono::high_resolution_clock::now();
    innbrs_out.clear();
    innbrs_out.reserve(N * 40);
    offsets.resize(N + 1);
    offsets[0] = 0;

    // std::shared_lock<std::shared_mutex> lock(innbr_mutex);

    if (total_node_size <= GPU_N) {
        // std::vector<uint8_t> del_flag(total_node_size * 2, 0);
        for (int id : node_ids) del_flags[id] = 1;
        for (int i = 0; i < N; i++) {
            const int node_id = node_ids[i];
            for (int nbr : h_innbrs[node_id])
                if (!del_flags[nbr]) innbrs_out.push_back(nbr);
            // innbrs_out.insert(
            //     innbrs_out.end(), h_innbrs[node_id].begin(),
            //     h_innbrs[node_id].end());  // 不筛掉删除节点（留着 GPU
            //     处理），直接返回所有入邻居
            offsets[i + 1] = innbrs_out.size();
        }
    } else {
        std::vector<int> del_ids = node_ids;
        std::sort(del_ids.begin(), del_ids.end());
        for (int i = 0; i < N; i++) {
            const int node_id = node_ids[i];
            for (int nbr : h_innbrs[node_id])
                if (!std::binary_search(del_ids.begin(), del_ids.end(), nbr)) {
                    innbrs_out.push_back(nbr);
                }
            offsets[i + 1] = innbrs_out.size();
        }
    }
    auto e = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0;
    fo.print("get_innbrs(" + TOS(time) + "s): total innbr num=" + TOS(innbrs_out.size()));
}

void Graph::update_innbrs_with_upd_nbrs(const int* pivot_ids, const int* upd_nbrs, int pivot_num) {
    assert(gtype == GraphType::GPU);

    // std::unique_lock<std::shared_mutex> lock(innbr_mutex);
    for (int i = 0; i < pivot_num; i++) {
        const int pivot_id = pivot_ids[i], old_nbr = upd_nbrs[i * 2],
                  new_nbr = upd_nbrs[i * 2 + 1];
        if (new_nbr >= 0) {
            h_innbrs[new_nbr].insert(pivot_id);
            if (old_nbr >= 0) h_innbrs[old_nbr].erase(pivot_id);
        }
    }
}

void Graph::update_innbrs_with_new_nbrs(const int* pivot_ids, const int* new_nbrs, int pivot_num) {
    assert(gtype == GraphType::GPU);

    // std::unique_lock<std::shared_mutex> lock(innbr_mutex);
    for (int i = 0; i < pivot_num; i++) {
        const int pivot_id = pivot_ids[i], *nbrs = new_nbrs + i * max_degree;
        for (int j = 0; j < max_degree; j++) {
            const int nbr_id = nbrs[j];
            if (nbr_id < 0) break;
            h_innbrs[nbr_id].insert(pivot_id);
        }
    }
}

void Graph::delete_nodes(const int* del_ids, const int* innbr_ids, const int* old_nbrs,
                         const int* new_nbrs, int del_num, int innbrs_num) {
    assert(gtype == GraphType::GPU);

    update_start_ids(del_ids, del_num);

    // 2. 对于 innbr_ids 中每个节点（即删除节点的所有入邻居），将其添加到其每个新邻居的入邻居表中
    for (int i = 0; i < innbrs_num; ++i) {
        const int innbr_id = innbr_ids[i], base = i * max_degree;
        int old_len = 0, new_len = 0;
        while (old_len < max_degree && old_nbrs[base + old_len] >= 0) ++old_len;
        while (new_len < max_degree && new_nbrs[base + new_len] >= 0) ++new_len;

        for (int j = 0; j < new_len; ++j) {
            const int new_nbr_id = new_nbrs[base + j];
            bool found = false;
            for (int k = 0; k < old_len; ++k) {
                if (old_nbrs[base + k] == new_nbr_id) {
                    found = true;
                    break;
                }
            }
            if (!found) h_innbrs[new_nbr_id].insert(innbr_id);
        }

        for (int j = 0; j < old_len; ++j) {
            const int old_nbr_id = old_nbrs[base + j];
            bool found = false;
            for (int k = 0; k < new_len; ++k) {
                if (new_nbrs[base + k] == old_nbr_id) {
                    found = true;
                    break;
                }
            }
            if (!found) h_innbrs[old_nbr_id].erase(innbr_id);
        }
    }
}

void Graph::insert_nodes(const float* h_ins_vecs, std::vector<int>& h_ins_ids_out, int ins_num) {
    assert(gtype == GraphType::GPU);
    // TODO: check expand need
    std::unique_lock<std::shared_mutex> lock(slot_mutex);
    const int start_idx = new_node_id;
    for (int i = 0; i < ins_num; i++) {
        const int ins_id = h_ins_ids_out[i] = new_node_id++;
        cpu_slots[ins_id].node_id = ins_id;
        cpu_slots[ins_id].deg = 0;
    }
    total_node_size = new_node_id;

    CUDA_CHECK(cudaMemcpy(gpu_slots + start_idx, &(cpu_slots[start_idx]), ins_num * sizeof(Slot),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_vec_slots + start_idx * dim, h_ins_vecs,
                          ins_num * sizeof(float) * dim, cudaMemcpyHostToDevice));

    // float time = std::chrono::duration_cast<std::chrono::milliseconds>(
    //                  std::chrono::high_resolution_clock::now() - s)
    //                  .count() /
    //              1000.0;
    fo.print("Insert " + TOS(ins_num) + " nodes from " + TOS(start_idx) + " to " +
             TOS(new_node_id - 1));
}

// flush_all: writeback all DIRTY slots to host synchronously
void Graph::flush_all_to_disk() {
    fo.iprint("Flush_all_to_disk...");
    std::unique_lock<std::shared_mutex> lock(slot_mutex);
    write_back_all_gpu_slots();
    write_back_all_cpu_slots();
    fo.print("Flush_all_to_disk finished.");
}

std::vector<int> Graph::get_search_start_nodes_fps(int M, int reset) {
    std::vector<int> res;
    if (reset == 1 || start_ids.size() < M) start_ids.clear();

    if (gtype == GraphType::GPU)
        FPS(res, M);
    else
        FPS_cpu(res, M);

    std::string str;
    for (int id : res) str += TOS(id) + " ";
    fo.iprint("Get top " + TOS(res.size()) + " access nodes: " + str);

    if (reset == 1) start_ids = res;

    return res;
}

void Graph::save_search_start_nodes() {
    std::ofstream fout(search_start_ids_path);
    if (!fout.is_open())
        fo.eprint("save_top_M_access_nodes Open file failed: " + search_start_ids_path.string());

    const int M = start_ids.size();
    fout << M << " ";
    for (int id : start_ids) fout << id << " ";
    fout.flush();
    fout.close();
    fo.iprint("Save top " + TOS(M) + " access nodes to: " + search_start_ids_path.string());
}

void Graph::load_search_start_nodes() {
    std::ifstream fin(search_start_ids_path);
    int M = -1;
    if (!fin.is_open()) {
        fo.print("get_top_M_access_nodes Open file failed: " + search_start_ids_path.string());
        return;
    }
    fin >> M;
    if (M > 0) {
        start_ids.resize(M);
        for (int i = 0; i < M; ++i) {
            fin >> start_ids[i];
            if (fin.eof()) {
                M = 0;
                start_ids.clear();
                break;
            }
        }
    }
    fin.close();

    fo.iprint("Load top " + TOS(M) + " access nodes from: " + search_start_ids_path.string());
}

};  // namespace efanna2e
