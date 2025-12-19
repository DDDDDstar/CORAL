#include <cfloat>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>

#include "disk_graph.h"
#include "fileout.h"

namespace efanna2e {
// -------------------- 工具函数 --------------------
static inline bool file_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

static inline uint64_t file_size(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return 0;
    return (uint64_t)st.st_size;
}

// 从 HostSlot 构造原始序列化 bytes（未压缩）
// 序列化格式： [deg (4B)] [nbrs (deg * 4B)] [dists (deg * 4B)]
static std::vector<char> serialize_slot_uncompressed(const HostSlot& s) {
    std::vector<char> out(4 + s.deg * 8);
    memcpy(out.data(), &s.deg, 4);
    memcpy(out.data() + 4, s.nbrs, 4 * s.deg);
    memcpy(out.data() + 4 + 4 * s.deg, s.dists, 4 * s.deg);
    return out;
}

// 反序列化
static void deserialize_slot_uncompressed(const char* buf, HostSlot& s) {
    int deg = 0;
    memcpy(&deg, buf, 4);
    s.deg = deg;
    memcpy(s.nbrs, buf + 4, 4 * deg);
    memcpy(s.dists, buf + 4 + 4 * deg, 4 * deg);
}

// 读取 node（邻居部分）。vec 读取独立（固定偏移）
void DiskStorage::load_node_slot(int node_id, HostSlot& slot_out) {
    assert(use);
    DiskPartition* part = get_partition(node_id);
    if (!part) fo.eprint("DiskStorage::load_node_slot: no partition for node " + TOS(node_id));
    part->load_node(node_id, slot_out);
}

void DiskStorage::load_node_vec(int node_id, VecSlot& vec_slot_out) {  // 读取向量（fixed-size）
    assert(use);
    DiskPartition* part = get_partition(node_id);
    if (!part) fo.eprint("DiskStorage::load_node_vec: no partition for node " + TOS(node_id));
    int vec_fd = open(base_vec_file.c_str(), O_RDONLY);
    if (vec_fd < 0) fo.eprint("DiskStorage::load_node_vec: open vectors for read");

    uint64_t off = part->vector_offset(node_id);
    float vec_out[dim];
    ssize_t r = pread(vec_fd, vec_out, dim * sizeof(float), off);
    close(vec_fd);
    if (r != (ssize_t)(dim * sizeof(float))) fo.eprint("vec read fail node " + TOS(node_id));
    vec_slot_out.copy(node_id, vec_out);
}

// 将向量写回（覆盖），位置固定（其实没用）
// bool DiskStorage::write_vector(int node_id, const float* vec) {
//     int vec_fd = open(base_vec_file.c_str(), O_RDWR);
//     if (vec_fd < 0) {
//         perror("open vec for write");
//         return false;
//     }
//     DiskPartition* part = get_partition(node_id);
//     if (!part) {
//         close(vec_fd);
//         return false;
//     }
//     uint64_t off = part->vector_offset(node_id);
//     ssize_t w = pwrite(vec_fd, vec, dim * sizeof(float), off);
//     fsync(vec_fd);
//     close(vec_fd);
//     return (w == (ssize_t)(dim * sizeof(float)));
// }

// append 写 node（邻居部分）到对应 partition
void DiskStorage::write_node_append(int node_id, const HostSlot& slot, bool want_flush = false) {
    DiskPartition* part = get_partition(node_id);
    if (!part) fo.eprint("write_node_append: no partition for node " + TOS(node_id));
    part->write_node_append(node_id, slot, want_flush);
}

void DiskStorage::write_batch_id(int batch_id) {
    std::ofstream bis(batch_id_path);
    bis << batch_id;
    bis.close();
}

int DiskStorage::read_batch_id() {
    std::ifstream bis(batch_id_path);
    std::stringstream buffer;
    buffer << bis.rdbuf();
    const int batch_id = std::stoi(buffer.str());
    bis.close();
    return batch_id;
}

// checkpoint: 持久化所有 partition 的索引
void DiskStorage::checkpoint_all() {
    for (auto& p : partitions) p->checkpoint();
}

// 启动/停止后台 compaction（由 DiskPartition 自己在构造中启动）
// ---- Prefetch API ----
// 预取：把一批节点读取到 pinned buffer（若启用 CUDA），或普通内存（同步）
// 这里我们提供一个简单的同步 prefetch (blocking)，并提供异步队列供后台线程处理
// blocking prefetch:
void DiskStorage::load_batch_slots(const std::vector<int>& node_ids, HostSlot* out_slots) {
    for (size_t i = 0; i < node_ids.size(); ++i) load_node_slot(node_ids[i], out_slots[i]);
}
// void DiskStorage::prefetch_and_wait_vec(const std::vector<int>& node_ids,
//                                         std::vector<float>& out_vecs) {
//     out_vecs.resize((size_t)node_ids.size() * (size_t)dim);
//     for (size_t i = 0; i < node_ids.size(); ++i)
//         load_node_vec(node_ids[i], out_vecs.data() + i * dim);
// }

// 异步预取：把任务 push 到队列，worker 会填充缓存并调用回调
// 回调签名： void callback(bool success, vector<HostSlot> slots, vector<float> vecs)
void DiskStorage::prefetch_slot_async(const std::vector<int>& node_ids,
                                      std::function<void(bool, std::vector<HostSlot>)> callback) {
    std::unique_lock<std::mutex> lk(prefetch_slot_mutex);
    prefetch_slot_queue.emplace_back(node_ids, callback);
    prefetch_slot_cv.notify_one();
}

void DiskStorage::prefetch_vec_async(const std::vector<int>& node_ids,
                                     std::function<void(bool, std::vector<float>)> callback) {
    std::unique_lock<std::mutex> lk(prefetch_vec_mutex);
    prefetch_vec_queue.emplace_back(node_ids, callback);
    prefetch_vec_cv.notify_one();
}

// void DiskStorage::slot_prefetch_worker_loop() {
//     while (true) {
//         std::pair<std::vector<int>, std::function<void(bool, std::vector<HostSlot>)>> job;
//         {
//             std::unique_lock<std::mutex> lk(prefetch_slot_mutex);
//             prefetch_slot_cv.wait(lk, [&] { return !prefetch_slot_queue.empty() || stop_all; });
//             if (stop_all && prefetch_slot_queue.empty()) return;
//             job = move(prefetch_slot_queue.front());
//             prefetch_slot_queue.erase(prefetch_slot_queue.begin());
//         }
//         std::vector<int> node_ids = move(job.first);
//         auto cb = job.second;
//         std::vector<HostSlot> slots(node_ids.size());
//         bool ok = true;
//         for (size_t i = 0; i < node_ids.size(); ++i)
//             if (!load_node_slot(node_ids[i], slots[i])) {
//                 ok = false;
//                 break;
//             }

//         cb(ok, move(slots));  // 回调（注意：回调在后台线程中被执行）
//     }
// }

// void DiskStorage::vec_prefetch_worker_loop() {
//     while (true) {
//         std::pair<std::vector<int>, std::function<void(bool, std::vector<float>)>> job;
//         {
//             std::unique_lock<std::mutex> lk(prefetch_vec_mutex);
//             prefetch_vec_cv.wait(lk, [&] { return !prefetch_vec_queue.empty() || stop_all; });
//             if (stop_all && prefetch_vec_queue.empty()) return;
//             job = move(prefetch_vec_queue.front());
//             prefetch_vec_queue.erase(prefetch_vec_queue.begin());
//         }
//         std::vector<int> node_ids = move(job.first);
//         auto cb = job.second;
//         std::vector<float> vecs((size_t)node_ids.size() * (size_t)dim);
//         bool ok = true;
//         for (size_t i = 0; i < node_ids.size(); ++i)
//             if (!load_node_vec(node_ids[i], vecs.data() + i * dim)) {
//                 ok = false;
//                 break;
//             }

//         cb(ok, move(vecs));  // 回调（注意：回调在后台线程中被执行）
//     }
// }

DiskStorage::DiskStorage(const std::string& base_file, const std::string& graph_file, int node_num,
                         int dim, int max_degree, bool construct)
    : base_vec_file(base_file),
      base_path(graph_file),
      batch_id_path(base_path + "/graph.batch_id"),
      node_num(node_num),
      node_offsets(node_num),
      dim(dim),
      max_degree(max_degree),
      stop_all(false),
      construct(construct),
      use(true) {
    partition_count = (node_num + CPU_N - 1) / CPU_N;
    nodes_per_partition = (node_num + partition_count - 1) / partition_count;
    // 初始化 partitions
    partitions.reserve(partition_count);
    for (int p = 0; p < partition_count; ++p) {
        int start = p * nodes_per_partition,
            cnt = nodes_per_partition <= node_num - start ? nodes_per_partition : node_num - start;
        if (cnt < 0) cnt = 0;
        partitions.emplace_back(new DiskPartition(base_path, p, start, cnt, dim, construct));
    }
    // 启动 checkpoint 线程（定期持久化 index）
    checkpoint_thread = std::thread(&DiskStorage::checkpoint_loop, this);

    // for (size_t i = 0; i < PREFETCH_THREADPOOL_SIZE; ++i)  // 初始化 prefetch thread pool
    // {
    //     prefetch_slot_threads.emplace_back(&DiskStorage::slot_prefetch_worker_loop, this);
    //     prefetch_vec_threads.emplace_back(&DiskStorage::vec_prefetch_worker_loop, this);
    // }
}

DiskStorage::~DiskStorage() {
    stop_all = true;
    {  // 停止 prefetch
        std::unique_lock<std::mutex> lk(prefetch_slot_mutex);
        prefetch_slot_cv.notify_all();
    }
    {  // 停止 prefetch
        std::unique_lock<std::mutex> lk(prefetch_vec_mutex);
        prefetch_vec_cv.notify_all();
    }
    for (auto& t : prefetch_slot_threads)
        if (t.joinable()) t.join();
    for (auto& t : prefetch_vec_threads)
        if (t.joinable()) t.join();

    // 停止 checkpoint
    if (checkpoint_thread.joinable()) checkpoint_thread.join();

    for (auto& p : partitions) delete p;  // partitions will be destroyed automatically
}

// -------------------- DiskPartition 管理（单分区） --------------------
// 每个 partition 管理自己的一组 node id，索引文件和数据文件。
// 优点：分区可以并行 compaction，内存索引也更可控。
// checkpoint：把内存 index 写回 idx_path 文件（覆盖）

void DiskPartition::persist_index() {
    std::unique_lock<std::mutex> lk(part_mutex);
    // 以临时文件写入，再替换，保证原子性
    std::string tmp = idx_path + ".tmp";
    int fd = open(tmp.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    // 写索引表
    size_t need = index_table.size() * sizeof(NodeDiskIndex);
    write(fd, index_table.data(), need);
    fsync(fd);
    close(fd);
    // 原子替换
    rename(tmp.c_str(), idx_path.c_str());
}

// 读取索引文件到内存
void DiskPartition::load_index() {
    int fd = open(idx_path.c_str(), O_RDONLY);
    index_table.resize(node_count);
    ssize_t r = read(fd, index_table.data(), file_size(idx_path));
    (void)r;
    close(fd);
}

// 强制触发 compaction（同步执行）
void DiskPartition::compact_once() {
    std::unique_lock<std::mutex> lk(part_mutex);
    // 创建临时 new data 文件
    std::string new_data = data_path + ".compact";
    int new_fd = open(new_data.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    uint64_t write_off = 0;
    // new index buffer
    std::vector<NodeDiskIndex> new_index(node_count);
    // 遍历当前 index，读取每个条目并写入 new_data（如果有数据）
    for (int i = 0; i < node_count; ++i) {
        NodeDiskIndex& e = index_table[i];
        if (e.size == 0) {
            new_index[i] = NodeDiskIndex{0, 0};
            continue;
        }
        // 读取原数据（先尝试 data_fd，再 append_fd）
        std::vector<char> comp(e.size);
        ssize_t r = pread(data_fd, comp.data(), e.size, e.offset);
        if (r != (ssize_t)e.size) {
            r = pread(append_fd, comp.data(), e.size, e.offset);
            if (r != (ssize_t)e.size) {
                // 数据丢失或索引指向无效块，跳过（视情况决定）
                fo.eprint("compact: missing block for local:" + TOS(i));
                new_index[i] = NodeDiskIndex{0, 0};
                continue;
            }
        }
        // 写入 new_fd
        ssize_t w = write(new_fd, comp.data(), e.size);
        if (w != (ssize_t)e.size) {
            close(new_fd);
            unlink(new_data.c_str());
            fo.eprint("compact write error");
            return;
        }
        new_index[i] = NodeDiskIndex{write_off, e.size};
        write_off += e.size;
    }
    fsync(new_fd);
    close(new_fd);

    // 备份旧文件，替换
    std::string old = data_path + ".old";
    rename(data_path.c_str(), old.c_str());
    rename(new_data.c_str(), data_path.c_str());
    // reopen data_fd on new file
    close(data_fd);
    data_fd = open(data_path.c_str(), O_RDWR);
    if (data_fd < 0) fo.eprint("reopen data after compact");
    // 清空 append file（所有旧 append 块都移动到 new data）
    close(append_fd);
    append_fd = open(append_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    index_table.swap(new_index);  // swap index_table
    persist_index();              // persist new index
    unlink(old.c_str());          // 删除 old 文件
    fo.print("partition " + TOS(partition_id) + " compaction complete");
}

// 背景线程循环
void DiskPartition::compaction_loop() {
    while (!stop_background) {
        std::this_thread::sleep_for(std::chrono::seconds(CHECKPOINT_INTERVAL_SECONDS));
        // heuristics: if append file size > threshold, trigger compaction
        uint64_t append_sz = file_size(append_path);
        if (append_sz > (uint64_t)64 * 1024 * 1024)  // 举例：64MB
            try {
                compact_once();
            } catch (...) {
                fo.eprint("compaction error partition: " + TOS(partition_id));
            }
    }
}

DiskPartition::DiskPartition(const std::string& base_path, int pid, int partition_start_id,
                             int partition_node_count, int dim, bool construct)
    : base_path(base_path),
      partition_id(pid),
      start_id(partition_start_id),
      node_count(partition_node_count),
      dim(dim),
      vec_size_bytes(dim * sizeof(float)),
      stop_background(false),
      construct(construct) {
    // 文件路径
    data_path = base_path + "/graph.part" + TOS(pid) + ".data";
    idx_path = base_path + "/graph.part" + TOS(pid) + ".idx";
    append_path = base_path + "/graph.part" + TOS(pid) + ".append";

    // 打开或创建索引（node_count entries）
    if (construct) {
        index_table.assign(node_count, NodeDiskIndex{0, 0});  // 初始化空索引
        persist_index();                                      // 写回索引文件（一次性 allocate）
    } else {
        load_index();
    }

    // 打开数据文件以供 append / read
    data_fd = open(data_path.c_str(), O_RDWR | O_CREAT, 0644);
    append_fd = open(append_path.c_str(), O_RDWR | O_CREAT, 0644);

    // 启动 background compaction 线程（按 partition）
    compaction_thread = std::thread(&DiskPartition::compaction_loop, this);
}

DiskPartition::~DiskPartition() {
    stop_background = true;  // 停止后台线程
    if (compaction_thread.joinable()) compaction_thread.join();
    if (data_fd >= 0) close(data_fd);
    if (append_fd >= 0) close(append_fd);
    persist_index();  // persist 索引
}

// 同步读取某个节点的数据（邻居 + distances）
// 返回 true 成功，slot_out 会被填充；注意 vec 需要调用者自己读取 vectors.bin（fixed-size）
void DiskPartition::load_node(int global_node_id, HostSlot& slot_out) {
    std::unique_lock<std::mutex> lk(part_mutex);
    if (!node_in_partition(global_node_id)) fo.eprint("Node not in partition");
    int local_id = global_node_id - start_id;
    NodeDiskIndex& e = index_table[local_id];
    if (e.size == 0) {  // 没有数据（degree == 0）
        slot_out.node_id = global_node_id;
        slot_out.deg = 0;
    }
    // posix pread 读取压缩块
    std::vector<char> comp(e.size);
    ssize_t r = pread(data_fd, comp.data(), e.size, e.offset);
    if (r != (ssize_t)e.size) {
        // 可能在 append 区（append file）或者在 compaction 刚好切换，尝试从 append file 读取
        r = pread(append_fd, comp.data(), e.size, e.offset);
        if (r != (ssize_t)e.size)  // 读取失败
            fo.eprint("read block fail node " + TOS(global_node_id) + " offset " + TOS(e.offset) +
                      " size " + TOS(e.size));
    }
    // comp 布局： [ node_id(4) ][ uncompressed_size(4) ][ compressed bytes ... ]
    if (e.size < 8) fo.eprint("corrupted block node");
    int block_node_id = 0;
    uint32_t uncompressed_size = 0;
    memcpy(&block_node_id, comp.data(), 4);
    memcpy(&uncompressed_size, comp.data() + 4, 4);
    if (block_node_id != global_node_id) {
        // 可能在 append file，尝试 append
        // 这里为了鲁棒性我们仍然尝试继续解压
    }
    const char* comp_ptr = comp.data() + 8;
    int comp_len = (int)e.size - 8;

    std::vector<char> uncompressed(uncompressed_size);
    // if (comp_len > 0) {
    //     int dec = LZ4_decompress_safe(comp_ptr, uncompressed.data(), comp_len,
    //     uncompressed_size); if (dec < 0) fo.eprint("LZ4 decompress failed node " +
    //     TOS(global_node_id));

    // } else  // 未压缩：直接拷贝
    memcpy(uncompressed.data(), comp_ptr, uncompressed_size);

    // 反序列化到 HostSlot
    slot_out.node_id = global_node_id;
    deserialize_slot_uncompressed(uncompressed.data(), slot_out);
}

// 将节点数据 append 到 partition 的 append 文件末尾（并更新内存索引）。
// 返回 true 成功；如果 want_flush==true 会调用 fsync 来保证持久化（开销高）
void DiskPartition::write_node_append(int global_node_id, const HostSlot& slot,
                                      bool want_flush = false) {
    std::unique_lock<std::mutex> lk(part_mutex);
    if (!node_in_partition(global_node_id))
        fo.eprint("DiskPartition::write_node_append: Node not in partition");
    int local_id = global_node_id - start_id;

    std::vector<char> raw = serialize_slot_uncompressed(slot);  // 序列化（未压缩内容）
    int raw_sz = (int)raw.size();

    // 压缩:
    std::vector<char> comp;
    const char* comp_ptr = nullptr;
    int comp_len = 0, maxComp = 0;
    // int comp_len = 0, maxComp = LZ4_compressBound(raw_sz);
    comp.resize(maxComp + 8);  // 8 bytes for header
    // header: node_id (4B), uncompressed_size (4B)
    memcpy(comp.data(), &global_node_id, 4);
    memcpy(comp.data() + 4, &raw_sz, 4);
    int csz = -1;
    // int csz = LZ4_compress_default(raw.data(), comp.data() + 8, raw_sz, maxComp);
    if (csz <= 0) {  // 压缩失败，退回不压缩写法
        comp.resize(raw_sz + 8);
        memcpy(comp.data(), &global_node_id, 4);
        memcpy(comp.data() + 4, &raw_sz, 4);
        memcpy(comp.data() + 8, raw.data(), raw_sz);
        comp_len = raw_sz;
    } else {
        comp.resize(csz + 8);
        comp_len = csz;
    }

    // append 写（写到 append_fd 尾部）
    off_t off = lseek(append_fd, 0, SEEK_END);
    ssize_t w = write(append_fd, comp.data(), comp.size());
    // 更新内存索引：offset 在 append file 中相对 append file 的起始（为了兼容 read，我们保存
    // offset 绝对值为文件内偏移） 注意：index.offset 指向的是 append_fd
    // 中的偏移（读取操作会尝试从 data_fd，再尝试 append_fd）
    index_table[local_id].offset = (uint64_t)off;
    index_table[local_id].size = (uint32_t)comp.size();
    // 如果 append file 超过阈值则 fsync（或由后台线程 flush），此处可选择不立即 flush
    if (want_flush) fsync(append_fd);
}

};  // namespace efanna2e