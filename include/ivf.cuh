#pragma once

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/VectorTransform.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/invlists/InvertedLists.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "uni.h"
#include "utils.cuh"

namespace efanna2e {

class TopLTracker {
   public:
    struct Node {
        int idx;
        uint32_t val;
    };

    TopLTracker(const std::vector<uint32_t>& values, size_t L) : values_(values), L_(L) {
        if (L_ == 0) {
            throw std::invalid_argument("L must be > 0");
        }
        heap_.reserve(L_);
        pos_.reserve(L_ * 2);
    }
    void reset() {
        heap_.clear();
        pos_.clear();
        heap_.reserve(L_);
        pos_.reserve(L_ * 2);
        for (int i = 0; i < values_.size(); ++i) on_increase(i);
    }

    // 外部把 values[idx] 改大后，调用这个接口通知数据结构更新
    void on_increase(int idx) {
        auto it = pos_.find(idx);
        if (it != pos_.end()) {
            // 已在 topL 堆中，值变大后只可能需要向下调整
            size_t p = it->second;
            heap_[p].val = values_[idx];
            sift_down(p);
            return;
        }

        // 不在堆中，看看是否能进入 topL
        if (heap_.size() < L_) {
            push_new(idx);
            return;
        }

        // 只有超过当前 topL 最小值，才有资格进入
        if (values_[idx] > heap_[0].val) {
            pos_.erase(heap_[0].idx);
            heap_[0] = Node{idx, values_[idx]};
            pos_[idx] = 0;
            sift_down(0);
        }
    }

    // 返回当前 topL，按值从大到小排序
    void get_topL_desc(std::vector<int>& result, int M = 0) const {
        if (!M) M = L_;
        assert(M <= L_);
        result.resize(M);
        std::vector<Node> node_res = heap_;
        std::sort(node_res.begin(), node_res.end(), [](const Node& a, const Node& b) {
            if (a.val != b.val) return a.val > b.val;
            return a.idx < b.idx;
        });
        for (int i = 0; i < M; ++i) result[i] = node_res[i].idx;
    }

    size_t size() const { return heap_.size(); }

   private:
    const std::vector<uint32_t>& values_;
    size_t L_;
    std::vector<Node> heap_;               // 最小堆
    std::unordered_map<int, size_t> pos_;  // 仅记录堆内元素位置

    void push_new(int idx) {
        heap_.push_back(Node{idx, values_[idx]});
        pos_[idx] = heap_.size() - 1;
        sift_up(heap_.size() - 1);
    }

    static bool less_node(const Node& a, const Node& b) {
        if (a.val != b.val) return a.val < b.val;
        return a.idx < b.idx;
    }

    void swap_nodes(size_t i, size_t j) {
        std::swap(heap_[i], heap_[j]);
        pos_[heap_[i].idx] = i;
        pos_[heap_[j].idx] = j;
    }

    void sift_up(size_t i) {
        while (i > 0) {
            size_t p = (i - 1) / 2;
            if (!less_node(heap_[i], heap_[p])) break;
            swap_nodes(i, p);
            i = p;
        }
    }

    void sift_down(size_t i) {
        size_t n = heap_.size();
        while (true) {
            size_t l = i * 2 + 1;
            size_t r = i * 2 + 2;
            size_t smallest = i;

            if (l < n && less_node(heap_[l], heap_[smallest])) {
                smallest = l;
            }
            if (r < n && less_node(heap_[r], heap_[smallest])) {
                smallest = r;
            }
            if (smallest == i) break;

            swap_nodes(i, smallest);
            i = smallest;
        }
    }
};

class IVFBuilder {
   public:
    struct Config {
        Config(fpath file_dir, int N, int dim, int topk, const float* h_base = nullptr,
               fpath base_file = "")
            : storage(N <= CPU_N ? 0 : 1),
              h_base(h_base),
              input_path(base_file),
              index_out(file_dir / "ivf.index"),
              hits_out(file_dir / "hits.bin"),
              num_vectors(N),
              dim(dim),
              topk(topk),
              nlist(PC.ivf_nlist),
              max_points_per_centroid(PC.ivf_points_per_centroid),
              sample_size(std::min(nlist * max_points_per_centroid, 20000000)) {}

        int storage;  // 0 cpu, 1 disk
        const float* h_base = nullptr;
        fpath input_path;
        fpath index_out, hits_out;

        int64_t num_vectors = -1;
        int dim = -1;
        int64_t add_chunk_size = 10'000'000;

        int nlist = -1;

        // PQ 参数：storage==1 时使用
        int M = 20;
        int nbits = 8;

        // OPQ 参数：仅 storage==1 时有效
        bool opq = true;
        int opq_M = 20;
        int opq_out_dim = 200;
        int opq_niter = 50;
        bool opq_verbose = false;

        uint64_t seed = 1234;
        bool verbose = false;
        DIST_METRIC metric = DIST_METRIC::L2_;

        bool normalize_train_sample = false;

        int max_points_per_centroid = 256;
        int sample_size = -1;

        int nprobe = PC.ivf_nprobe, topk = 0;
    };

    struct CompactId {
        int pre, next;
        void init(int i, int max) {
            pre = i - 1;
            next = i + 1 >= max ? -1 : i + 1;
        }
    };

   public:
    IVFBuilder() = default;
    IVFBuilder(Config cfg, int start_node_num, std::string mode)
        : tracker_(hits, start_node_num), cfg_(std::move(cfg)), mode_(mode) {
        validateConfig();

        if (PC.knn_rebuild) {
            compact_n = cfg_.num_vectors;
            if (cfg_.storage == 0) {
                // base_compact.resize(compact_n * cfg_.dim);
                // std::memcpy(base_compact.data(), cfg_.h_base,
                //             compact_n * cfg_.dim * sizeof(float));
            } else {
                current_data_path_ = cfg_.input_path;
                compact_file_a_ = cfg_.index_out.parent_path() / "base_compact_a.f32";
                compact_file_b_ = cfg_.index_out.parent_path() / "base_compact_b.f32";
                current_data_is_original_ = true;
            }
            h_compact_ids.resize(compact_n);
            std::iota(h_compact_ids.begin(), h_compact_ids.end(), 0);
            // #pragma omp parallel for
            //             for (int i = 0; i < cfg_.num_vectors; i++) h_compact_ids[i].init(i,
            //             cfg_.num_vectors);
        }
        hits.resize(cfg_.num_vectors, 0);
    }

    ~IVFBuilder() {
        if (save_fut.valid()) save_fut.get();
        if (rebuild_fut.valid()) rebuild_fut.get();
        release_adc_cache_();
    }

    void rebuild() {
        // if (rebuild_fut.valid()) rebuild_fut.get();
        // fo.print("IVF rebuild...");

        // rebuild_fut = std::async(std::launch::async, [this] {
        auto s = now_time();

        const faiss::MetricType metric = faissMetric();

        if (save_fut.valid()) save_fut.get();

        size_t cnt = 0;
        if (cfg_.storage == 0) {
            int64_t chunk_rows = cfg_.add_chunk_size, get_rows = 0;
            std::vector<float> xbuf(chunk_rows * cfg_.dim);

            std::vector<int> compact_ids_new;
            compact_ids_new.reserve(compact_n);

            index_->reset();

            for (size_t i = 0; i < compact_n; i++) {
                const int id = h_compact_ids[i];
                if (hits[id] > PC.knn_max_hit) continue;

                compact_ids_new.push_back(id);

                std::memcpy(xbuf.data() + get_rows * cfg_.dim, cfg_.h_base + (size_t)id * cfg_.dim,
                            sizeof(float) * cfg_.dim);
                if (++get_rows == chunk_rows) {
                    add_chunk_with_gpu_assign(index_.get(), *coarse_assigner_, get_rows,
                                              xbuf.data());
                    get_rows = 0;
                }
            }
            if (get_rows)
                add_chunk_with_gpu_assign(index_.get(), *coarse_assigner_, get_rows, xbuf.data());
            compact_n = compact_ids_new.size();
            // {
            //     std::unique_lock<std::shared_mutex> lock(mutex_);
            h_compact_ids.swap(compact_ids_new);
            set_nprobe();
            // }
        } else {
            const fpath input_file = current_data_path_, output_file = next_compact_path();

            std::ifstream fin(input_file, std::ios::binary);
            if (!fin) fo.eprint("cannot open base file for add: " + cfg_.input_path.string());

            std::ofstream fout(output_file, std::ios::binary | std::ios::trunc);
            if (!fout)
                fo.eprint("rebuild(storage=1): cannot open output file: " + output_file.string());

            std::vector<float> xbuf(cfg_.add_chunk_size * (int64_t)cfg_.dim);

            // {
            //     std::unique_lock<std::shared_mutex> lock(mutex_);
            index_->reset();

            int64_t done = 0;
            for (size_t i = 0; i < compact_n; i++) {
                const int id = h_compact_ids[i];
                if (hits[id] <= PC.knn_max_hit) {
                    const size_t row_in_input = current_data_is_original_ ? (size_t)id : i;
                    fin.seekg(row_in_input * cfg_.dim * sizeof(float), std::ios::beg);
                    if (!fin) fo.eprint("rebuild seek failed: id=" + TOS(id));

                    fin.read(reinterpret_cast<char*>(xbuf.data() + done * cfg_.dim),
                             sizeof(float) * cfg_.dim);
                    if (!fin) fo.eprint("rebuild read failed: id=" + TOS(id));

                    h_compact_ids[cnt++] = id;
                    if (++done == cfg_.add_chunk_size) {
                        add_chunk_with_gpu_assign(index_.get(), *coarse_assigner_, done,
                                                  xbuf.data());
                        fout.write(reinterpret_cast<const char*>(xbuf.data()),
                                   done * cfg_.dim * sizeof(float));
                        if (!fout)
                            fo.eprint("rebuild(storage=1): failed writing next compact file");
                        done = 0;
                    }
                }
            }
            if (done) {
                add_chunk_with_gpu_assign(index_.get(), *coarse_assigner_, done, xbuf.data());
                fout.write(reinterpret_cast<const char*>(xbuf.data()),
                           done * cfg_.dim * sizeof(float));
                if (!fout) fo.eprint("rebuild(storage=1): failed writing next compact file");
            }
            compact_n = cnt;
            current_data_path_ = output_file;
            current_data_is_original_ = false;
            set_nprobe();
            // }
            fin.close();
            fout.close();

            auto* ivf_sub = get_ivf_subindex(index_.get());
            auto* ivfpq_sub = dynamic_cast<faiss::IndexIVFPQ*>(ivf_sub);
            if (!ivfpq_sub) fo.eprint("rebuild(storage=1): subindex is not IVFPQ");

            refresh_adc_runtime_cache_storage1_();
        }

        fo.print("IVF rebuild finished(" + TOS(time_diff(s)) + "s): comapct_n=" + TOS(compact_n));
        // });
    }

    void prepare() {
        auto s = now_time();
        float t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;

        reservoirSample(sample_);

        auto s1 = now_time();
        t1 = time_diff(s, s1);

        const int index_dim = effectiveDim();
        const faiss::MetricType metric = faissMetric();

        if (cfg_.storage == 0) {
            std::unique_ptr<faiss::Index> quantizer = makeCpuFlatIndex(index_dim);

            auto ivf = std::make_unique<faiss::IndexIVFFlat>(
                dynamic_cast<faiss::IndexFlat*>(quantizer.get()), index_dim, cfg_.nlist, metric);

            ivf->own_fields = true;
            quantizer.release();

            ivf = gpu_train_and_return(std::move(ivf), cfg_.sample_size, sample_.data());
            if (!ivf->is_trained) fo.eprint("IVFFlat GPU training failed");
            fo.print("IVFFlat train finished (GPU-assisted).");
            auto s2 = now_time();
            t2 = time_diff(s1, s2);

            index_ = std::move(ivf);
            set_nprobe();
            build_or_refresh_coarse_assigner();
            add_matrix_with_gpu_assign(index_.get(), *coarse_assigner_, cfg_.h_base,
                                       cfg_.num_vectors, cfg_.add_chunk_size);
            auto s3 = now_time();
            t3 = time_diff(s2, s3);
            fo.print("IVFFlat add finished.");

        } else {
            if (cfg_.opq) {
                // 1) 先创建 OPQ + IVFPQ 复合索引
                const int pq_dim = (cfg_.opq_out_dim <= 0 ? cfg_.dim : cfg_.opq_out_dim);
                std::unique_ptr<faiss::Index> quantizer = makeCpuFlatIndex(pq_dim);
                auto ivfpq = std::make_unique<faiss::IndexIVFPQ>(
                    dynamic_cast<faiss::IndexFlat*>(quantizer.get()), pq_dim, cfg_.nlist, cfg_.M,
                    cfg_.nbits, metric);
                ivfpq->own_fields = true;
                ivfpq->nprobe = cfg_.nprobe;
                ivfpq->verbose = cfg_.verbose;
                quantizer.release();

                auto opq = std::make_unique<faiss::OPQMatrix>(cfg_.dim, cfg_.opq_M, pq_dim);
                opq->niter = cfg_.opq_niter;
                opq->verbose = cfg_.opq_verbose || cfg_.verbose;

                auto pre =
                    std::make_unique<faiss::IndexPreTransform>(opq.release(), ivfpq.release());

                // 2) 直接训练整个 IndexPreTransform
                auto trained_pre =
                    trainCompositeIndex_(std::move(pre), cfg_.sample_size, sample_.data());

                if (!trained_pre || !trained_pre->is_trained)
                    fo.eprint("IVFOPQ composite training failed");

                fo.print("IVFOPQ train finished (directly training whole IndexPreTransform).");

                auto s2 = now_time();
                t2 = time_diff(s1, s2);

                // 3) 训练完成后保存到 index_
                index_ = std::move(trained_pre);
                set_nprobe();
                build_or_refresh_coarse_assigner();

                // 4) 分块 add 原始向量；IndexPreTransform add 路径里会自动做 OPQ
                std::ifstream fin(cfg_.input_path, std::ios::binary);
                if (!fin) fo.eprint("cannot open base file for add: " + cfg_.input_path.string());

                std::vector<float> xbuf(cfg_.add_chunk_size * (int64_t)cfg_.dim);

                int64_t done = 0;
                while (done < cfg_.num_vectors) {
                    int64_t rows = std::min<int64_t>(cfg_.add_chunk_size, cfg_.num_vectors - done);
                    fin.read(reinterpret_cast<char*>(xbuf.data()),
                             sizeof(float) * rows * cfg_.dim);
                    if (!fin) fo.eprint("failed reading base chunk");

                    add_chunk_with_gpu_assign(index_.get(), *coarse_assigner_, rows, xbuf.data());
                    done += rows;

                    if (cfg_.verbose) {
                        fo.print("add progress: " + TOS(done) + "/" + TOS(cfg_.num_vectors) +
                                 ", ntotal=" + TOS(index_->ntotal));
                    }
                }

                fo.print("IVFOPQ add finished.");

                auto s3 = now_time();
                t3 = time_diff(s2, s3);

            } else {
                // -------------------------------
                // storage == 1 && no OPQ
                // -------------------------------
                std::unique_ptr<faiss::Index> quantizer = makeCpuFlatIndex(index_dim);
                auto ivfpq = std::make_unique<faiss::IndexIVFPQ>(
                    dynamic_cast<faiss::IndexFlat*>(quantizer.get()), index_dim, cfg_.nlist,
                    cfg_.M, cfg_.nbits, metric);
                ivfpq->own_fields = true;
                ivfpq->nprobe = cfg_.nprobe;
                ivfpq->verbose = cfg_.verbose;
                quantizer.release();

                ivfpq = gpu_train_and_return(std::move(ivfpq), cfg_.sample_size, sample_.data());
                if (!ivfpq->is_trained) fo.eprint("IVFPQ training failed");
                fo.print("IVFPQ train finished.");

                auto s2 = now_time();
                t2 = time_diff(s1, s2);

                index_ = std::move(ivfpq);
                set_nprobe();
                build_or_refresh_coarse_assigner();

                std::ifstream fin(cfg_.input_path, std::ios::binary);
                if (!fin) fo.eprint("cannot open base file for add: " + cfg_.input_path.string());

                std::vector<float> xbuf(cfg_.add_chunk_size * (int64_t)cfg_.dim);

                int64_t done = 0;
                while (done < cfg_.num_vectors) {
                    int64_t rows = std::min<int64_t>(cfg_.add_chunk_size, cfg_.num_vectors - done);
                    fin.read(reinterpret_cast<char*>(xbuf.data()),
                             sizeof(float) * rows * cfg_.dim);
                    if (!fin) fo.eprint("failed reading base chunk");

                    add_chunk_with_gpu_assign(index_.get(), *coarse_assigner_, rows, xbuf.data());
                    done += rows;

                    if (cfg_.verbose) {
                        fo.print("add progress: " + TOS(done) + "/" + TOS(cfg_.num_vectors) +
                                 ", ntotal=" + TOS(index_->ntotal));
                    }
                }

                fo.print("IVFPQ add finished.");

                auto s3 = now_time();
                t3 = time_diff(s2, s3);
            }

            if (mode_ == "search") {
                auto* ivf_sub = get_ivf_subindex(index_.get());
                auto* ivfpq_sub = dynamic_cast<faiss::IndexIVFPQ*>(ivf_sub);
                if (!ivfpq_sub) fo.eprint("prepare(storage=1): subindex is not IVFPQ");
                refresh_adc_runtime_cache_storage1_();
            }
        }

        // save_fut = std::async(std::launch::async, [this]() {
        faiss::write_index(index_.get(), cfg_.index_out.c_str());
        fo.print("IVF saved to " + cfg_.index_out.string());
        // });

        fo.print("IVF build finished(" + TOS(t1) + "s, " + TOS(t2) + "s, " + TOS(t3) + "s).");
        already_build_ = true;
    }

    bool load() {
        try {
            fo.print("IVF loaded from " + cfg_.index_out.string());
            std::unique_ptr<faiss::Index> base(faiss::read_index(cfg_.index_out.c_str()));
            if (!base) {
                fo.print("Failed to load ivf index from " + cfg_.index_out.string());
                return false;
            }
            index_ = std::move(base);
            bool ok = set_nprobe();
            if (ok) {
                build_or_refresh_coarse_assigner();
                if (cfg_.storage == 1 && mode_ == "search") {
                    auto* ivf_sub = get_ivf_subindex(index_.get());
                    auto* ivfpq_sub = dynamic_cast<faiss::IndexIVFPQ*>(ivf_sub);
                    if (!ivfpq_sub) fo.eprint("load(storage=1): subindex is not IVFPQ");
                    refresh_adc_runtime_cache_storage1_();
                }
                already_build_ = true;
            }
            fo.print("IVF loaded success!");
            return ok;
        } catch (const std::exception& e) {
            fo.print(std::string(e.what()));
            return false;
        }
    }

    int index_size() { return index_->ntotal; }

    const int* search(const float* queries, int nq, bool record_hit, float* iso_ratio = nullptr) {
        static std::vector<faiss::idx_t> I;
        static std::vector<int> ids_res;
        static std::vector<float> D;
        static size_t last_cnt = 0;

        const int k = cfg_.topk;
        const int total = nq * k;

        if ((int)I.size() < total) {
            I.resize(total);
            ids_res.resize(total);
            D.resize(total);
        }

        // {
        // std::shared_lock<std::shared_mutex> lock(mutex_);
        index_->search(nq, queries, k, D.data(), I.data());

        int iso_num = 0;
        if (PC.knn_rebuild) {
            for (int i = 0; i < total; ++i) {
                auto idx = I[i];
                if (idx < 0) {
                    ids_res[i] = -1;
                    continue;
                }
                // assert(idx >= 0);
                // if (idx < 0) fo.eprint("ivf search error: idx < 0, N=" +
                // TOS(index_->ntotal));

                const int id = h_compact_ids[idx];
                ids_res[i] = id;

                if (record_hit) {
                    if (!hits[id])
                        iso_num++;
                    else if (hits[id] == PC.knn_max_hit)
                        over_cnt++;
                    hits[id]++;
                    tracker_.on_increase(id);
                }
            }

            if (record_hit) {
                *iso_ratio = (float)iso_num / (nq * k);
                if ((*iso_ratio) < PC.knn_rebuild_thres &&
                    over_cnt - last_cnt > PC.knn_rebuild_min_cnt) {
                    rebuild();
                    last_cnt = over_cnt;
                }
            }

        } else {
            for (int i = 0; i < total; ++i) {
                const int id = (int)I[i];
                if (record_hit) {
                    if (!hits[id]) iso_num++;
                    hits[id]++;
                    tracker_.on_increase(id);
                }
                ids_res[i] = id;
            }
            if (record_hit) *iso_ratio = (float)iso_num / (nq * k);
        }
        // }

        return ids_res.data();
    }

    inline size_t get_over_cnt() { return over_cnt.load(); }

    void get_top_hit_nodes(std::vector<int>& nodes, int topn) {
        fo.print("ivf get_top_hit_nodes...");
        if (topn == cfg_.num_vectors) {
            nodes.resize(cfg_.num_vectors);
            std::iota(nodes.begin(), nodes.end(), 0);
            return;
        }
        if (tracker_.size() < topn) {
            std::vector<int> idx(cfg_.num_vectors);
            std::iota(idx.begin(), idx.end(), 0);
            std::nth_element(idx.begin(), idx.begin() + topn, idx.end(),
                             [&](int a, int b) { return hits[a] > hits[b]; });
            idx.resize(topn);
            nodes.swap(idx);
        } else
            tracker_.get_topL_desc(nodes, topn);
        // std::string str;
        // for (int id : nodes) str += TOS(id) + "-" + TOS(hits[id]) + " ";
        // fo.print("ivf get_top_hit_nodes: " + str);
    }

    void save_hits() {
        std::ofstream fhit(cfg_.hits_out, std::ios::binary);
        fhit.write((char*)(&cfg_.num_vectors), sizeof(int64_t));
        fhit.write((char*)hits.data(), sizeof(uint32_t) * cfg_.num_vectors);
        fo.print("ivf_builder save_hits finished.");
        // return welford_variance(hits);
    }

    void load_hits() {
        fo.print("ivf load hits from " + cfg_.hits_out.string());
        std::ifstream fhit(cfg_.hits_out, std::ios::binary);
        int64_t N;
        fhit.read((char*)(&N), sizeof(int64_t));
        if (N < cfg_.num_vectors) fo.eprint("load_hits: N < num_vectors");
        hits.resize(cfg_.num_vectors);
        fhit.read((char*)hits.data(), sizeof(uint32_t) * cfg_.num_vectors);
        tracker_.reset();
    }

    // =========================================================
    // storage==1: 计算 nq 个 query 到各自 batch 个 base ids 的 ADC 距离
    //
    // 输入:
    //   queries : [nq, dim]
    //   base_ids: [nq * batch]
    //            第 q 个 query 对应 base_ids[q * batch + 0 ... q * batch + batch-1]
    //
    // 输出:
    //   out_dists: [nq * batch]
    //            out_dists[q * batch + b] = dist(query_q, base_ids[q,b])
    //
    // 注意:
    //   1) 不考虑 knn_rebuild，base_ids 直接视为索引内部 id
    //   2) 当前仅支持 nbits == 8
    // =========================================================
    // void compute_adc_dists_to_base_batch(const float* queries, int nq, const int* base_ids,
    //                                      int batch, float* d_out_dists, size_t lut_mem_mb =
    //                                      512);
    // void compute_adc_dists_to_base(const float* queries, int nq, const int* base_ids, int batch,
    //                                float* d_out_dists, size_t lut_mem_mb = 512);

    inline void change_mode_to_search() { mode_ = "search"; }

   private:
    struct GpuCoarseAssigner {
        std::unique_ptr<faiss::gpu::GpuIndexFlat> gpu_flat;
        std::vector<float> coarse_D;
        std::vector<faiss::idx_t> list_nos;

        GpuCoarseAssigner() = default;

        GpuCoarseAssigner(faiss::gpu::StandardGpuResources& res, const faiss::IndexIVF* ivf,
                          int device = 0) {
            if (!ivf) fo.eprint("GpuCoarseAssigner: ivf is null");
            if (!ivf->quantizer) fo.eprint("GpuCoarseAssigner: ivf->quantizer is null");

            std::vector<float> centroids((size_t)ivf->nlist * ivf->d);
            ivf->quantizer->reconstruct_n(0, ivf->nlist, centroids.data());

            faiss::gpu::GpuIndexFlatConfig gcfg;
            gcfg.device = device;

            if (ivf->metric_type == faiss::METRIC_L2) {
                gpu_flat = std::make_unique<faiss::gpu::GpuIndexFlatL2>(&res, ivf->d, gcfg);
            } else if (ivf->metric_type == faiss::METRIC_INNER_PRODUCT) {
                gpu_flat = std::make_unique<faiss::gpu::GpuIndexFlatIP>(&res, ivf->d, gcfg);
            } else {
                fo.eprint("GpuCoarseAssigner: unsupported metric");
            }

            gpu_flat->add(ivf->nlist, centroids.data());
        }

        const faiss::idx_t* assign(const float* x, faiss::idx_t n) {
            if (n <= 0) return nullptr;
            coarse_D.resize((size_t)n);
            list_nos.resize((size_t)n);
            gpu_flat->search(n, x, 1, coarse_D.data(), list_nos.data());
            return list_nos.data();
        }
    };

    TopLTracker tracker_;
    std::string mode_;
    Config cfg_;
    std::unique_ptr<faiss::Index> index_;
    std::unique_ptr<faiss::PCAMatrix> pca_;
    faiss::gpu::StandardGpuResources gpu_res;
    std::unique_ptr<GpuCoarseAssigner> coarse_assigner_;
    bool coarse_assigner_ready_ = false;

    std::vector<float> sample_;
    // std::vector<float> base_compact;
    // std::vector<CompactId> h_compact_ids;
    std::vector<int> h_compact_ids;
    std::vector<uint32_t> hits;
    size_t compact_n = 0;
    std::atomic<size_t> over_cnt = 0;

    std::future<void> save_fut, rebuild_fut;
    // std::shared_mutex mutex_;

    fpath current_data_path_, compact_file_a_, compact_file_b_;
    bool current_data_is_original_ = true;

    bool already_build_ = false;

    // -------- storage==1 / ADC runtime cache --------
    int pq_code_size_ = 0;
    int pq_M_ = 0;
    int pq_ksub_ = 0;
    int pq_dsub_ = 0;
    int pq_dim_ = 0;

    bool adc_ready_ = false;

    std::vector<float> h_pq_centroids_;
    std::vector<float> h_coarse_centroids_;

    float* d_pq_centroids_ = nullptr;
    float* d_coarse_centroids_ = nullptr;

    std::unique_ptr<faiss::IndexPreTransform> trainCompositeIndex_(
        std::unique_ptr<faiss::IndexPreTransform> cpu_index, faiss::idx_t n, const float* x) {
        if (!cpu_index) fo.eprint("trainCompositeIndex_: cpu_index is null");
        if (n <= 0) fo.eprint("trainCompositeIndex_: n must be > 0");

        faiss::gpu::GpuClonerOptions opts;
        opts.useFloat16 = false;

        faiss::Index* gpu_index =
            faiss::gpu::index_cpu_to_gpu(&gpu_res, 0, cpu_index.get(), &opts);
        if (!gpu_index) fo.eprint("trainCompositeIndex_: index_cpu_to_gpu failed");

        gpu_index->train(n, x);

        std::unique_ptr<faiss::Index> trained_cpu_base(faiss::gpu::index_gpu_to_cpu(gpu_index));
        delete gpu_index;

        if (!trained_cpu_base) fo.eprint("trainCompositeIndex_: index_gpu_to_cpu failed");

        auto* trained_typed = dynamic_cast<faiss::IndexPreTransform*>(trained_cpu_base.release());
        if (!trained_typed)
            fo.eprint("trainCompositeIndex_: type cast to IndexPreTransform failed");

        return std::unique_ptr<faiss::IndexPreTransform>(trained_typed);
    }

    inline size_t row_bytes() const { return (size_t)cfg_.dim * sizeof(float); }

    inline fpath next_compact_path() const {
        if (current_data_is_original_) return compact_file_a_;
        return (current_data_path_ == compact_file_a_) ? compact_file_b_ : compact_file_a_;
    }

    void release_adc_cache_() {
        if (d_pq_centroids_) cudaFree(d_pq_centroids_), d_pq_centroids_ = nullptr;
        if (d_coarse_centroids_) cudaFree(d_coarse_centroids_), d_coarse_centroids_ = nullptr;
        adc_ready_ = false;
    }

    void refresh_adc_runtime_cache_storage1_() {
        release_adc_cache_();

        auto* ivf_base = get_ivf_subindex(index_.get());
        auto* ivfpq = dynamic_cast<faiss::IndexIVFPQ*>(ivf_base);
        if (!ivfpq) fo.eprint("refresh_adc_runtime_cache_storage1_: subindex is not IndexIVFPQ");

        ivfpq->make_direct_map();

        pq_code_size_ = ivfpq->code_size;
        pq_M_ = ivfpq->pq.M;
        pq_ksub_ = ivfpq->pq.ksub;
        pq_dsub_ = ivfpq->pq.dsub;
        pq_dim_ = ivfpq->d;

        if (cfg_.nbits != 8 || pq_ksub_ != 256 || pq_code_size_ != pq_M_) {
            fo.eprint("ADC runtime currently only supports nbits=8 (one byte per sub-code)");
        }

        h_pq_centroids_.assign(ivfpq->pq.centroids.begin(), ivfpq->pq.centroids.end());

        h_coarse_centroids_.resize((size_t)ivfpq->nlist * pq_dim_);
        ivfpq->quantizer->reconstruct_n(0, ivfpq->nlist, h_coarse_centroids_.data());

        cudaMalloc(&d_pq_centroids_, sizeof(float) * h_pq_centroids_.size());
        cudaMemcpy(d_pq_centroids_, h_pq_centroids_.data(), sizeof(float) * h_pq_centroids_.size(),
                   cudaMemcpyHostToDevice);

        cudaMalloc(&d_coarse_centroids_, sizeof(float) * h_coarse_centroids_.size());
        cudaMemcpy(d_coarse_centroids_, h_coarse_centroids_.data(),
                   sizeof(float) * h_coarse_centroids_.size(), cudaMemcpyHostToDevice);

        adc_ready_ = true;
    }

    void gpu_train_cpu_index(faiss::Index* cpu_index, faiss::idx_t n, const float* x) {
        if (!cpu_index) fo.eprint("gpu_train_cpu_index: cpu_index is null");
        if (n <= 0) fo.eprint("gpu_train_cpu_index: n must be > 0");

        faiss::gpu::GpuClonerOptions opts;
        opts.useFloat16 = false;

        faiss::Index* gpu_index = faiss::gpu::index_cpu_to_gpu(&gpu_res, 0, cpu_index, &opts);
        if (!gpu_index) fo.eprint("gpu_train_cpu_index: index_cpu_to_gpu failed");

        gpu_index->train(n, x);

        std::unique_ptr<faiss::Index> trained_cpu(faiss::gpu::index_gpu_to_cpu(gpu_index));
        delete gpu_index;

        if (!trained_cpu) fo.eprint("gpu_train_cpu_index: index_gpu_to_cpu failed");

        fo.eprint("gpu_train_cpu_index should not be called directly on raw pointer");
    }

    template <typename IndexT>
    std::unique_ptr<IndexT> gpu_train_and_return(std::unique_ptr<IndexT> cpu_index, faiss::idx_t n,
                                                 const float* x) {
        if (!cpu_index) fo.eprint("gpu_train_and_return: cpu_index is null");
        if (n <= 0) fo.eprint("gpu_train_and_return: n must be > 0");

        faiss::gpu::GpuClonerOptions opts;
        opts.useFloat16 = false;

        faiss::Index* gpu_index =
            faiss::gpu::index_cpu_to_gpu(&gpu_res, 0, cpu_index.get(), &opts);
        if (!gpu_index) fo.eprint("gpu_train_and_return: index_cpu_to_gpu failed");

        gpu_index->train(n, x);

        std::unique_ptr<faiss::Index> trained_cpu_base(faiss::gpu::index_gpu_to_cpu(gpu_index));
        delete gpu_index;

        if (!trained_cpu_base) fo.eprint("gpu_train_and_return: index_gpu_to_cpu failed");

        IndexT* trained_typed = dynamic_cast<IndexT*>(trained_cpu_base.release());
        if (!trained_typed) fo.eprint("gpu_train_and_return: type cast failed");

        return std::unique_ptr<IndexT>(trained_typed);
    }

    std::vector<float> apply_transform_noalloc(const faiss::VectorTransform& vt, faiss::idx_t n,
                                               const float* x) const {
        std::vector<float> xt((size_t)n * vt.d_out);
        vt.apply_noalloc(n, x, xt.data());
        return xt;
    }

    void build_or_refresh_coarse_assigner() {
        auto* ivf = get_ivf_subindex(index_.get());
        if (!ivf) fo.eprint("build_or_refresh_coarse_assigner: IVF subindex not found");

        coarse_assigner_ = std::make_unique<GpuCoarseAssigner>(gpu_res, ivf, 0);
        coarse_assigner_ready_ = true;
        fo.print("build_or_refresh_coarse_assigner: coarse assigner ready.");
    }

    faiss::IndexIVF* get_ivf_subindex(faiss::Index* index) const {
        if (!index) return nullptr;
        if (auto* ivf = dynamic_cast<faiss::IndexIVF*>(index)) return ivf;
        if (auto* pt = dynamic_cast<faiss::IndexPreTransform*>(index))
            return dynamic_cast<faiss::IndexIVF*>(pt->index);
        return nullptr;
    }

    void add_chunk_with_gpu_assign(faiss::Index* top_index, GpuCoarseAssigner& assigner,
                                   faiss::idx_t n, const float* x) {
        if (n <= 0) return;
        if (!top_index) fo.eprint("add_chunk_with_gpu_assign: top_index is null");

        if (auto* pt = dynamic_cast<faiss::IndexPreTransform*>(top_index)) {
            auto* ivf = dynamic_cast<faiss::IndexIVF*>(pt->index);
            if (!ivf) fo.eprint("add_chunk_with_gpu_assign: pt->index is not IVF");
            const float* xt = pt->apply_chain(n, x);
            const faiss::idx_t* list_nos = assigner.assign(xt, n);
            ivf->add_core(n, xt, nullptr, list_nos);
            pt->ntotal = ivf->ntotal;
            if (xt != x) delete[] xt;
            return;
        }

        if (auto* ivf = dynamic_cast<faiss::IndexIVF*>(top_index)) {
            const faiss::idx_t* list_nos = assigner.assign(x, n);
            ivf->add_core(n, x, nullptr, list_nos);
            return;
        }

        fo.eprint("add_chunk_with_gpu_assign: unsupported index type");
    }

    void add_matrix_with_gpu_assign(faiss::Index* top_index, GpuCoarseAssigner& assigner,
                                    const float* x, faiss::idx_t n, faiss::idx_t chunk_rows) {
        if (!top_index) fo.eprint("add_matrix_with_gpu_assign: top_index is null");
        if (n <= 0) return;

        auto* ivf = get_ivf_subindex(top_index);
        if (!ivf) fo.eprint("add_matrix_with_gpu_assign: cannot find IVF subindex");

        for (faiss::idx_t i = 0; i < n; i += chunk_rows) {
            const faiss::idx_t rows = std::min<faiss::idx_t>(chunk_rows, n - i);
            add_chunk_with_gpu_assign(top_index, assigner, rows, x + (size_t)i * cfg_.dim);
            fo.print("add_matrix_with_gpu_assign: added " + TOS(i + rows) + " /" + TOS(n));
        }
    }

    void add_matrix_with_gpu_assign_by_ids(faiss::Index* top_index, GpuCoarseAssigner& assigner,
                                           const float* base,  // 原始数据
                                           const int* ids,     // 选择的 id
                                           faiss::idx_t n, faiss::idx_t chunk_rows) {
        std::vector<float> xbuf(chunk_rows * cfg_.dim);
        for (faiss::idx_t i = 0; i < n; i += chunk_rows) {
            const faiss::idx_t rows = std::min<faiss::idx_t>(chunk_rows, n - i);
            // gather
            for (faiss::idx_t j = 0; j < rows; j++)
                std::memcpy(xbuf.data() + (size_t)j * cfg_.dim,
                            base + (size_t)ids[i + j] * cfg_.dim, sizeof(float) * cfg_.dim);

            add_chunk_with_gpu_assign(top_index, assigner, rows, xbuf.data());
        }
    }

    int effectiveDim() const {
        if (cfg_.storage == 1 && cfg_.opq) {
            return (cfg_.opq_out_dim <= 0 ? cfg_.dim : cfg_.opq_out_dim);
        }
        return cfg_.dim;
    }

    bool set_nprobe() {
        if (auto* ivf = dynamic_cast<faiss::IndexIVF*>(index_.get())) {
            ivf->nprobe = cfg_.nprobe;
            return true;
        }

        if (auto* pt = dynamic_cast<faiss::IndexPreTransform*>(index_.get())) {
            auto* ivf = dynamic_cast<faiss::IndexIVF*>(pt->index);
            if (!ivf) return false;
            ivf->nprobe = cfg_.nprobe;
            return true;
        }

        return false;
    }

    void validateConfig() const {
        if (cfg_.storage == 0 && cfg_.h_base == nullptr) fo.eprint("h_base is null");
        if (cfg_.storage == 1 && cfg_.input_path.empty()) fo.eprint("input_path is empty");
        if (cfg_.num_vectors <= 0) fo.eprint("num_vectors must be > 0");
        if (cfg_.dim <= 0) fo.eprint("dim must be > 0");
        if (cfg_.sample_size <= 0 || cfg_.sample_size > cfg_.num_vectors)
            fo.eprint("invalid sample_size");
        if (cfg_.add_chunk_size <= 0) fo.eprint("invalid add_chunk_size");
        if (cfg_.nlist <= 1 || cfg_.nlist > cfg_.sample_size) fo.eprint("invalid nlist");

        if (cfg_.storage == 1) {
            const int index_dim = effectiveDim();

            if (cfg_.M <= 0) fo.eprint("invalid PQ M");
            if (index_dim % cfg_.M != 0)
                fo.eprint("PQ dim must be divisible by M, pq_dim=" + TOS(index_dim) +
                          ", M=" + TOS(cfg_.M));

            if (cfg_.opq) {
                if (cfg_.opq_M <= 0) fo.eprint("invalid OPQ M");
                if (cfg_.dim % cfg_.opq_M != 0)
                    fo.eprint("input dim must be divisible by opq_M, dim=" + TOS(cfg_.dim) +
                              ", opq_M=" + TOS(cfg_.opq_M));

                if (cfg_.opq_out_dim > 0 && cfg_.opq_out_dim % cfg_.M != 0)
                    fo.eprint("opq_out_dim must be divisible by PQ M, opq_out_dim=" +
                              TOS(cfg_.opq_out_dim) + ", M=" + TOS(cfg_.M));
            }
        }
    }

    void reservoirSample(std::vector<float>& sample_) {
        fo.print("Start reservoir sampling ...");

        const int64_t D = cfg_.dim;
        const int64_t S = cfg_.sample_size;
        const int64_t N = cfg_.num_vectors;

        sample_.resize((size_t)S * D);

        if (!cfg_.storage) {
            const int64_t step = N / S;
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int64_t> dist(0, step > 0 ? step - 1 : 0);

            for (int64_t i = 0; i < S; i++) {
                const int64_t sample_idx = i * step + dist(gen);
                assert(sample_idx < N);
                memcpy(sample_.data() + i * D, cfg_.h_base + sample_idx * D, D * sizeof(float));
            }
        } else {
            std::ifstream fin(cfg_.input_path, std::ios::binary);
            if (!fin) fo.eprint("cannot open input file: " + cfg_.input_path.string());

            const int64_t io_block_rows = 8192;
            std::vector<float> block((size_t)io_block_rows * D);

            fin.read(reinterpret_cast<char*>(sample_.data()), sample_.size() * sizeof(float));
            if (!fin) fo.eprint("failed reading first S rows for reservoir sample_");

            std::mt19937_64 rng(cfg_.seed);
            int64_t seen = S;

            while (seen < N) {
                const int64_t rows = std::min<int64_t>(io_block_rows, N - seen);
                fin.read(reinterpret_cast<char*>(block.data()), (size_t)rows * D * sizeof(float));
                if (!fin) fo.eprint("failed reading input for reservoir sample_");

                for (int64_t r = 0; r < rows; ++r, ++seen) {
                    std::uniform_int_distribution<uint64_t> dis(0, (uint64_t)seen);
                    const uint64_t j = dis(rng);
                    if (j < (uint64_t)S) {
                        std::memcpy(sample_.data() + (size_t)j * D, &block[r * D],
                                    sizeof(float) * D);
                    }
                }

                if (cfg_.verbose && ((seen % 10'000'000) == 0 || seen == N))
                    fo.print("Reservoir sample_ progress: " + TOS(seen) + "/" + TOS(N));
            }
        }

        if (cfg_.normalize_train_sample &&
            (cfg_.metric == DIST_METRIC::IP_ || cfg_.metric == DIST_METRIC::COS_)) {
            normalizeRows(sample_.data(), S, (int)D);
        }

        fo.print("Reservoir sampling finished: size=" + TOS(sample_.size() / D));
    }

    static void normalizeRows(float* x, int64_t rows, int dim) {
        for (int64_t i = 0; i < rows; ++i) {
            float* row = x + (size_t)i * dim;
            double ss = 0.0;
            for (int d = 0; d < dim; ++d) ss += double(row[d]) * double(row[d]);
            const double inv = ss > 0.0 ? 1.0 / std::sqrt(ss) : 1.0;
            for (int d = 0; d < dim; ++d) row[d] = float(double(row[d]) * inv);
        }
    }

    faiss::MetricType faissMetric() const {
        switch (cfg_.metric) {
            case DIST_METRIC::L2_:
                return faiss::METRIC_L2;
            case DIST_METRIC::IP_:
            case DIST_METRIC::COS_:
                return faiss::METRIC_INNER_PRODUCT;
            default:
                fo.eprint("faissMetric: unknown metric");
                return faiss::METRIC_L2;
        }
    }

    std::unique_ptr<faiss::Index> makeCpuFlatIndex(int dim) const {
        const faiss::MetricType metric = faissMetric();
        if (metric == faiss::METRIC_L2)
            return std::make_unique<faiss::IndexFlatL2>(dim);
        else
            return std::make_unique<faiss::IndexFlatIP>(dim);
    }
};

}  // namespace efanna2e