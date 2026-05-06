#include <cub/cub.cuh>

#include "clock.cuh"

namespace efanna2e {
// 如果多个线程对同一个 key 写不同 val，会后写覆盖前写。这里 val 是 cache_idx（理论上同一 key
// 不应该插入成多个 cache_idx），所以 OK。
__device__ __forceinline__ void hash_insert(const DevHash& ht, int key, int val) {
    uint32_t h = hash_u32((uint32_t)key) & (uint32_t)ht.mask;
    int first_tomb = -1;
#pragma unroll 1
    for (int i = 0; i < ht.max_probe; ++i) {
        const int k = ht.keys[h];  // plain load first
        if (k == key) {
            ht.vals[h] = val;
            return;
        }
        if (k == kEmptyKey) {  // try claim empty
            const int prev = atomicCAS(&ht.keys[h], kEmptyKey, key);
            if (prev == kEmptyKey || prev == key) {
                ht.vals[h] = val;
                return;
            }
            // someone raced, continue
        } else if (k == kTombKey)  // 避免“在 tomb 处先插入、而 key 其实稍后的位置已经存在”的情况
            if (first_tomb < 0) first_tomb = (int)h;
        h = (h + 1) & ht.mask;
    }
    if (first_tomb >= 0) {
        const int prev = atomicCAS(&ht.keys[first_tomb], kTombKey, key);
        if (prev == kTombKey || prev == key) ht.vals[first_tomb] = val;
    } else
        printf("ERROR: hash_insert failed\n");
}

__global__ void lookup_ids_to_cache_idxs_vecs_kernel(DevHash ht, DevHash vec_ht,
                                                     Slot* __restrict__ slots, const int* ids,
                                                     int n, int* out_cache_idxs,
                                                     int* out_vec_cache_idxs, int* out_nbr_ids,
                                                     int max_degree) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    const int idx = hash_lookup(ht, ids[tid]), vec_idx = hash_lookup(vec_ht, ids[tid]);
    assert(idx >= 0);
    assert(vec_idx >= 0);
    out_cache_idxs[tid] = idx;
    out_vec_cache_idxs[tid] = vec_idx;
    const int deg = slots[idx].deg, *nbrs = slots[idx].nbrs;
    int* nbr_ids_out = out_nbr_ids + tid * max_degree;
    for (int i = 0; i < deg; i++) nbr_ids_out[i] = nbrs[i];
    for (int i = deg; i < max_degree; i++) nbr_ids_out[i] = -1;
}

__global__ void lookup_ids_to_cache_idxs_vecs_kernel(DevHash ht, DevHash vec_ht, const int* ids,
                                                     int n, int* out_cache_idxs,
                                                     int* out_vec_cache_idxs) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    const int idx = hash_lookup(ht, ids[tid]), vec_idx = hash_lookup(vec_ht, ids[tid]);
    assert(idx >= 0);
    assert(vec_idx >= 0);
    out_cache_idxs[tid] = idx;
    out_vec_cache_idxs[tid] = vec_idx;
}

__global__ void lookup_ids_to_cache_idxs_kernel(DevHash ht, Slot* __restrict__ slots,
                                                const int* ids, int n, int* out_cache_idxs,
                                                int* out_nbr_ids, int max_degree) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    const int idx = hash_lookup(ht, ids[tid]);
    assert(idx >= 0);
    out_cache_idxs[tid] = idx;
    const int deg = slots[idx].deg, *nbrs = slots[idx].nbrs;
    int* nbr_ids_out = out_nbr_ids + tid * max_degree;
    for (int i = 0; i < deg; i++) nbr_ids_out[i] = nbrs[i];
    for (int i = deg; i < max_degree; i++) nbr_ids_out[i] = -1;
}

__global__ void lookup_ids_to_cache_idxs_kernel(DevHash ht, const int* ids, int n,
                                                int* out_cache_idxs) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    const int idx = hash_lookup(ht, ids[tid]);
    assert(idx >= 0);
    out_cache_idxs[tid] = idx;
}

__global__ void lookup_vec_ids_to_cache_idxs_kernel(DevHash vec_ht, const int* ids, int n,
                                                    int* out_vec_cache_idxs) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    const int id = ids[tid];
    if (id >= 0) {
        const int vec_idx = hash_lookup(vec_ht, id);
        assert(vec_idx >= 0);
        out_vec_cache_idxs[tid] = vec_idx;
    } else
        out_vec_cache_idxs[tid] = -1;
}

// 或许可以用 warp 优化？
__global__ void clock_scan_kernel(const Slot* __restrict__ slots, const uint32_t* __restrict__ pin,
                                  uint8_t* __restrict__ ref, int N1, int* hand, int scan_len,
                                  int need,
                                  int* __restrict__ out_victims,  // [need] (append)
                                  int* __restrict__ out_victim_ids,
                                  int* __restrict__ inout_found  // single int on device
) {
    // 多线程并行扫窗口；用 atomicAdd 追加 victims
    const int bid = blockIdx.x, tid = threadIdx.x, stride = gridDim.x * blockDim.x;
    __shared__ int s_hand;
    if (tid == 0) s_hand = *hand;
    __syncthreads();

    for (int off = bid * blockDim.x + tid; off < scan_len; off += stride) {
        // if (atomicAdd(inout_found, 0) >= need) break;
        int idx = (s_hand + off) % N1;
        // if (idx >= N1) idx -= N1;  // scan_len 通常 << N1, 这样足够
        if (pin[idx]) continue;
        // ref==1: 清零给第二次机会；ref==0: 选为 victim
        if (ref[idx]) ref[idx] = 0;
        // 如果另一个 kernel 正在 ref[idx]=1，可能把刚置位的 1 覆盖掉。CLOCK
        // 本身允许这种近似（不会崩，只是会略偏向淘汰）。如果在意热点保护，可改成原子位操作
        // atomicExch((unsigned int*)(ref + idx), 0);
        else {
            const int pos = atomicAdd(inout_found, 1);
            if (pos < need) {
                out_victims[pos] = idx;
                out_victim_ids[pos] = slots[idx].node_id;
            } else
                break;
        }
    }
    // 单线程更新 next_hand（hand 前进 scan_len）
    if (!bid && !tid) *hand = (s_hand + scan_len) % N1;
}

__global__ void vec_clock_scan_kernel(const int* __restrict__ vecs_id,
                                      const uint32_t* __restrict__ pin, uint8_t* __restrict__ ref,
                                      int N1, int* hand, int scan_len, int need,
                                      int* __restrict__ out_victims,  // [need] (append)
                                      int* __restrict__ out_victim_ids,
                                      int* __restrict__ inout_found  // single int on device
) {
    // 多线程并行扫窗口；用 atomicAdd 追加 victims
    const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x),
              stride = (int)(gridDim.x * blockDim.x);
    for (int off = tid; off < scan_len; off += stride) {
        if (atomicAdd(inout_found, 0) >= need) break;
        int idx = *hand + off;
        if (idx >= N1) idx -= N1;  // scan_len 通常 << N1, 这样足够
        if (pin[idx]) continue;
        // ref==1: 清零给第二次机会；ref==0: 选为 victim
        if (ref[idx]) ref[idx] = 0;
        // 如果另一个 kernel 正在 ref[idx]=1，可能把刚置位的 1 覆盖掉。CLOCK
        // 本身允许这种近似（不会崩，只是会略偏向淘汰）。如果在意热点保护，可改成原子位操作
        // atomicExch((unsigned int*)(ref + idx), 0);
        else {
            const int pos = atomicAdd(inout_found, 1);
            if (pos < need) {
                out_victims[pos] = idx;
                out_victim_ids[pos] = vecs_id[idx];
            }
        }
    }
    // 单线程更新 next_hand（hand 前进 scan_len）
    if (tid == 0) {
        int nh = *hand + scan_len;
        if (nh >= N1) nh -= N1;
        *hand = nh;
    }
}

__global__ void clear_dirty_pin_setref_kernel(uint8_t* dirty, uint32_t* pin, uint8_t* ref,
                                              const int* cache_idxs, int n, uint32_t pin_id) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        const int idx = cache_idxs[tid];
        if (idx >= 0) {
            dirty[idx] = 0;
            atomicMax(pin + idx, pin_id);  // pin 住新数据
            ref[idx] = 1;
        }
    }
}

__global__ void clear_pin_setref_kernel(uint32_t* pin, uint8_t* ref, const int* cache_idxs, int n,
                                        uint32_t pin_id) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        const int idx = cache_idxs[tid];
        if (idx >= 0) {
            atomicMax(pin + idx, pin_id);  // pin 住新数据
            ref[idx] = 1;
        }
    }
}

__global__ void hash_init_kernel(DevHash ht, DevHash vec_ht) {
    assert(ht.H == vec_ht.H);
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ht.H; i += stride) {
        vec_ht.keys[i] = ht.keys[i] = kEmptyKey;
        vec_ht.vals[i] = ht.vals[i] = -1;
    }
}

__global__ void hash_batch_erase_insert_kernel(DevHash ht, const int* __restrict__ erase_keys,
                                               int erase_n, const int* __restrict__ ins_keys,
                                               const int* __restrict__ ins_vals, int ins_n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
    for (int i = tid; i < erase_n; i += stride) {  // erase
        const int k = erase_keys[i];
        if (k >= 0) hash_erase(ht, k);
    }
    for (int i = tid; i < ins_n; i += stride) {  // insert
        const int k = ins_keys[i], v = ins_vals[i];
        if (k >= 0 && v >= 0) hash_insert(ht, k, v);
    }
}

__global__ void gather_hits_cache_idx_kernel(const DevHash ht, const int* ids, int n,
                                             int* out_cache_idxs) {  // -1 for miss
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int id = ids[tid];
    out_cache_idxs[tid] = (id >= 0) ? hash_lookup(ht, id) : -1;
}

__global__ void set_ref_by_cache_idxs_kernel(uint8_t* __restrict__ ref,
                                             const int* __restrict__ cache_idxs, int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        const int c = cache_idxs[tid];
        if (c >= 0) {
            ref[c] = 1;
        }  // benign race
    }
}

__global__ void set_ref_pin_and_miss_kernel(
    const int* __restrict__ uniq_ids, const int* __restrict__ cache_idxs, int n,
    uint8_t* __restrict__ ref, uint32_t* __restrict__ pin, uint32_t pin_id,
    int* __restrict__ miss_ids,  // output miss ids compacted by DeviceSelect::Flagged
    uint8_t* __restrict__ miss_flags) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        const int c = cache_idxs[tid], id = uniq_ids[tid];
        miss_flags[tid] = (uint8_t)(id >= 0 && c < 0);
        miss_ids[tid] = uniq_ids[tid];
        if (c >= 0) {
            atomicMax(pin + c, pin_id);
            ref[c] = 1;
        }
    }
}

__global__ void flag_miss_kernel(
    const int* __restrict__ uniq_ids, const int* __restrict__ cache_idxs, int n,
    int* __restrict__ miss_ids,        // output miss ids compacted by DeviceSelect::Flagged
    uint8_t* __restrict__ miss_flags)  // [n] 1 if miss else 0
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        miss_flags[tid] = (uint8_t)(cache_idxs[tid] < 0);
        miss_ids[tid] = uniq_ids[tid];
    }
}

__global__ void gather_dirty_slots_by_index_kernel(const Slot* __restrict__ d_slots,
                                                   const uint8_t* __restrict__ dirty,
                                                   const int* __restrict__ idxs,
                                                   Slot* __restrict__ out_slots,
                                                   int* __restrict__ out_idx, int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x, lane = threadIdx.x % warpSize;
    if (tid < n) {
        const int idx = idxs[tid], flag = dirty[idx];
        const unsigned active = __activemask(), mask = __ballot_sync(active, flag);
        const int cnt = __popc(mask);
        if (cnt > 0) {
            int base_idx = -1;
            if (lane == 0) base_idx = atomicAdd(out_idx, cnt);
            base_idx = __shfl_sync(active, base_idx, 0);
            if (flag) out_slots[base_idx + __popc(mask & ((1u << lane) - 1))] = d_slots[idx];
        }
    }
}

__global__ void gather_all_dirty_slot_idx_kernel(const Slot* __restrict__ d_slots,
                                                 const uint8_t* __restrict__ dirty,
                                                 int* __restrict__ out_dirty_idxs,
                                                 int* __restrict__ out_idx, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x, lane = threadIdx.x % warpSize;
    if (idx < n) {
        const int flag = dirty[idx];
        const unsigned active = __activemask(), mask = __ballot_sync(active, flag);
        const int cnt = __popc(mask);
        if (cnt > 0) {
            int base_idx = -1;
            if (lane == 0) base_idx = atomicAdd(out_idx, cnt);
            base_idx = __shfl_sync(active, base_idx, 0);
            if (flag) out_dirty_idxs[base_idx + __popc(mask & ((1u << lane) - 1))] = idx;
        }
    }
}

__global__ void gather_slots_by_idx_kernel(const Slot* __restrict__ d_slots,
                                           const int* __restrict__ idxs,
                                           Slot* __restrict__ out_slots, int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) out_slots[tid] = d_slots[idxs[tid]];
}

__global__ void scatter_slots_kernel(Slot* __restrict__ d_slots,
                                     const int* __restrict__ cache_idxs,
                                     const Slot* __restrict__ in_slots, int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        const int c = cache_idxs[tid];
        if (c >= 0) d_slots[c] = in_slots[tid];
    }
}

__global__ void scatter_vecs_kernel(float* __restrict__ d_vecs, int* __restrict__ d_vec_ids,
                                    int dim, const int* __restrict__ cache_idxs,
                                    const float* __restrict__ in_vecs,   // [n*dim]
                                    const int* __restrict__ in_vec_ids,  // [n]
                                    int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x, total = n * dim;
    if (tid < total) {
        const int i = tid / dim, d = tid - i * dim, c = cache_idxs[i];
        if (c >= 0) {
            d_vecs[(size_t)c * dim + d] = in_vecs[(size_t)i * dim + d];
            if (!d) d_vec_ids[c] = in_vec_ids[i];
        }
    }
}

__global__ void fill_int_kernel(int* a, int n, int v) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
    for (int i = tid; i < n; i += stride) a[i] = v;
}

__global__ void fill_u8_kernel(uint8_t* a, int n, uint8_t v) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
    for (int i = tid; i < n; i += stride) a[i] = v;
}

__global__ void gather_vec_ids_kernel(const DevHash ht, const Slot* __restrict__ slots,
                                      const int* __restrict__ uniq_ids, int uniq_n,
                                      int* __restrict__ vec_ids, int max_degree, bool need_vec) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < uniq_n) {
        const int id = uniq_ids[tid], idx = hash_lookup(ht, id);
        assert(idx >= 0);
        const int *nbrs_in = slots[idx].nbrs, deg = slots[idx].deg;
        int* nbrs_out = vec_ids + tid * (max_degree + 1);
        for (int i = 0; i < deg; i++) nbrs_out[i] = nbrs_in[i];
        for (int i = deg; i < max_degree; i++) nbrs_out[i] = -1;
        nbrs_out[max_degree] = need_vec ? id : -1;
    }
}

__global__ void unpin_all_data_from_ids_kernel(DevHash ht, DevHash vec_ht,
                                               Slot* __restrict__ slots, const int* ids, int n,
                                               uint32_t* __restrict__ pin,
                                               uint32_t* __restrict__ vec_pin, uint32_t pin_id) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    const int idx = hash_lookup(ht, ids[tid]), vec_idx = hash_lookup(vec_ht, ids[tid]);
    assert(idx >= 0);
    assert(vec_idx >= 0);
    const int deg = slots[idx].deg, *nbrs = slots[idx].nbrs;
    for (int i = 0; i < deg; i++) {
        const int nbr_vec_idx = hash_lookup(vec_ht, nbrs[i]);
        assert(nbr_vec_idx >= 0);
        atomicCAS(vec_pin + nbr_vec_idx, pin_id, 0);
    }
    atomicCAS(pin + idx, pin_id, 0);
    atomicCAS(vec_pin + vec_idx, pin_id, 0);
}

int GpuClockCache::get_miss_ids(const DevHash& ht, const int* d_ids, int n, uint8_t* d_ref,
                                uint32_t* d_pin, uint32_t pin_id, cudaStream_t stream) {
    const dim3 blk(256), grd((n + blk.x - 1) / blk.x);

    gather_hits_cache_idx_kernel<<<grd, blk, 0, stream_hash_>>>(ht, d_ids, n, d_cache_idxs_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev_look_, stream_hash_));
    CUDA_CHECK(cudaStreamWaitEvent(stream, ev_look_));
    // hit => ref=1 (device-only, no D2H)
    set_ref_pin_and_miss_kernel<<<grd, blk, 0, stream>>>(d_ids, d_cache_idxs_, n, d_ref, d_pin,
                                                         pin_id, d_miss_ids_tmp_, d_miss_flags_);
    CUDA_CHECK(cudaGetLastError());
    size_t flagged_bytes = 0;
    CUDA_CHECK(cub::DeviceSelect::Flagged(nullptr, flagged_bytes, d_miss_ids_tmp_, d_miss_flags_,
                                          d_miss_ids_, d_miss_count_, n, stream));
    alloc_cub_temp(flagged_bytes);
    CUDA_CHECK(cub::DeviceSelect::Flagged(d_cub_temp_, cub_temp_bytes_, d_miss_ids_tmp_,
                                          d_miss_flags_, d_miss_ids_, d_miss_count_, n, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(miss_n_host_, d_miss_count_, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaEventRecord(ev_tmp_, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_tmp_));  // 只等 miss_n_host_

    return *miss_n_host_;
}

void GpuClockCache::unpin_from_ids(const int* d_ids, int n, uint32_t pin_id, cudaStream_t stream) {
    const int blk = 256, grd = (n + blk - 1) / blk;
    unpin_all_data_from_ids_kernel<<<grd, blk, 0, stream>>>(ht_, vec_ht_, d_slots_, d_ids, n,
                                                            d_pin_, d_vec_pin_, pin_id);
}

EnsureResult GpuClockCache::ensure_vecs_from_device_ids(const int* d_node_ids_to_use, int n,
                                                        uint32_t pin_id, cudaStream_t stream) {
    EnsureResult ret;
    if (n <= 0) return ret;
    //  Step A: copy ids into workspace, sort + unique
    ensure_workspace(n);
    CUDA_CHECK(cudaMemcpyAsync(d_ids_, d_node_ids_to_use, sizeof(int) * n,
                               cudaMemcpyDeviceToDevice,
                               stream));  // copy to d_ids_

    ret.uniq_vec_n = n;
    const int miss_n = get_miss_ids(vec_ht_, d_ids_, n, d_vec_ref_, d_vec_pin_, pin_id, stream);
    plan_.pin_id = pin_id;
    if (miss_n > 0) {
        ret.miss_vec_n = miss_n;
        for (int miss_i = 0; miss_i < miss_n; miss_i += cfg_.staging_cap) {
            const int batch = std::min(cfg_.staging_cap, miss_n - miss_i);
            // 选择 victims（CLOCK on GPU）+ 拉 dirty + 组装 plan
            prepare_vec_swap_plan_clock(d_miss_ids_ + miss_i, batch, stream);
            execute_vec_swap_in(stream);
            execute_hash_updates(vec_ht_);
        }
        CUDA_CHECK(cudaEventRecord(ev_look_, stream_hash_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, ev_look_));
    }
    return ret;
}

// Ensure all ids in d_node_ids_to_use[0..n) are in GPU cache:
// 1) sort+unique ids
// 2) lookup -> miss
// 3) copy miss_ids to host
// 4) evict+swap in using LRU (host)
// 5) update device hash
// need_vec: 是否需要节点自己的向量数据（节点邻居向量数据肯定需要）
EnsureResult GpuClockCache::ensure_cached_from_device_ids(const int* d_node_ids_to_use, int n,
                                                          uint32_t pin_id, bool need_vec,
                                                          cudaStream_t stream) {
    fo.print("ersure with n=" + TOS(n));
    auto s = now_time();
    EnsureResult ret;
    if (n <= 0) return ret;
    //  Step A: copy ids into workspace, sort + unique
    ensure_workspace(n);
    CUDA_CHECK(cudaMemcpyAsync(d_ids_, d_node_ids_to_use, sizeof(int) * n,
                               cudaMemcpyDeviceToDevice, stream));  // copy to d_ids_
    // sort keys
    size_t sort_bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr, sort_bytes, d_ids_, d_ids_sorted_, n, 0, 32,
                                              stream));
    alloc_cub_temp(sort_bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(d_cub_temp_, cub_temp_bytes_, d_ids_, d_ids_sorted_,
                                              n, 0, 32, stream));
    // unique
    size_t unique_bytes = 0;
    CUDA_CHECK(cub::DeviceSelect::Unique(nullptr, unique_bytes, d_ids_sorted_, d_ids_unique_,
                                         d_unique_count_, n, stream));
    alloc_cub_temp(unique_bytes);
    CUDA_CHECK(cub::DeviceSelect::Unique(d_cub_temp_, cub_temp_bytes_, d_ids_sorted_,
                                         d_ids_unique_, d_unique_count_, n, stream));
    CUDA_CHECK(cudaMemcpyAsync(uniq_n_host_, d_unique_count_, sizeof(int), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaEventRecord(ev_tmp_, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_tmp_));  // 只等 uniq_n_host_ 这一个 host 读回

    int uniq_n = *uniq_n_host_;
    ret.uniq_n = uniq_n;
    int miss_n = get_miss_ids(ht_, d_ids_unique_, uniq_n, d_ref_, d_pin_, pin_id, stream);
    if (miss_n > 0) {
        fo.print("Miss: " + TOS(miss_n) + "/" + TOS(uniq_n) + "/" + TOS(n) + ", " +
                 TOS(miss_n * 100.0 / uniq_n) + "%");
        ret.miss_n = miss_n;
        // 分块处理 miss：每块都在 IO stream 完成换入换出 + hash 更新
        for (int miss_i = 0; miss_i < miss_n; miss_i += cfg_.staging_cap) {
            const int batch = std::min(cfg_.staging_cap, miss_n - miss_i);
            // 选择 victims（CLOCK on GPU）+ 拉 dirty + 组装 plan
            prepare_swap_plan_clock(d_miss_ids_ + miss_i, batch, stream);
            execute_swap_in(stream);
            execute_hash_updates(ht_);
        }
    }

    const dim3 blk(256), grd((uniq_n + blk.x - 1) / blk.x);

    gather_vec_ids_kernel<<<grd, blk, 0, stream_hash_>>>(ht_, d_slots_, d_ids_unique_, uniq_n,
                                                         d_vec_ids_, cfg_.max_degree, need_vec);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev_look_, stream_hash_));
    CUDA_CHECK(cudaStreamWaitEvent(stream, ev_look_));

    const int vec_n = uniq_n * (cfg_.max_degree + 1);
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr, sort_bytes, d_vec_ids_, d_vec_ids1_, vec_n,
                                              0, 32, stream));
    alloc_cub_temp(sort_bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(d_cub_temp_, cub_temp_bytes_, d_vec_ids_,
                                              d_vec_ids1_, vec_n, 0, 32, stream));
    // unique
    CUDA_CHECK(cub::DeviceSelect::Unique(nullptr, unique_bytes, d_vec_ids1_, d_vec_ids_,
                                         d_unique_count_, vec_n, stream));
    alloc_cub_temp(unique_bytes);
    CUDA_CHECK(cub::DeviceSelect::Unique(d_cub_temp_, cub_temp_bytes_, d_vec_ids1_, d_vec_ids_,
                                         d_unique_count_, vec_n, stream));

    CUDA_CHECK(cudaMemcpyAsync(uniq_n_host_, d_unique_count_, sizeof(int), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaEventRecord(ev_tmp_, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_tmp_));
    uniq_n = *uniq_n_host_;
    ret.uniq_vec_n = uniq_n;

    miss_n = get_miss_ids(vec_ht_, d_vec_ids_, uniq_n, d_vec_ref_, d_vec_pin_, pin_id, stream);
    plan_.pin_id = pin_id;
    if (miss_n > 0) {
        ret.miss_vec_n = miss_n;
        for (int miss_i = 0; miss_i < miss_n; miss_i += cfg_.staging_cap) {
            const int batch = std::min(cfg_.staging_cap, miss_n - miss_i);
            // 选择 victims（CLOCK on GPU）+ 拉 dirty + 组装 plan
            prepare_vec_swap_plan_clock(d_miss_ids_ + miss_i, batch, stream);
            execute_vec_swap_in(stream);
            execute_hash_updates(vec_ht_);
        }
        CUDA_CHECK(cudaEventRecord(ev_look_, stream_hash_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, ev_look_));
    }

    fo.print("ensure_cached_from_device_ids: " + TOS(time_diff(s)) + "s");
    return ret;
}

EnsureResult GpuClockCache::ensure_self_from_device_ids(const int* d_node_ids_to_use, int n,
                                                        uint32_t pin_id, bool need_vec,
                                                        cudaStream_t stream) {
    EnsureResult ret;
    if (n <= 0) return ret;
    //  Step A: copy ids into workspace, sort + unique
    ensure_workspace(n);
    CUDA_CHECK(cudaMemcpyAsync(d_ids_, d_node_ids_to_use, sizeof(int) * n,
                               cudaMemcpyDeviceToDevice, stream));  // copy to d_ids_
    // sort keys
    size_t sort_bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr, sort_bytes, d_ids_, d_ids_sorted_, n, 0, 32,
                                              stream));
    alloc_cub_temp(sort_bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(d_cub_temp_, cub_temp_bytes_, d_ids_, d_ids_sorted_,
                                              n, 0, 32, stream));
    // unique
    size_t unique_bytes = 0;
    CUDA_CHECK(cub::DeviceSelect::Unique(nullptr, unique_bytes, d_ids_sorted_, d_ids_unique_,
                                         d_unique_count_, n, stream));
    alloc_cub_temp(unique_bytes);
    CUDA_CHECK(cub::DeviceSelect::Unique(d_cub_temp_, cub_temp_bytes_, d_ids_sorted_,
                                         d_ids_unique_, d_unique_count_, n, stream));
    CUDA_CHECK(cudaMemcpyAsync(uniq_n_host_, d_unique_count_, sizeof(int), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaEventRecord(ev_tmp_, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_tmp_));  // 只等 uniq_n_host_ 这一个 host 读回

    int uniq_n = *uniq_n_host_;
    ret.uniq_n = uniq_n;
    int miss_n = get_miss_ids(ht_, d_ids_unique_, uniq_n, d_ref_, d_pin_, pin_id, stream);
    if (miss_n > 0) {
        ret.miss_n = miss_n;
        // 分块处理 miss：每块都在 IO stream 完成换入换出 + hash 更新
        for (int miss_i = 0; miss_i < miss_n; miss_i += cfg_.staging_cap) {
            const int batch = std::min(cfg_.staging_cap, miss_n - miss_i);
            // 选择 victims（CLOCK on GPU）+ 拉 dirty + 组装 plan
            prepare_swap_plan_clock(d_miss_ids_ + miss_i, batch, stream);
            execute_swap_in(stream);
            execute_hash_updates(ht_);
        }
    }
    if (!need_vec) return ret;

    ret.uniq_vec_n = uniq_n;
    miss_n = get_miss_ids(vec_ht_, d_ids_unique_, uniq_n, d_vec_ref_, d_vec_pin_, pin_id, stream);
    plan_.pin_id = pin_id;
    if (miss_n > 0) {
        ret.miss_vec_n = miss_n;
        for (int miss_i = 0; miss_i < miss_n; miss_i += cfg_.staging_cap) {
            const int batch = std::min(cfg_.staging_cap, miss_n - miss_i);
            // 选择 victims（CLOCK on GPU）+ 拉 dirty + 组装 plan
            prepare_vec_swap_plan_clock(d_miss_ids_ + miss_i, batch, stream);
            execute_vec_swap_in(stream);
            execute_hash_updates(vec_ht_);
        }
        CUDA_CHECK(cudaEventRecord(ev_look_, stream_hash_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, ev_look_));
    }
    return ret;
}

void GpuClockCache::lookup_slots_vecs(const int* d_node_ids_to_use, int n, int* d_cache_idxs_out,
                                      int* d_vec_cache_idxs_out, cudaStream_t stream,
                                      bool need_vec) {
    const dim3 blk(256), grd1((n + blk.x - 1) / blk.x),
        grd2((n * cfg_.max_degree + blk.x - 1) / blk.x);

    if (need_vec) {
        assert(d_vec_cache_idxs_out != nullptr);
        lookup_ids_to_cache_idxs_vecs_kernel<<<grd1, blk, 0, stream_hash_>>>(
            ht_, vec_ht_, d_node_ids_to_use, n, d_cache_idxs_out, d_vec_cache_idxs_out);
        CUDA_CHECK(cudaGetLastError());
    } else {
        lookup_ids_to_cache_idxs_kernel<<<grd1, blk, 0, stream_hash_>>>(ht_, d_node_ids_to_use, n,
                                                                        d_cache_idxs_out);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(ev_look_, stream_hash_));
    CUDA_CHECK(cudaStreamWaitEvent(stream, ev_look_));
}

void GpuClockCache::lookup_all_data(const int* d_node_ids_to_use, int n, int* d_cache_idxs_out,
                                    int* d_vec_cache_idxs_out, int* d_nbr_vec_cache_idxs_out,
                                    cudaStream_t stream, bool need_vec) {
    fo.print("lookup_all_data: n=" + TOS(n));
    const dim3 blk(256), grd1((n + blk.x - 1) / blk.x),
        grd2((n * cfg_.max_degree + blk.x - 1) / blk.x);

    if (need_vec) {
        assert(d_vec_cache_idxs_out != nullptr);
        lookup_ids_to_cache_idxs_vecs_kernel<<<grd1, blk, 0, stream_hash_>>>(
            ht_, vec_ht_, d_slots_, d_node_ids_to_use, n, d_cache_idxs_out, d_vec_cache_idxs_out,
            d_vec_ids_, cfg_.max_degree);
    } else
        lookup_ids_to_cache_idxs_kernel<<<grd1, blk, 0, stream_hash_>>>(
            ht_, d_slots_, d_node_ids_to_use, n, d_cache_idxs_out, d_vec_ids_, cfg_.max_degree);
    CUDA_CHECK(cudaGetLastError());

    lookup_vec_ids_to_cache_idxs_kernel<<<grd2, blk, 0, stream_hash_>>>(
        vec_ht_, d_vec_ids_, n * cfg_.max_degree, d_nbr_vec_cache_idxs_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev_look_, stream_hash_));
    CUDA_CHECK(cudaStreamWaitEvent(stream, ev_look_));
    fo.print("lookup_all_data finished");
}

void GpuClockCache::lookup_vecs(const int* d_node_ids_to_use, int n, int* d_vec_cache_idxs_out,
                                cudaStream_t stream) {
    const dim3 blk(256), grd((n * cfg_.max_degree + blk.x - 1) / blk.x);

    assert(d_vec_cache_idxs_out != nullptr);
    lookup_vec_ids_to_cache_idxs_kernel<<<grd, blk, 0, stream_hash_>>>(vec_ht_, d_node_ids_to_use,
                                                                       n, d_vec_cache_idxs_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev_look_, stream_hash_));
    if (stream)
        CUDA_CHECK(cudaStreamWaitEvent(stream, ev_look_));
    else
        CUDA_CHECK(cudaEventSynchronize(ev_look_));
}

void GpuClockCache::prepare_swap_plan_clock(const int* d_miss_ids, int miss_n,
                                            cudaStream_t stream_io) {
    plan_.n = miss_n;
    // (1) CLOCK scan to fill d_victim_cache_idxs_[0..miss_n)
    // 多轮扫描直到够 miss_n 个 victim
    // int oversub = std::min(std::max(1, cfg_.oversub_init), cfg_.oversub_max);
    int oversub = cfg_.oversub_init;
    if (miss_n <= 4096)
        oversub = std::max(oversub, 16);
    else if (miss_n > 65536)
        oversub = std::min(oversub, 4);
    oversub = std::min(oversub, cfg_.oversub_max);
    CUDA_CHECK(cudaMemsetAsync(d_clock_found_, 0, sizeof(int), stream_io));
    while (true) {
        const int scan_len =
            (int)std::min<int64_t>((int64_t)cfg_.N1, (int64_t)miss_n * (int64_t)oversub);
        // reset found on device to current found (append semantics)
        clock_scan_kernel<<<cfg_.scan_blocks, cfg_.scan_threads, 0, stream_io>>>(
            d_slots_, d_pin_, d_ref_, cfg_.N1, d_clock_hand_, scan_len, miss_n,
            d_victim_cache_idxs_, d_victim_node_ids_, d_clock_found_);
        CUDA_CHECK(cudaGetLastError());
        // 读回 found 和 next_hand（host 必须知道是否继续）
        CUDA_CHECK(cudaMemcpyAsync(found_host_, d_clock_found_, sizeof(int),
                                   cudaMemcpyDeviceToHost, stream_io));
        CUDA_CHECK(cudaEventRecord(ev_tmp_, stream_io));
        CUDA_CHECK(cudaEventSynchronize(ev_tmp_));

        if (*found_host_ < miss_n) {  // 扩大扫描窗口，避免多轮
            oversub = std::min(oversub * 2, cfg_.oversub_max);
            if (scan_len == cfg_.N1 && oversub == cfg_.oversub_max) {
                fo.eprint("oversub_max reached, but still not enough victims. ");
                // still not enough: likely too many pinned; correctness requires user fix pin
                // usage We can loop again (hand advances), but warn once. (No stdout spam here;
                // you can add rate-limited logging if needed.)
            }
        } else
            break;
    }
    plan_.d_erase_keys = d_victim_node_ids_;
    plan_.d_ins_keys = d_miss_ids;
    plan_.d_ins_vals = d_victim_cache_idxs_;
    CUDA_CHECK(cudaMemcpyAsync(plan_.in_node_id, d_miss_ids, sizeof(int) * miss_n,
                               cudaMemcpyDeviceToHost, stream_io));
    // (4) 按需 dirty write back
    CUDA_CHECK(cudaMemsetAsync(d_dirty_slot_idx_, 0, sizeof(int), stream_io));
    gather_dirty_slots_by_index_kernel<<<(miss_n + 255) / 256, 256, 0, stream_io>>>(
        d_slots_, d_dirty_, d_victim_cache_idxs_, d_slots_gather_, d_dirty_slot_idx_, miss_n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(dirty_slot_num_host_, d_dirty_slot_idx_, sizeof(int),
                               cudaMemcpyDeviceToHost, stream_io));
    CUDA_CHECK(cudaEventRecord(ev_tmp_, stream_io));
    CUDA_CHECK(cudaEventSynchronize(ev_tmp_));
    // (5) CPU gather in_slots to pinned staging
    for (int i = 0; i < miss_n; ++i) {
        const int nid = plan_.in_node_id[i];
        plan_.h_slots_in[i] = slots_cpu_[nid];
        plan_.h_slots_in[i].node_id = nid;
    }
    const int dn = *dirty_slot_num_host_;
    if (dn > 0) {
        CUDA_CHECK(cudaMemcpyAsync(plan_.h_slots_out, d_slots_gather_, sizeof(Slot) * dn,
                                   cudaMemcpyDeviceToHost, stream_io));
        CUDA_CHECK(cudaEventRecord(ev_tmp_, stream_io));
        CUDA_CHECK(cudaEventSynchronize(ev_tmp_));
#pragma omp parallel for
        for (int i = 0; i < dn; ++i)  // scatter back to CPU slots
            slots_cpu_[plan_.h_slots_out[i].node_id] = plan_.h_slots_out[i];
    }
}

void GpuClockCache::prepare_vec_swap_plan_clock(const int* d_miss_ids, int miss_n,
                                                cudaStream_t stream_io) {
    plan_.n = miss_n;
    // (1) CLOCK scan to fill d_victim_cache_idxs_[0..miss_n)
    // 多轮扫描直到够 miss_n 个 victim
    // int oversub = std::min(std::max(1, cfg_.oversub_init), cfg_.oversub_max);
    int oversub = cfg_.oversub_init;
    if (miss_n <= 4096)
        oversub = std::max(oversub, 16);
    else if (miss_n > 65536)
        oversub = std::min(oversub, 4);
    oversub = std::min(oversub, cfg_.oversub_max);
    *found_host_ = 0;
    CUDA_CHECK(cudaMemsetAsync(d_clock_found_, 0, sizeof(int), stream_io));
    while (*found_host_ < miss_n) {
        int scan_len =
            (int)std::min<int64_t>((int64_t)cfg_.N1, (int64_t)miss_n * (int64_t)oversub);
        // reset found on device to current found (append semantics)
        vec_clock_scan_kernel<<<cfg_.scan_blocks, cfg_.scan_threads, 0, stream_io>>>(
            d_vecs_id_, d_vec_pin_, d_vec_ref_, cfg_.N1, d_vec_clock_hand_, scan_len, miss_n,
            d_victim_cache_idxs_, d_victim_node_ids_, d_clock_found_);
        CUDA_CHECK(cudaGetLastError());
        // 读回 found 和 next_hand（host 必须知道是否继续）
        CUDA_CHECK(cudaMemcpyAsync(found_host_, d_clock_found_, sizeof(int),
                                   cudaMemcpyDeviceToHost, stream_io));
        CUDA_CHECK(cudaEventRecord(ev_tmp_, stream_io));
        CUDA_CHECK(cudaEventSynchronize(ev_tmp_));

        if (*found_host_ < miss_n) {  // 扩大扫描窗口，避免多轮
            oversub = std::min(oversub * 2, cfg_.oversub_max);
            if (scan_len == cfg_.N1 && oversub == cfg_.oversub_max) {
                fo.eprint("oversub_max reached, but still not enough victims. ");
                // still not enough: likely too many pinned; correctness requires user fix pin
                // usage We can loop again (hand advances), but warn once. (No stdout spam here;
                // you can add rate-limited logging if needed.)
            }
        }
    }
    plan_.d_erase_keys = d_victim_node_ids_;
    plan_.d_ins_keys = d_miss_ids;
    plan_.d_ins_vals = d_victim_cache_idxs_;
    CUDA_CHECK(cudaMemcpyAsync(plan_.in_node_id, d_miss_ids, sizeof(int) * miss_n,
                               cudaMemcpyDeviceToHost, stream_io));
    CUDA_CHECK(cudaEventRecord(ev_tmp_, stream_io));
    CUDA_CHECK(cudaEventSynchronize(ev_tmp_));
    // (5) CPU gather in_vecs to pinned staging
    for (int i = 0; i < miss_n; ++i)
        std::memcpy(plan_.h_vecs_in + (size_t)i * (size_t)cfg_.dim,
                    vecs_cpu_ + (size_t)plan_.in_node_id[i] * (size_t)cfg_.dim,
                    sizeof(float) * cfg_.dim);
}

void GpuClockCache::execute_swap_in(cudaStream_t stream) {
    const int n = plan_.n;
    // H2D contiguous
    CUDA_CHECK(cudaMemcpyAsync(d_slots_in_contig_, plan_.h_slots_in, sizeof(Slot) * n,
                               cudaMemcpyHostToDevice, stream));
    // scatter to cache lines
    scatter_slots_kernel<<<(n + 255) / 256, 256, 0, stream>>>(d_slots_, d_victim_cache_idxs_,
                                                              d_slots_in_contig_, n);
    CUDA_CHECK(cudaGetLastError());
    // clear dirty and pin for these cache lines
    clear_dirty_pin_setref_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        d_dirty_, d_pin_, d_ref_, plan_.d_ins_vals, n, plan_.pin_id);
    CUDA_CHECK(cudaGetLastError());
}

void GpuClockCache::execute_vec_swap_in(cudaStream_t stream) {
    const int n = plan_.n;
    // H2D contiguous
    CUDA_CHECK(cudaMemcpyAsync(d_vecs_in_contig_, plan_.h_vecs_in,
                               sizeof(float) * (size_t)n * (size_t)cfg_.dim,
                               cudaMemcpyHostToDevice, stream));
    // scatter to cache lines
    scatter_vecs_kernel<<<(n * cfg_.dim + 255) / 256, 256, 0, stream>>>(
        d_vecs_, d_vecs_id_, cfg_.dim, d_victim_cache_idxs_, d_vecs_in_contig_, plan_.d_ins_keys,
        n);
    CUDA_CHECK(cudaGetLastError());
    // clear pin for these vec cache lines
    clear_pin_setref_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        d_vec_pin_, d_vec_ref_, plan_.d_ins_vals, n, plan_.pin_id);
    CUDA_CHECK(cudaGetLastError());
}

void GpuClockCache::execute_hash_updates(DevHash& ht) {
    const int n = plan_.n;
    const int threads = 256, blocks = std::min<int>(4096, (n + threads - 1) / threads);
    hash_batch_erase_insert_kernel<<<blocks, threads, 0, stream_hash_>>>(
        ht, plan_.d_erase_keys, n, plan_.d_ins_keys, plan_.d_ins_vals, n);
    CUDA_CHECK(cudaGetLastError());
}

// Flush all dirty cache lines back to CPU (full scan).
void GpuClockCache::flush_all() {
    // copy dirty flags + node_ids to host, then for each dirty line do D2H Slot and scatter.
    int *d_dirty_idxs, *d_dirty_num, *h_dirty_num;
    Slot* h_dirty_slots;
    CUDA_CHECK(cudaMallocHost(&h_dirty_num, sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&h_dirty_slots, sizeof(Slot) * cfg_.N1));
    CUDA_CHECK(cudaMalloc(&d_dirty_idxs, cfg_.N1 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dirty_num, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_dirty_num, 0, sizeof(int)));
    const dim3 blk(256), grd((cfg_.N1 + blk.x - 1) / blk.x);
    gather_all_dirty_slot_idx_kernel<<<grd, blk>>>(d_slots_, d_dirty_, d_dirty_idxs, d_dirty_num,
                                                   cfg_.N1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_dirty_num, d_dirty_num, sizeof(int), cudaMemcpyDeviceToHost));

    int off = 0, dirty_num = *h_dirty_num;
    if (dirty_num <= 0) return;

    const dim3 grd1((cfg_.staging_cap + blk.x - 1) / blk.x);
    while (off < dirty_num) {
        const int chunk = std::min<int>(cfg_.staging_cap, dirty_num - off);
        gather_slots_by_idx_kernel<<<grd1, blk>>>(d_slots_, d_dirty_idxs + off, d_slots_gather_,
                                                  chunk);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_dirty_slots + off, d_slots_gather_, sizeof(Slot) * chunk,
                              cudaMemcpyDeviceToHost));
        off += chunk;
    }
#pragma omp parallel for
    for (int i = 0; i < dirty_num; i++) slots_cpu_[h_dirty_slots[i].node_id] = h_dirty_slots[i];

    fill_u8_kernel<<<grd, blk>>>(d_dirty_, cfg_.N1, 0);  // clear all dirty
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFreeHost(h_dirty_num));
    CUDA_CHECK(cudaFreeHost(h_dirty_slots));
    CUDA_CHECK(cudaFree(d_dirty_idxs));
    CUDA_CHECK(cudaFree(d_dirty_num));
}

void GpuClockCache::init_device_structs() {
    // init hash table
    const int threads = 256, blocks = std::min<int>(4096, (cfg_.hash_H + threads - 1) / threads);
    hash_init_kernel<<<blocks, threads>>>(ht_, vec_ht_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

};  // namespace efanna2e