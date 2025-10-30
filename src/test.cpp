#include "test.h"

#include <algorithm>
#include <cfloat>
#include <cmath>

#include "uni.h"

using namespace efanna2e;

bool Test::compare_pairs(const std::pair<float, int> &a, const std::pair<float, int> &b) {
    return metric == DIST_METRIC::L2 ? a.first < b.first   // 按 float 值升序
                                     : a.first > b.first;  // 按 float 值降序
}

Test::Test(GPUFuncs *gpufuncs, const float *h_test_query, const float *h_base, const int dim,
           const int base_n, const int k, DIST_METRIC metric, const int max_degree)
    : gpufuncs(gpufuncs),
      dim(dim),
      query(h_test_query),
      base(h_base),
      base_n(base_n),
      k(k),
      metric(metric),
      max_degree(max_degree) {}

void Test::Run() {
    float *dists, *res_topk_dists, *new_nbr_dists, *graph_dist;
    int *idxs, *res_topk_idxs, *new_nbr_ids, *graph, *graph_deg;
    CN *cand_nbrs_res;
#ifdef algo0
    const int cand_size = std::max(k, max_degree + 1);
#else
    const int cand_size = k + max_degree;
#endif

    CUDA_CHECK(cudaMallocHost(&dists, TEST_BATCH * base_n * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&idxs, TEST_BATCH * base_n * sizeof(int)));
    gpufuncs->compute_knn_dist_test(query, dists, idxs);

    for (int batch_i = 0; batch_i < TEST_BATCH; batch_i++) {
        for (int i = 0; i < base_n; i++) {
            float dist = metric == DIST_METRIC::L2
                             ? compute_dist_l2(query + batch_i * dim, base + i * dim)
                             : compute_dist_ip(query + batch_i * dim, base + i * dim);

            if (fabs(dist - dists[batch_i * base_n + i]) > 1e-6 || idxs[batch_i * base_n + i] != i)
                fo.eprint("compute_knn_dist_test ERROR with batch_i: " + std::to_string(batch_i) +
                          "\ngt(idx, dist): (" + std::to_string(i) + ", " + std::to_string(dist) +
                          ") VS res: (" + std::to_string(idxs[i]) + ", " +
                          std::to_string(dists[i]) + ")");
        }
    }
    fo.print("compute_knn_dist_test PASS");

    for (int batch_i = 1; batch_i < TEST_BATCH; batch_i++) {
        for (int i = 0; i < base_n; i++) {
            dists[batch_i * base_n + i] =
                metric == DIST_METRIC::L2 ? compute_dist_l2(query + batch_i * dim, base + i * dim)
                                          : compute_dist_ip(query + batch_i * dim, base + i * dim);
            idxs[batch_i * base_n + i] = i;
        }
    }
    CUDA_CHECK(cudaMallocHost(&res_topk_idxs, TEST_BATCH * k * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&res_topk_dists, TEST_BATCH * k * sizeof(float)));
    gpufuncs->topk_test(dists, idxs, res_topk_idxs, res_topk_dists);

    for (int batch_i = 0; batch_i < TEST_BATCH; batch_i++) {
        std::vector<std::pair<float, int>> pairs;
        std::string gt, res;
        for (int i = 0; i < base_n; i++)
            pairs.push_back(
                std::make_pair(dists[batch_i * base_n + i], idxs[batch_i * base_n + i]));
        std::sort(pairs.begin(), pairs.end(),
                  [this](const auto &a, auto &b) { return compare_pairs(a, b); });

        for (int i = 0; i < k; i++) {
            gt += std::to_string(pairs[i].second) + "(" + std::to_string(pairs[i].first) + ") ";
            res += std::to_string(res_topk_idxs[batch_i * k + i]) + "(" +
                   std::to_string(res_topk_dists[batch_i * k + i]) + ") ";
        }
        if (gt != res)
            fo.eprint("topk_test ERROR with batch_i: " + std::to_string(batch_i) + "\ngt: " + gt +
                      "\nVS\nres: " + res);
    }
    fo.print("topk_test PASS");

    CUDA_CHECK(cudaMallocHost(&cand_nbrs_res, TEST_BATCH * k * cand_size * sizeof(CN)));
    gpufuncs->cand_compute_test(res_topk_idxs, cand_nbrs_res);

    for (int batch_i = 0; batch_i < TEST_BATCH; batch_i++) {
        for (int pivot_i = 0; pivot_i < k; pivot_i++) {
            const float *pivot_vec = base + res_topk_idxs[batch_i * k + pivot_i] * dim;
#ifdef algo0
            if (pivot_i == 0)
                for (int i = 0; i < cand_size; i++) {
                    const int id = i < k ? res_topk_idxs[batch_i * k + i] : -1;
                    const float *base_vec = base + id * dim;
                    const STATUS status =
                        i == 0 ? Repeated : ((i > 0 && i < k) ? Initial : Discarded);
                    float dist =
                        i == 0 ? (metric == DIST_METRIC::L2 ? L2_CLOSEST : IP_CLOSEST)
                               : (i < k ? (metric == DIST_METRIC::L2
                                               ? compute_dist_l2(pivot_vec, base_vec)
                                               : compute_dist_ip(pivot_vec, base_vec))
                                        : (metric == DIST_METRIC::L2 ? L2_FARTHEST : IP_FARTHEST));
                    if (i < k) dists[i] = dist;

                    const CN &res = cand_nbrs_res[batch_i * k * cand_size + i];
                    if (fabs(res.dist - dist) > 1e-6 || res.id != id || res.idx != i ||
                        res.status != status)
                        fo.eprint(
                            "cand_compute_test ERROR with batch_i: " + std::to_string(batch_i) +
                            ", pivot_i: " + std::to_string(pivot_i) + ", i: " + std::to_string(i) +
                            "\ngt(id, dist, idx, status): " + std::to_string(id) + ", " +
                            std::to_string(dist) + ", " + std::to_string(i) + ", " +
                            std::to_string(status) + "\nVS\nres: " + std::to_string(res.id) + ", " +
                            std::to_string(res.dist) + ", " + std::to_string(res.idx) + ", " +
                            std::to_string(res.status));
                }
            else
                for (int i = 0; i < cand_size; i++) {
                    const int id = i == 0 ? res_topk_idxs[batch_i * k] : -1;
                    const STATUS status = i > 0 ? Discarded : AllRetained;
                    float dist = i == 0 ? dists[pivot_i]
                                        : (metric == DIST_METRIC::L2 ? L2_FARTHEST : IP_FARTHEST);

                    const CN &res =
                        cand_nbrs_res[batch_i * k * cand_size + pivot_i * cand_size + i];
                    if (fabs(res.dist - dist) > 1e-6 || res.id != id || res.idx != i ||
                        res.status != status)
                        fo.eprint(
                            "cand_compute_test ERROR with batch_i: " + std::to_string(batch_i) +
                            ", pivot_i: " + std::to_string(pivot_i) + ", i: " + std::to_string(i) +
                            "\ngt(id, dist, idx, status): " + std::to_string(id) + ", " +
                            std::to_string(dist) + ", " + std::to_string(i) + ", " +
                            std::to_string(status) + "\nVS\nres: " + std::to_string(res.id) + ", " +
                            std::to_string(res.dist) + ", " + std::to_string(res.idx) + ", " +
                            std::to_string(res.status));
                }
#else
            for (int i = 0; i < cand_size; i++) {
                const CN &res = cand_nbrs_res[batch_i * k * cand_size + pivot_i * cand_size + i];
                if (i >= k && (res.id != -1 || res.status != Discarded))
                    fo.eprint("cand_compute_test ERROR with batch_i: " + std::to_string(batch_i) +
                              ", pivot_i: " + std::to_string(pivot_i) +
                              ", i: " + std::to_string(i) +
                              "\ngt(id, dist, idx, status): " + std::to_string(-1) + ", _, _, " +
                              std::to_string(Discarded) + "\nVS\nres: " + std::to_string(res.id) +
                              ", _, _, " + std::to_string(res.status));
                else if (i == pivot_i && res.status != Repeated)
                    fo.eprint("cand_compute_test ERROR with batch_i: " + std::to_string(batch_i) +
                              ", pivot_i: " + std::to_string(pivot_i) +
                              ", i: " + std::to_string(i) +
                              "\ngt(id, dist, idx, status): _, _, _, " + std::to_string(Repeated) +
                              "\nVS\nres: _, _, _, " + std::to_string(res.status));
                else if (i < k && i != pivot_i) {
                    const int id = res_topk_idxs[batch_i * k + i];
                    const float *base_vec = base + id * dim;
                    const float dist =
                        pivot_i == i
                            ? (metric == DIST_METRIC::L2 ? L2_CLOSEST : IP_CLOSEST)
                            : (i < k ? (metric == DIST_METRIC::L2
                                            ? compute_dist_l2(pivot_vec, base_vec)
                                            : compute_dist_ip(pivot_vec, base_vec))
                                     : (metric == DIST_METRIC::L2 ? L2_FARTHEST : IP_FARTHEST));

                    if (fabs(res.dist - dist) > 1e-6 || res.id != id || res.idx != i ||
                        res.status != Initial)
                        fo.eprint(
                            "cand_compute_test ERROR with batch_i: " + std::to_string(batch_i) +
                            ", pivot_i: " + std::to_string(pivot_i) + ", i: " + std::to_string(i) +
                            "\ngt(id, dist, idx, status): " + std::to_string(id) + ", " +
                            std::to_string(dist) + ", " + std::to_string(i) + ", " +
                            std::to_string(Initial) + "\nVS\nres: " + std::to_string(res.id) +
                            ", " + std::to_string(res.dist) + ", " + std::to_string(res.idx) +
                            ", " + std::to_string(res.status));
                }
            }
#endif
        }
    }
    fo.print("cand_compute_test PASS");

#ifdef GIG
    CUDA_CHECK(cudaMallocHost(&graph, base_n * max_degree * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&graph_deg, base_n * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&graph_dist, base_n * max_degree * sizeof(float)));
    gpufuncs->cand_ignore_test(res_topk_idxs, cand_nbrs_res, graph, graph_deg, graph_dist);
#else
    CUDA_CHECK(cudaMallocHost(&new_nbr_ids, TEST_BATCH * k * max_degree * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&new_nbr_dists, TEST_BATCH * k * max_degree * sizeof(float)));
    gpufuncs->cand_ignore_test(cand_nbrs_res, new_nbr_ids, new_nbr_dists);
#endif
    for (int batch_i = 0; batch_i < TEST_BATCH; batch_i++) {
        for (int i = 0; i < k; i++) {
            const int pivot_id = res_topk_idxs[batch_i * k + i];
            int new_nbr_num = 0;
            std::vector<int> new_nbr_ids_gt;
            std::vector<float> new_nbr_dists_gt;
            std::string gts, ress;
            std::vector<std::pair<int, float>> alternatives;

            std::sort(cand_nbrs_res + batch_i * k * cand_size + i * cand_size,
                      cand_nbrs_res + batch_i * k * cand_size + (i + 1) * cand_size,
                      [this](const CN &a, const CN &b) {
                          return metric == DIST_METRIC::L2 ? a.dist < b.dist : a.dist > b.dist;
                      });

            new_nbr_ids_gt.reserve(max_degree);
            new_nbr_dists_gt.reserve(max_degree);
            for (int j = 0; j < cand_size && new_nbr_num < max_degree; j++) {
                const CN &cand_nbr = cand_nbrs_res[batch_i * k * cand_size + i * cand_size + j];
                if (cand_nbr.id == -1 || cand_nbr.status == Repeated) continue;
                bool ignore = false;

                for (int l = 0; l < new_nbr_num; l++) {
                    if (new_nbr_ids_gt[l] == cand_nbr.id) {
                        ignore = true;
                        break;
                    }
                    const float *nbr_vec = base + new_nbr_ids_gt[l] * dim,
                                *cand_vec = base + cand_nbr.id * dim,
                                dist = metric == DIST_METRIC::L2
                                           ? compute_dist_l2(nbr_vec, cand_vec)
                                           : compute_dist_ip(nbr_vec, cand_vec);
                    if (metric == DIST_METRIC::L2 ? cand_nbr.dist > dist : cand_nbr.dist < dist) {
                        alternatives.push_back(std::make_pair(cand_nbr.id, cand_nbr.dist));
                        ignore = true;
                        break;
                    }
                }
                if (!ignore) {
                    new_nbr_ids_gt.push_back(cand_nbr.id);
                    new_nbr_dists_gt.push_back(cand_nbr.dist);
                    new_nbr_num++;
                    gts += std::to_string(cand_nbr.id) + "(" + std::to_string(i) + "," +
                           std::to_string(j) + "," + std::to_string(cand_nbr.dist) + ") ";
                }
            }
            for (int j = 0; j < alternatives.size() && new_nbr_num < max_degree; j++) {
                if (vec_exists(new_nbr_ids_gt, alternatives[j].first)) continue;

                new_nbr_ids_gt.push_back(alternatives[j].first);
                new_nbr_dists_gt.push_back(alternatives[j].second);
                new_nbr_num++;
                gts += "!" + std::to_string(alternatives[j].first) + "(" +
                       std::to_string(alternatives[j].second) + ") ";
            }
            if (new_nbr_num == 0)
                fo.eprint("cand_ignore_test ERROR: new_nbr_num == 0 with batch_i: " +
                          std::to_string(batch_i) + ", i: " + std::to_string(i));

#ifdef GIG
            for (int j = 0; j < graph_deg[pivot_id]; j++) {
                const int idx = pivot_id * max_degree + j, id = graph[idx];
                const float dist = graph_dist[idx];
                ress += std::to_string(id) + "(" + std::to_string(dist) + ") ";
                if (id == -1)
                    fo.eprint("cand_ignore_test ERROR: id == -1 with pivot_id: " + TOS(pivot_id));

                auto find_itr = std::find(new_nbr_ids_gt.begin(), new_nbr_ids_gt.end(), id);
                if (find_itr == new_nbr_ids_gt.end() ||
                    fabs(dist - new_nbr_dists_gt[find_itr - new_nbr_ids_gt.begin()]) > 1e-6)
                    fo.eprint("cand_ignore_test ERROR with batch_i: " + std::to_string(batch_i) +
                              ", i: " + std::to_string(i) + ", j: " + std::to_string(j) + "\ngts(" +
                              std::to_string(new_nbr_ids_gt.size()) + "): " + gts + "\nVS\nres(" +
                              std::to_string(j + 1) + "): " + ress);
                new_nbr_num--;
            }
#else
            for (int j = 0; j < max_degree; j++) {
                const int idx = batch_i * k * max_degree + i * max_degree + j,
                          id = new_nbr_ids[idx];
                const float dist = new_nbr_dists[idx];
                ress += std::to_string(id) + "(" + std::to_string(dist) + ") ";
                if (id == -1) continue;

                auto find_itr = std::find(new_nbr_ids_gt.begin(), new_nbr_ids_gt.end(), id);
                if (find_itr == new_nbr_ids_gt.end() ||
                    fabs(dist - new_nbr_dists_gt[find_itr - new_nbr_ids_gt.begin()]) > 1e-6)
                    fo.eprint("cand_ignore_test ERROR with batch_i: " + std::to_string(batch_i) +
                              ", i: " + std::to_string(i) + ", j: " + std::to_string(j) + "\ngts(" +
                              std::to_string(new_nbr_ids_gt.size()) + "): " + gts + "\nVS\nres(" +
                              std::to_string(j + 1) + "): " + ress);
                new_nbr_num--;
            }
#endif
            if (new_nbr_num > 0)
                fo.eprint("cand_ignore_test ERROR: new_nbr_num > 0 with batch_i: " +
                          std::to_string(batch_i) + ", i: " + std::to_string(i));
        }
    }

    fo.print("cand_ignore_test PASS");

    CUDA_CHECK(cudaFreeHost(dists));
    CUDA_CHECK(cudaFreeHost(idxs));
    CUDA_CHECK(cudaFreeHost(res_topk_idxs));
    CUDA_CHECK(cudaFreeHost(res_topk_dists));
    CUDA_CHECK(cudaFreeHost(cand_nbrs_res));
    CUDA_CHECK(cudaFreeHost(new_nbr_ids));
    CUDA_CHECK(cudaFreeHost(new_nbr_dists));

    fo.print("Test done.");
}

float Test::compute_dist_l2(const float *a, const float *b) {
    float dist = 0;
    for (int i = 0; i < dim; i++) dist += (a[i] - b[i]) * (a[i] - b[i]);
    return dist;
}

float Test::compute_dist_ip(const float *a, const float *b) {
    float dist = 0;
    for (int i = 0; i < dim; i++) dist += a[i] * b[i];
    return dist;
}