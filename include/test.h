#pragma once

#include "uni.h"
#include "gpufuncs.cuh"
#include "fileout.h"

namespace efanna2e
{
    class Test
    {
    public:
        Test(GPUFuncs *gpufuncs, const float *h_test_query, const float *h_base,
             const int dim, const int base_n, const int k, DIST_METRIC metric,
             const int max_degree);
        void Run();

    private:
        GPUFuncs *gpufuncs;
        const float *query, *base;
        const int dim, base_n, k, max_degree;
        const DIST_METRIC metric;

        float compute_dist_l2(const float *a, const float *b);
        float compute_dist_ip(const float *a, const float *b);
        bool compare_pairs(
            const std::pair<float, int> &a, const std::pair<float, int> &b);
    };
}