#pragma once

#ifdef OPT_FLAT_DTYPE

#include <vector>

#include <faiss/Index.h>

#include <faiss/impl/FaissAssert.h>

#include <faiss/utils/dtype.h>
#include <faiss/utils/distances.h>

namespace faiss {

template <typename T>
struct IndexFlat_T: Index {

    std::vector<T> base;
    std::vector<float> norms;

    IndexFlat_T() {
    }

    IndexFlat_T (idx_t d, MetricType metric = METRIC_L2):
            Index (d, metric) {
    }

    void add (idx_t n, const float* y) override {
        const float* yi = y;
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < d; j++) {
                base.push_back (static_cast<T> (yi[j]));
            }
            if (metric_type == METRIC_L2) {
                norms.push_back (fvec_norm_L2sqr (yi, d));
            }
            yi += d;
        }
        ntotal += n;
    }

    void reset () override {
        base.clear ();
        norms.clear ();
        ntotal = 0;
    }

    void search (idx_t n, const float* x,
            idx_t k, float* distances, idx_t* labels) const override {
        Converter_T<T> converter (n * d, x);
        if (metric_type == METRIC_INNER_PRODUCT) {
            float_minheap_array_t res = {
                size_t(n), size_t(k), labels, distances};
            knn_inner_product_T (converter.x, base.data (), d, n,
                    ntotal, &res);
        } else if (metric_type == METRIC_L2) {
            float_maxheap_array_t res = {
                size_t(n), size_t(k), labels, distances};
            knn_L2sqr_T (converter.x, base.data(), norms.data(), d, n,
                    ntotal, &res);
        }
        else {
            FAISS_THROW_FMT("unsupported metric type: %d", (int)metric_type);
        }
    }

};

}

#endif