#pragma once

#ifdef OPT_FLAT_DTYPE

#include <vector>

#include <faiss/Index.h>

#include <faiss/impl/FaissAssert.h>

#include <faiss/utils/dtype.h>

namespace faiss {

template <typename T>
struct IndexFlat_T: Index {

    std::vector<T> base;
    std::vector<float> norms;
#ifdef OPT_FLAT_MKL_PACK
    std::vector<uint8_t> packed_base;
#endif

    IndexFlat_T() {
    }

    IndexFlat_T (idx_t d, MetricType metric = METRIC_L2):
            Index (d, metric) {
    }

    void add (idx_t n, const float* y) override {
        const float* yi = y;
        if (metric_type == METRIC_PROJECTION) {
            for (size_t i = 0; i < n; i++) {
                float rnorm = 1.0f / std::sqrt (vec_IP_T (yi, yi, d));
                for (size_t j = 0; j < d; j++) {
                    base.push_back (static_cast<T> (yi[j] * rnorm));
                }
                yi += d;
            }
        }
        else {
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < d; j++) {
                    base.push_back (static_cast<T> (yi[j]));
                }
                if (metric_type == METRIC_L2_EXPAND) {
                    norms.push_back (vec_IP_T (yi, yi, d));
                }
                yi += d;
            }
        }
        ntotal += n;
    }

    void reset () override {
        base.clear ();
        norms.clear ();
        ntotal = 0;
    }

#ifdef OPT_FLAT_MKL_PACK
    void pack_base (bool enable) {
        if (enable && packed_base.empty ()) {
            if (typeid (T) != typeid (float)) {
                FAISS_THROW_MSG ("only float type supports MKL PACK");
            }
            size_t pack_size = cblas_sgemm_pack_get_size (CblasBMatrix,
				1, ntotal, d);
		    packed_base.resize (pack_size);
		    cblas_sgemm_pack (CblasRowMajor, CblasBMatrix, CblasTrans,
				1, ntotal, d, 1.0f, (float*) (base.data ()), d,
                (float*) (packed_base.data ()));
        }
        else if (!enable && !packed_base.empty ()) {
            packed_base.clear ();
        }
    }
#endif

    void search (idx_t n, const float* x,
            idx_t k, float* distances, idx_t* labels) const override {
        Converter_T<T> converter (n * d, x);
        if (metric_type == METRIC_INNER_PRODUCT ||
                metric_type == METRIC_PROJECTION) {
            float_minheap_array_t res = {
                size_t(n), size_t(k), labels, distances};
            knn_inner_product_T (converter.x, base.data (), d, n,
                    ntotal, &res);
        } else if (metric_type == METRIC_L2) {
            float_maxheap_array_t res = {
                size_t(n), size_t(k), labels, distances};
            knn_L2Sqr_T (converter.x, base.data(), d, n, ntotal, &res);
        } else if (metric_type == METRIC_L2_EXPAND) {
            float_maxheap_array_t res = {
                size_t(n), size_t(k), labels, distances};
            knn_L2Sqr_expand_T (converter.x, base.data(), d, n, ntotal,
                    &res, norms.data());
        }
        else {
            FAISS_THROW_FMT("unsupported metric type: %d", (int)metric_type);
        }
    }

};

}

#endif