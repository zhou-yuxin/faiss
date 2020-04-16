#pragma once

#ifdef OPT_IVFFLAT_DTYPE

#include <vector>

#include <faiss/IndexIVF.h>

#include <faiss/impl/FaissAssert.h>

#include <faiss/utils/dtype.h>

namespace faiss {

template <typename T>
struct IndexIVFFlat_T: IndexIVF {

    std::vector<std::vector<float>> norms;

    IndexIVFFlat_T () {
    }

    IndexIVFFlat_T (Index* quantizer, size_t d, size_t nlist,
            MetricType metric = METRIC_L2):
            IndexIVF (quantizer, d, nlist, sizeof(T) * d, metric) {
        norms.resize (nlist);
    }

    void add_with_ids (idx_t n, const float* y, const idx_t* yids) override {
        add_core (n, y, yids, nullptr);
    }

    virtual void add_core (idx_t n, const float* y, const int64_t* yids,
            const int64_t* precomputed_idx) {
        FAISS_THROW_IF_NOT (is_trained);
        assert (invlists);
        direct_map.check_can_add (yids);
        const int64_t* idx;
        ScopeDeleter<int64_t> del;
        if (precomputed_idx) {
            idx = precomputed_idx;
        }
        else {
            int64_t* idx0 = new int64_t[n];
            del.set (idx0);
            quantizer->assign (n, y, idx0);
            idx = idx0;
        }
        int64_t n_add = 0;
        for (size_t i = 0; i < n; i++) {
            idx_t id = yids ? yids[i] : ntotal + i;
            idx_t list_no = idx [i];
            size_t offset;
            if (list_no >= 0) {
                const float* yi = y + i * d;
                if (metric_type == METRIC_PROJECTION) {
                    float rnorm = 1.0f / std::sqrt (vec_IP_T (yi, yi, d));
                    T* scaled_y = new T [d];
                    for (size_t j = 0; j < d; j++) {
                        scaled_y [j] = static_cast<T> (yi [j] * rnorm);
                    }
                    offset = invlists->add_entry (list_no, id,
                            (const uint8_t*)scaled_y);
                    delete[] scaled_y;
                }
                else {
                    Converter_T<T> converter (d, yi);
                    offset = invlists->add_entry (list_no, id,
                            (const uint8_t*)converter.x);
                    if (metric_type == METRIC_L2_EXPAND) {
                        norms [list_no].push_back (vec_IP_T (yi, yi, d));
                    }
                }
                n_add++;
            }
            else {
                offset = 0;
            }
            direct_map.add_single_id (id, list_no, offset);
        }
        if (verbose) {
            printf ("IndexIVFFlat_T::add_core: added %ld / %ld vectors\n",
                    n_add, n);
        }
        ntotal += n;
    }

    void encode_vectors(idx_t, const float*, const idx_t*,
            uint8_t*, bool = false) const override {
        FAISS_THROW_MSG ("not implemented");
    }

    InvertedListScanner *get_InvertedListScanner (bool store_pairs)
            const override {
        if (metric_type == METRIC_INNER_PRODUCT ||
                metric_type == METRIC_PROJECTION) {
            return get_IP_scanner_T<T> (d, store_pairs);
        }
        else if (metric_type == METRIC_L2) {
            return get_L2Sqr_scanner_T<T> (d, store_pairs);
        }
        else if (metric_type == METRIC_L2_EXPAND) {
            return get_L2Sqr_expand_scanner_T<T> (d, store_pairs,
                    norms.data());
        }
        else {
            FAISS_THROW_FMT("unsupported metric type: %d", (int)metric_type);
        }
    }

};

}

#endif