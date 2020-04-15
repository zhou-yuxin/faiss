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
        } else {
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
                Converter_T<T> converter (d, yi);
                offset = invlists->add_entry (list_no, id,
                        (const uint8_t*)converter.x);
                if (metric_type == METRIC_L2_EXPAND) {
                    norms [list_no].push_back (vec_IP_T (yi, yi, d));
                }
                else if (metric_type == METRIC_COSINE) {
                    norms [list_no].push_back (1.0f / std::sqrt (
                            vec_IP_T (yi, yi, d)));
                }
                n_add++;
            } else {
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
        if (metric_type == METRIC_INNER_PRODUCT) {

            struct IP {

                inline float operator () (size_t /*ilist*/, size_t /*jy*/,
                        const T* x, const T* yj, size_t d) const {
                    return vec_IP_T (x, yj, d);
                }
            }
            *distance = new IP;
            return new IVFFlatScanner_T<IP, CMin<float, int64_t>>
                    (d, store_pairs, distance);
        } else if (metric_type == METRIC_L2) {

            struct L2Sqr {

                inline float operator () (size_t /*ilist*/, size_t /*jy*/,
                        const T* x, const T* yj, size_t d) const {
                    return vec_L2sqr_T (x, y, d);
                }

            }
            *distance = new L2Sqr {
                .y_norm_sqr = norms.data(),
            };
            return new IVFFlatScanner_T<L2Sqr, CMax<float, int64_t>>
                    (d, store_pairs, distance);
        } else if (metric_type == METRIC_L2_EXPAND) {

            struct L2SqrExpand {

                const std::vector<float>* y_norm_sqr;

                inline float operator () (size_t ilist, size_t jy,
                        const T* x, const T* yj, size_t d) const {
                    return y_norm_sqr [ilist] [jy] - 2 * vec_IP_T (x, yj, d);
                }

            }
            *distance = new L2SqrExpand {
                .y_norm_sqr = norms.data(),
            };
            return new IVFFlatScanner_T<L2SqrExpand, CMax<float, int64_t>>
                    (d, store_pairs, distance);
        } else if (metric_type == METRIC_COSINE) {

            struct Cosine {

                const std::vector<float>* y_norm_recip;

                inline float operator () (size_t ilist, size_t jy,
                        const T* x, const T* yj, size_t d) const {
                    return vec_IP_T (x, yj, d) * y_norm_recip [ilist] [jy];
                }

            }
            *distance = new Cosine {
                .y_norm_recip = norms.data(),
            };
            return new IVFFlatScanner_T<Cosine, CMin<float, int64_t>>
                    (d, store_pairs, distance);
        } else {
            FAISS_THROW_FMT("unsupported metric type: %d", (int)metric_type);
        }
    }

private:
    template <typename Dis, typename Comp>
    struct IVFFlatScanner_T: InvertedListScanner {
        size_t d;
        size_t code_size;
        bool store_pairs;
        const T* converted_x;
        idx_t list_no;
        Dis* distance;

        IVFFlatScanner_T (size_t d, bool store_pairs, Dis* distance):
                d (d), code_size (sizeof(T) * d), store_pairs (store_pairs),
                converted_x(nullptr), list_no(-1), distance (distance) {
        }

        virtual ~IVFFlatScanner_T () {
            if (converted_x) {
                del_converted_x_T (d, converted_x);
            }
            delete distance;
        }

        void set_query (const float* query) override {
            if (converted_x) {
                del_converted_x_T (d, converted_x);
            }
            converted_x = convert_x_T<T> (d, query);
        }

        void set_list (idx_t lidx, float) override {
            list_no = lidx;
        }

        float distance_to_code (const uint8_t*) const override {
            FAISS_THROW_MSG ("not implemented");
        }

        size_t scan_codes (size_t list_size, const uint8_t* codes,
                const idx_t* ids, float* simi, idx_t* idxi,
                size_t k) const override {
            size_t nup = 0;
            for (size_t i = 0; i < list_size; i++) {
                float dis = (*distance) (list_no, i,
                        converted_x, (const T*)codes, d);
                codes += code_size;
                if (Comp::cmp (simi[0], dis)) {
                    heap_pop<Comp> (k, simi, idxi);
                    int64_t id = store_pairs ? lo_build (list_no, i) : ids[i];
                    heap_push<Comp> (k, simi, idxi, dis, id);
                    nup++;
                }
            }
            return nup;
        }

    };

};

}

#endif