#ifdef OPT_DISCRETIZATION

#include <memory>
#include <vector>
#include <cassert>
#include <climits>

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#include <faiss/Clustering.h>
#include <faiss/InvertedLists.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/IndexIVFFlatDiscrete.h>
#include <faiss/utils/DiscreteRefiner.h>

namespace faiss {

using label_t = uint32_t;

struct ID {
    label_t ilist;
    label_t iy_in_list;
};

struct Fetcher : DiscreteRefiner<ID>::Fetcher {

    const InvertedLists* ivlists;

    Fetcher (const InvertedLists* ivlists) : ivlists (ivlists) {
    }

    DiscreteRefiner<ID>::idx_t get_label (const ID& id) const override {
        return ivlists->get_single_id (id.ilist, id.iy_in_list);
    }

    const float* get_vector (const ID& id) const override {
        return (float*) (ivlists->get_single_code (id.ilist,
                id.iy_in_list));
    }

};

struct DiscreteLists {

    const size_t d;
    const size_t nlist;

    DiscreteLists (size_t d, size_t nlist) : d (d), nlist (nlist) {
    }

    virtual ~DiscreteLists () {
    }

    virtual bool is_in_range (float min_v, float max_v) const = 0;

    virtual void add (size_t ilist, size_t iy_in_list, const float* y) = 0;

    virtual void reset () = 0;

    virtual size_t calculate (const float* x, size_t ilist, float dis_scale,
            float* dis_lower_bounds, ID* ids) const = 0;

};

struct Int8Lists : DiscreteLists {

    template <typename T>
    struct Format {
        union {
            T value;
            int8_t int8s[sizeof (T)];
        };
    };

    const size_t vector_size;
    std::vector<int8_t>* lists;

    Int8Lists (size_t d, size_t nlist) :
            DiscreteLists (d, nlist),
            vector_size (d + sizeof (float) + sizeof (label_t)),
            lists (new std::vector<int8_t> [nlist]) {
    }

    ~Int8Lists () {
        delete[] lists;
    }

    bool is_in_range (float min_v, float max_v) const override {
        return -128.0f <= min_v && max_v <= 127.0f;
    }

    void add (size_t ilist, size_t iy_in_list, const float* y) override {
        std::vector<int8_t>* list = lists + ilist;
        Format<float> error;
        error.value = 0.0f;
        for (size_t i = 0; i < d; i++) {
            float v = y[i];
            float disc_v = roundf (v);
            assert (-128.0f <= disc_v && disc_v <= 127.0f);
            list->emplace_back (disc_v);
            float diff = v - disc_v;
            error.value += diff * diff;
        }
        error.value = sqrtf (error.value);
        for (size_t i = 0; i < sizeof (float); i++) {
            list->emplace_back (error.int8s[i]);
        }
        Format<label_t> iy;
        iy.value = iy_in_list;
        for (size_t i = 0; i < sizeof (label_t); i++) {
            list->emplace_back (iy.int8s[i]);
        }
    }

    void reset () override {
        for (size_t i = 0; i < nlist; i++) {
            lists[i].clear ();
        }
    }

    size_t calculate (const float* x, size_t ilist, float dis_scale,
            float* dis_lower_bounds, ID* ids) const override {
        std::vector<int8_t>* list = lists + ilist;
        assert (list->size () % vector_size == 0);
        size_t n = list->size () / vector_size;
        const int8_t* y = list->data ();
        for (size_t i = 0; i < n; i++) {
            const int8_t* yi = y + i * vector_size;
            const Format<float>* error = (Format<float>*) (yi + d);
            const Format<label_t>* iy = (Format<label_t>*) (error + 1);
            dis_lower_bounds[i] = (get_distance (x, yi) - error->value)
                    * dis_scale;
            ID* id = ids + i;
            id->ilist = ilist;
            id->iy_in_list = iy->value;
        }
        return n;
    }

    float get_distance (const float* x, const int8_t* y) const {
        size_t rd = d;
#ifdef  __SSE4_2__
#ifdef __AVX2__
#if defined(__AVX512F__) && defined(__AVX512DQ__)
        __m512 msum16 = _mm512_setzero_ps ();
        while (rd >= 16) {
            __m512 mx = _mm512_loadu_ps (x);
            __m512 my = _mm512_cvtepi32_ps (_mm512_cvtepi8_epi32 (
                    _mm_loadu_si128 ((const __m128i*) (y))));
            __m512 mdiff = _mm512_sub_ps (mx, my);
            msum16 = _mm512_add_ps (msum16, _mm512_mul_ps (mdiff, mdiff));
            x += 16;
            y += 16;
            rd -= 16;
        }
        __m256 msum8 = _mm512_extractf32x8_ps (msum16, 1);
        msum8 = _mm256_add_ps (msum8, _mm512_extractf32x8_ps (msum16, 0));
        if (rd >= 8) {
#else
        __m256 msum8 = _mm256_setzero_ps ();
        while (rd >= 8) {
#endif
            __m256 mx = _mm256_loadu_ps (x);
            __m256 my = _mm256_cvtepi32_ps (_mm256_cvtepi8_epi32 (
                    _mm_loadl_epi64 ((const __m128i*) (y))));
            __m256 mdiff = _mm256_sub_ps (mx, my);
            msum8 = _mm256_add_ps (msum8, _mm256_mul_ps (mdiff, mdiff));
            x += 8;
            y += 8;
            rd -= 8;
        }
        __m128 msum4 = _mm256_extractf128_ps(msum8, 1);
        msum4 = _mm_add_ps (msum4, _mm256_extractf128_ps(msum8, 0));
        if (rd >= 4) {
#else
        __m128 msum4 = _mm_setzero_ps ();
        while (rd >= 4) {
#endif
            __m128 mx = _mm_loadu_ps (x);
            __m128 my = _mm_cvtepi32_ps (_mm_cvtepi8_epi32 (
                    _mm_castps_si128 (_mm_load1_ps ((const float*) (y)))));
            __m128 mdiff = _mm_sub_ps (mx, my);
            msum4 = _mm_add_ps (msum4, _mm_mul_ps (mdiff, mdiff));
            x += 4;
            y += 4;
            rd -= 4;
        }
        msum4 = _mm_hadd_ps (msum4, msum4);
        msum4 = _mm_hadd_ps (msum4, msum4);
        float sum = _mm_cvtss_f32 (msum4);
#else
        float sum = 0.0f;
#endif
        for (size_t i = 0; i < rd; i++) {
            float diff = x[i] - float (y[i]);
            sum += diff * diff;
        }
        return sum;
    }

};

    // struct BFP16 {

    //     struct Format {
    //         union {
    //             float fp32;
    //             uint16_t uint16s[2];
    //         };
    //     };

    //     uint16_t storage;
    // };

struct IVFFlatDiscreteSpace {

    const size_t d;
    const float k;
    const float b;
    const float dis_scale;
    std::vector<DiscreteLists*> lists_array;

    IVFFlatDiscreteSpace (size_t d, size_t nlist, float k, float b) :
            d (d), k (k), b (b), dis_scale (1.0f / k) {
        lists_array.emplace_back (new Int8Lists (d, nlist));
    }

    ~IVFFlatDiscreteSpace () {
        for (DiscreteLists* lists : lists_array) {
            delete lists;
        }
    }

    void add (size_t ilist, size_t iy_in_list, const float* y) {
        float* yt = new float [d];
        std::unique_ptr<float> yt_del (yt);
        float min_v = std::numeric_limits<float>::max();
        float max_v = -min_v;
        for (size_t i = 0; i < d; i++) {
            float v = k * y[i] + b;
            yt[i] = v;
            if (v < min_v) {
                min_v = v;
            }
            if (v > max_v) {
                max_v = v;
            }
        }
        assert (min_v <= max_v);
        for (DiscreteLists* lists : lists_array) {
            if (lists->is_in_range (min_v, max_v)) {
                lists->add (ilist, iy_in_list, yt);
                return;
            }
        }
        FAISS_THROW_MSG ("<y> cannot be added into any DiscreteLists");
    }

    void reset () {
        for (DiscreteLists* lists : lists_array) {
            lists->reset ();
        }
    }

    size_t calculate (const float* x, size_t ilist,
            float* dis_lower_bounds, ID* ids) const {
        float* xt = new float [d];
        for (size_t i = 0; i < d; i++) {
            xt[i] = k * x[i] + b;
        }
        size_t ncalculate = 0;
        for (DiscreteLists* lists : lists_array) {
            ncalculate += lists->calculate (xt, ilist, dis_scale,
                    dis_lower_bounds + ncalculate, ids + ncalculate);
        }
        delete[] xt;
        return ncalculate;
    }

};

IndexIVFFlatDiscrete::IndexIVFFlatDiscrete () :
        quantizer (nullptr), own_quantizer (false), ivlists (nullptr),
        disc (nullptr), nprobe (1), chunk_size (0), parallel_mode (0) {
}

IndexIVFFlatDiscrete::IndexIVFFlatDiscrete (Index* quantizer,
        size_t d, size_t nlist, MetricType metric, const char* disc_exp) :
        Index (d, metric), quantizer (quantizer), own_quantizer (false),
        ivlists (nullptr), disc (nullptr), nprobe (1), chunk_size (0),
        parallel_mode (0) {
    FAISS_THROW_IF_NOT (d == quantizer->d);
    if (metric != METRIC_L2) {
        FAISS_THROW_MSG ("only L2 is supported");
    }
    parse_disc_exp (disc_exp, nlist);
    ivlists = new ArrayInvertedLists (nlist, d * sizeof (float));
    is_trained = quantizer->is_trained && (quantizer->ntotal == nlist);
}

IndexIVFFlatDiscrete::~IndexIVFFlatDiscrete () {
    if (quantizer && own_quantizer) {
        delete quantizer;
    }
    if (ivlists) {
        delete ivlists;
    }
    if (disc) {
        delete disc;
    }
}

void IndexIVFFlatDiscrete::rebuild_discrete_space (const char* exp) {
    size_t nlist = ivlists->nlist;
    parse_disc_exp (exp, nlist);
    for (size_t i = 0; i < nlist; i++) {
        size_t list_size = ivlists->list_size (i);
        const float* y = (const float*) (ivlists->get_codes (i));
        for (size_t j = 0; j < list_size; j++) {
            disc->add (i, j, y);
            y += d;
        }
    }
}

void IndexIVFFlatDiscrete::train (idx_t n, const float* y) {
    size_t nlist = ivlists->nlist;
    ClusteringParameters cp;
    cp.niter = 10;
    if (!is_trained) {
        Clustering clus (d, nlist, cp);
        quantizer->reset ();
        clus.train (n, y, *quantizer);
        quantizer->is_trained = true;
        is_trained = quantizer->is_trained && (quantizer->ntotal == nlist);
    }
}

void IndexIVFFlatDiscrete::add (idx_t n, const float* y) {
    FAISS_THROW_IF_NOT (is_trained);
    idx_t* ilists = new idx_t [n];
    quantizer->assign (n, y, ilists);
    for (idx_t i = 0; i < n; i++) {
        idx_t ilist = ilists[i];
        const float* yi = y + i * d;
        size_t iy_in_list = ivlists->add_entry (ilist, ntotal + i,
                (uint8_t*) (yi));
        disc->add (ilist, iy_in_list, yi);
    }
    delete[] ilists;
    ntotal += n;
}

void IndexIVFFlatDiscrete::reset () {
    ivlists->reset ();
    disc->reset ();
    ntotal = 0;
}

void IndexIVFFlatDiscrete::search (idx_t n, const float* x,
        idx_t k, float* distances, idx_t* labels) const {
    float* quantizer_distances = new float[n * nprobe];
    idx_t* ilists = new idx_t [n * nprobe];
    std::unique_ptr<idx_t> ilists_del (ilists);
    quantizer->search (n, x, nprobe, quantizer_distances, ilists);
    delete[] quantizer_distances;
    ivlists->prefetch_lists (ilists, n * nprobe);
    Fetcher fetcher (ivlists);
    if (parallel_mode == 0) {
        #pragma omp parallel if (n > 1)
        {
            #pragma omp for schedule (dynamic)
            for (idx_t i = 0; i < n; i++) {
                const float* xi = x + i * d;
                const idx_t* ilists_i = ilists + i * nprobe;
                size_t total_size = 0;
                for (size_t j = 0; j < nprobe; j++) {
                    total_size += ivlists->list_size (ilists_i[j]);
                }
                float* dis_lower_bounds = new float [total_size];
                ID* ids = new ID [total_size];
                float* dis_lower_bounds_j = dis_lower_bounds;
                ID* ids_j = ids;
                for (size_t j = 0; j < nprobe; j++) {
                    size_t ncalculate = disc->calculate (xi, ilists_i[j],
                            dis_lower_bounds_j, ids_j);
                    assert (ncalculate == ivlists->list_size (ilists_i[j]));
                    dis_lower_bounds_j += ncalculate;
                    ids_j += ncalculate;
                }
                DiscreteRefiner<ID>::refine (d, xi, total_size,
                        dis_lower_bounds, ids, &fetcher,
                        chunk_size >= k ? chunk_size : 4 * k, k,
                        distances + i * k, labels + i * k);
                delete[] dis_lower_bounds;
                delete[] ids;
            }
        }
    }
    else {
        FAISS_THROW_FMT ("unsupported parallel mode: %d", parallel_mode);
    }
}

void IndexIVFFlatDiscrete::parse_disc_exp (const char* exp, size_t nlist) {
    std::vector<char> type_buf;
    type_buf.resize (strlen (exp) + 1);
    char* type_exp = type_buf.data ();
    float k, b;
    if (sscanf (exp, "x%f+%f>>%s", &k, &b, type_exp) != 3) {
        FAISS_THROW_FMT ("invalid expression: '%s'", exp);
    }
    if (disc) {
        delete disc;
    }
    if (strcasecmp (type_exp, "int8") == 0) {
        disc = new IVFFlatDiscreteSpace (d, nlist, k, b);
    }
    else if (strcasecmp (type_exp, "uint8") == 0) {
        disc = new IVFFlatDiscreteSpace (d, nlist, k, b - 128.0f);
    }
    // else if (strcasecmp (type_exp, "int4") == 0) {
    //     disc = new Uint4Space (d, nlist, k, b + 8.0f);
    // }
    // else if (strcasecmp (type_exp, "uint4") == 0) {
    //     disc = new Uint4Space (d, nlist, k, b);
    // }
    else {
        FAISS_THROW_FMT ("unsupported type: '%s'", type_exp);
    }
    disc_exp.assign (exp);
}

}

#endif