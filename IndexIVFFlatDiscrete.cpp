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

    template <typename T>
    struct Format {
        union {
            T value;
            uint8_t bytes[sizeof (T)];
        };
    };

    const size_t d;
    const size_t nlist;
    const size_t raw_vector_size;
    const size_t vector_size;
    std::vector<uint8_t>* lists;

    DiscreteLists (size_t d, size_t nlist, size_t vector_size) :
            d (d), nlist (nlist), raw_vector_size (vector_size),
            vector_size (vector_size + sizeof (float) + sizeof (label_t)),
            lists (new std::vector<uint8_t> [nlist]) {
    }

    ~DiscreteLists () {
        delete[] lists;
    }

    void add (size_t ilist, size_t iy_in_list, const float* y) {
        std::vector<uint8_t>* list = lists + ilist;
        Format<float> error;
        error.value = sqrtf (add_raw (list, y));
        for (size_t i = 0; i < sizeof (float); i++) {
            list->emplace_back (error.bytes[i]);
        }
        Format<label_t> iy;
        iy.value = iy_in_list;
        for (size_t i = 0; i < sizeof (label_t); i++) {
            list->emplace_back (iy.bytes[i]);
        }
    }

    void reset () {
        for (size_t i = 0; i < nlist; i++) {
            lists[i].clear ();
        }
    }

    size_t calculate (const float* x, size_t ilist, float dis_scale,
            float* dis_lower_bounds, ID* ids) const {
        std::vector<uint8_t>* list = lists + ilist;
        assert (list->size () % vector_size == 0);
        size_t n = list->size () / vector_size;
        const uint8_t* y = list->data ();
        for (size_t i = 0; i < n; i++) {
            const uint8_t* yi = y + i * vector_size;
            const float* error = (float*) (yi + raw_vector_size);
            const label_t* iy = (label_t*) (error + 1);
            dis_lower_bounds[i] = (sqrtf (get_distance (x, yi)) - *error)
                    * dis_scale;
            ID* id = ids + i;
            id->ilist = ilist;
            id->iy_in_list = *iy;
        }
        return n;
    }

    virtual bool is_in_range (float min_v, float max_v) const = 0;

    virtual float add_raw (std::vector<uint8_t>* list, const float* y) = 0;

    virtual float get_distance (const float* x, const uint8_t* y) const = 0;

};

struct Int8Lists : DiscreteLists {

    Int8Lists (size_t d, size_t nlist) : DiscreteLists (d, nlist, d) {
    }

    bool is_in_range (float min_v, float max_v) const override {
        return -128.0f <= min_v && max_v <= 127.0f;
    }

    float add_raw (std::vector<uint8_t>* list, const float* y) override {
        float error = 0.0f;
        for (size_t i = 0; i < d; i++) {
            float v = y[i];
            float disc_v = roundf (v);
            assert (-128.0f <= disc_v && disc_v <= 127.0f);
            list->emplace_back (disc_v);
            float diff = v - disc_v;
            error += diff * diff;
        }
        return error;
    }

    float get_distance (const float* x, const uint8_t* y) const override {
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

struct Bfp16Lists : DiscreteLists {

    Bfp16Lists (size_t d, size_t nlist) : DiscreteLists (d, nlist, d * 2) {
    }

    bool is_in_range (float, float) const override {
        return true;
    }

    float add_raw (std::vector<uint8_t>* list, const float* y) override {
        float error = 0.0f;
        for (size_t i = 0; i < d; i++) {
            float v = y[i];
            Format<float> disc;
            disc.value = v;
            list->emplace_back (disc.bytes[2]);
            list->emplace_back (disc.bytes[3]);
            disc.bytes[0] = 0;
            disc.bytes[1] = 0;
            float diff = v - disc.value;
            error += diff * diff;
        }
        return error;
    }

    float get_distance (const float* x, const uint8_t* y) const override {
        size_t rd = d;
#ifdef  __SSE4_2__
#ifdef __AVX2__
#if defined(__AVX512F__) && defined(__AVX512DQ__)
        __m512 msum16 = _mm512_setzero_ps ();
        while (rd >= 16) {
            __m512 mx = _mm512_loadu_ps (x);
            __m512 my = _mm512_castsi512_ps (_mm512_slli_epi32 (
                    _mm512_cvtepi16_epi32 (_mm256_loadu_si256 (
                    (const __m256i*) (y))), 16));
            __m512 mdiff = _mm512_sub_ps (mx, my);
            msum16 = _mm512_add_ps (msum16, _mm512_mul_ps (mdiff, mdiff));
            x += 16;
            y += 32;
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
            __m256 my = _mm256_castsi256_ps (_mm256_slli_epi32 (
                    _mm256_cvtepi16_epi32(
                    _mm_loadu_si128 ((const __m128i*) (y))), 16));
            __m256 mdiff = _mm256_sub_ps (mx, my);
            msum8 = _mm256_add_ps (msum8, _mm256_mul_ps (mdiff, mdiff));
            x += 8;
            y += 16;
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
            __m128 my = _mm_castsi128_ps (_mm_slli_epi32 (_mm_cvtepi16_epi32(
                            _mm_loadl_epi64 ((const __m128i*) (y))), 16));
            __m128 mdiff = _mm_sub_ps (mx, my);
            msum4 = _mm_add_ps (msum4, _mm_mul_ps (mdiff, mdiff));
            x += 4;
            y += 8;
            rd -= 4;
        }
        msum4 = _mm_hadd_ps (msum4, msum4);
        msum4 = _mm_hadd_ps (msum4, msum4);
        float sum = _mm_cvtss_f32 (msum4);
#else
        float sum = 0.0f;
#endif
        Format<float> v;
        v.bytes[0] = 0;
        v.bytes[1] = 0;
        for (size_t i = 0; i < rd; i++) {
            v.bytes[2] = y[0];
            v.bytes[3] = y[1];
            float diff = x[i] - v.value;
            sum += diff * diff;
            y += 2;
        }
        return sum;
    }

};

struct IVFFlatDiscreteSpace {

    const size_t d;
    const float k;
    const float b;
    const float dis_scale;
    std::vector<std::unique_ptr<DiscreteLists>> lists_array;

    IVFFlatDiscreteSpace (size_t d, size_t nlist, float k, float b,
            const char* type_exp) :
            d (d), k (k), b (b), dis_scale (1.0f / k) {
        char* exp_buf = new char [strlen (type_exp) + 1];
        std::unique_ptr<char[]> exp_buf_del (exp_buf);
        strcpy (exp_buf, type_exp);
        char* saved_ptr;
        for (char* tok = strtok_r (exp_buf, "+", &saved_ptr); tok;
                tok = strtok_r (nullptr, "+", &saved_ptr)) {
            DiscreteLists* lists;
            if (strcasecmp (tok, "int8") == 0) {
                lists = new Int8Lists (d, nlist);
            }
            else if (strcasecmp (tok, "bfp16") == 0) {
                lists = new Bfp16Lists (d, nlist);
            }
            else {
                FAISS_THROW_FMT ("unsupported type: '%s'", tok);
            }
            lists_array.emplace_back (lists);
        }
    }

    void add (size_t ilist, size_t iy_in_list, const float* y) {
        float* yt = new float [d];
        std::unique_ptr<float[]> yt_del (yt);
        float min_v = std::numeric_limits<float>::max ();
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
        for (std::unique_ptr<DiscreteLists>& lists : lists_array) {
            if (lists->is_in_range (min_v, max_v)) {
                lists->add (ilist, iy_in_list, yt);
                return;
            }
        }
        FAISS_THROW_MSG ("<y> cannot be added into any DiscreteLists");
    }

    void reset () {
        for (std::unique_ptr<DiscreteLists>& lists : lists_array) {
            lists->reset ();
        }
    }

    size_t calculate (const float* x, size_t ilist,
            float* dis_lower_bounds, ID* ids) const {
        float* xt = new float [d];
        std::unique_ptr<float[]> xt_del (xt);
        for (size_t i = 0; i < d; i++) {
            xt[i] = k * x[i] + b;
        }
        size_t ncalculate = 0;
        for (const std::unique_ptr<DiscreteLists>& lists : lists_array) {
            ncalculate += lists->calculate (xt, ilist, dis_scale,
                    dis_lower_bounds + ncalculate, ids + ncalculate);
        }
        return ncalculate;
    }

};

IndexIVFFlatDiscrete::IndexIVFFlatDiscrete () :
        quantizer (nullptr), own_quantizer (false), ivlists (nullptr),
        disc (nullptr), use_residual (false), nprobe (1), chunk_size (0),
        parallel_mode (0) {
}

IndexIVFFlatDiscrete::IndexIVFFlatDiscrete (Index* quantizer,
        size_t d, size_t nlist, MetricType metric, const char* disc_exp) :
        Index (d, metric), quantizer (quantizer), own_quantizer (false),
        ivlists (nullptr), disc (nullptr), use_residual (false), nprobe (1),
        chunk_size (0), parallel_mode (0) {
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
    float* residual = new float [d];
    std::unique_ptr<float[]> residual_del (residual);
    for (size_t i = 0; i < nlist; i++) {
        size_t n = ivlists->list_size (i);
        const float* y = (const float*) (ivlists->get_codes (i));
        for (size_t j = 0; j < n; j++) {
            if (use_residual) {
                quantizer->compute_residual (y, residual, i);
                disc->add (i, j, residual);
            }
            else {
                disc->add (i, j, y);
            }
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
    std::unique_ptr<idx_t[]> ilists_del (ilists);
    quantizer->assign (n, y, ilists);
    float* residual = new float [d];
    std::unique_ptr<float[]> residual_del (residual);
    for (idx_t i = 0; i < n; i++) {
        idx_t ilist = ilists[i];
        const float* yi = y + i * d;
        size_t iy_in_list = ivlists->add_entry (ilist, ntotal + i,
                (uint8_t*) (yi));
        if (use_residual) {
            quantizer->compute_residual (y, residual, ilist);
            disc->add (ilist, iy_in_list, residual);
        }
        else {
            disc->add (ilist, iy_in_list, yi);
        }
    }
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
    std::unique_ptr<idx_t[]> ilists_del (ilists);
    quantizer->search (n, x, nprobe, quantizer_distances, ilists);
    delete[] quantizer_distances;
    ivlists->prefetch_lists (ilists, n * nprobe);
    Fetcher fetcher (ivlists);
    if (parallel_mode == 0) {
        #pragma omp parallel if (n > 1)
        {
            float* residual = new float [d];
            std::unique_ptr<float[]> residual_del (residual);
            #pragma omp for schedule (dynamic)
            for (idx_t i = 0; i < n; i++) {
                const float* xi = x + i * d;
                const idx_t* ilists_i = ilists + i * nprobe;
                size_t total_size = 0;
                for (size_t j = 0; j < nprobe; j++) {
                    total_size += ivlists->list_size (ilists_i[j]);
                }
                float* dis_lower_bounds = new float [total_size];
                std::unique_ptr<float[]> dis_del (dis_lower_bounds);
                ID* ids = new ID [total_size];
                std::unique_ptr<ID[]> ids_del (ids);
                float* dis_lower_bounds_j = dis_lower_bounds;
                ID* ids_j = ids;
                for (size_t j = 0; j < nprobe; j++) {
                    idx_t ilist = ilists_i[j];
                    size_t ncalculate;
                    if (use_residual) {
                        quantizer->compute_residual (xi, residual, ilist);
                        ncalculate = disc->calculate (residual, ilist,
                                dis_lower_bounds_j, ids_j);
                    }
                    else {
                        ncalculate = disc->calculate (xi, ilist,
                                dis_lower_bounds_j, ids_j);
                    }
                    assert (ncalculate == ivlists->list_size (ilist));
                    dis_lower_bounds_j += ncalculate;
                    ids_j += ncalculate;
                }
                DiscreteRefiner<ID>::refine (d, xi, total_size,
                        dis_lower_bounds, ids, &fetcher,
                        chunk_size >= k ? chunk_size : 4 * k, k,
                        distances + i * k, labels + i * k);
            }
        }
    }
    else {
        FAISS_THROW_FMT ("unsupported parallel mode: %d", parallel_mode);
    }
}

void IndexIVFFlatDiscrete::parse_disc_exp (const char* exp, size_t nlist) {
    char* type_exp = new char [strlen (exp) + 1];
    std::unique_ptr<char[]> exp_del (type_exp);
    bool residual;
    float k, b;
    if (sscanf (exp, "x%f%f>>%s", &k, &b, type_exp) == 3) {
        residual = false;
    }
    else if (sscanf (exp, "Rx%f%f>>%s", &k, &b, type_exp) == 3) {
        residual = true;
    }
    else {
        FAISS_THROW_FMT ("invalid expression: '%s'", exp);
    }
    IVFFlatDiscreteSpace* space = new IVFFlatDiscreteSpace (d, nlist, k, b,
            type_exp);
    if (disc) {
        delete disc;
    }
    disc = space;
    disc_exp.assign (exp);
    use_residual = residual;
}

}

#endif