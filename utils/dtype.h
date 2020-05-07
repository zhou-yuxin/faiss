#pragma once

#include <stdlib.h>
#include <immintrin.h>

#include <mkl.h>

#include <faiss/IndexIVF.h>
#include <faiss/utils/Heap.h>
#include <faiss/impl/bfp16.h>

#ifdef OPT_DTYPE_UTILS

namespace faiss {

//==================================Convertion================================

inline const float* convert_x_T_impl (size_t, const float* x, float*) {
    return x;
}

template <typename T>
const T* convert_x_T_impl (size_t d, const float* x, T*) {
    T* conv_x = new T[d];
    for (size_t i = 0; i < d; i++) {
        conv_x[i] = static_cast<T> (x[i]);
    }
    return conv_x;
}

template <typename T>
inline const T* convert_x_T (size_t d, const float* x) {
    return convert_x_T_impl (d, x, (T*)nullptr);
}

inline void del_converted_x_T (size_t, const float*) {
}

template <typename T>
inline void del_converted_x_T (size_t, const T* conv_x) {
    delete[] conv_x;
}

template <typename T>
struct Converter_T {

    const size_t d;
    const T* const x;

    Converter_T (size_t d, const float* x):
            d (d), x (convert_x_T<T> (d, x)) {
    }

    ~Converter_T () {
        del_converted_x_T (d, x);
    }

};

//==============================Distance Function=============================

template <typename Tdis, typename T>
Tdis vec_IP_ref_T (const T* x, const T* y, size_t d, Tdis sum = 0) {
    for (size_t i = 0; i < d; i++) {
        sum += static_cast<Tdis> (x[i]) * static_cast<Tdis> (y[i]);
    }
    return sum;
}

inline float vec_IP_ref_T (const float* x, const float* y, size_t d) {
    return vec_IP_ref_T<float> (x, y, d);
}

inline float vec_IP_ref_T (const bfp16_t* x, const bfp16_t* y,
        size_t d) {
    return vec_IP_ref_T<float> (x, y, d);
}

template <typename Tdis, typename T>
Tdis vec_L2Sqr_ref_T (const T* x, const T* y, size_t d, Tdis sum = 0) {
    for (size_t i = 0; i < d; i++) {
        Tdis diff = static_cast<Tdis> (x[i]) - static_cast<Tdis> (y[i]);
        sum += diff * diff;
    }
    return sum;
}

inline float vec_L2Sqr_ref_T (const float* x, const float* y, size_t d) {
    return vec_L2Sqr_ref_T<float> (x, y, d);
}

inline float vec_L2Sqr_ref_T (const bfp16_t* x, const bfp16_t* y,
        size_t d) {
    return vec_L2Sqr_ref_T<float> (x, y, d);
}

#ifdef __SSE4_1__

#define USE_SIMD_128

inline __m128 _mm_loadu_ps_T (const float* x) {
    return _mm_loadu_ps (x);
}

inline __m128 _mm_loadu_ps_T (const bfp16_t* x) {
    return _mm_castsi128_ps (_mm_unpacklo_epi16 (
            _mm_setzero_si128 (),
            _mm_loadl_epi64 ((const __m128i*)x)));
}

template <typename T>
float vec_IP_fp_128b_T (const T* x, const T* y, size_t d,
        __m128 msum = _mm_setzero_ps ()) {
    while (d >= 4) {
        __m128 mx = _mm_loadu_ps_T (x);
        x += 4;
        __m128 my = _mm_loadu_ps_T (y);
        y += 4;
        msum = _mm_add_ps (msum, _mm_mul_ps (mx, my));
        d -= 4;
    }
    msum = _mm_hadd_ps (msum, msum);
    msum = _mm_hadd_ps (msum, msum);
    float sum = _mm_cvtss_f32 (msum);
    return d == 0 ? sum : vec_IP_ref_T<float> (x, y, d, sum);
}

inline float vec_IP_128b_T (const float* x, const float* y, size_t d) {
    return vec_IP_fp_128b_T (x, y, d);
}

inline float vec_IP_128b_T (const bfp16_t* x, const bfp16_t* y,
        size_t d) {
    return vec_IP_fp_128b_T (x, y, d);
}

template <typename T>
float vec_L2Sqr_fp_128b_T (const T* x, const T* y, size_t d,
        __m128 msum = _mm_setzero_ps ()) {
    while (d >= 4) {
        __m128 mx = _mm_loadu_ps_T (x);
        x += 4;
        __m128 my = _mm_loadu_ps_T (y);
        y += 4;
        __m128 mdiff = _mm_sub_ps (mx, my);
        msum = _mm_add_ps (msum, _mm_mul_ps (mdiff, mdiff));
        d -= 4;
    }
    msum = _mm_hadd_ps (msum, msum);
    msum = _mm_hadd_ps (msum, msum);
    float sum = _mm_cvtss_f32 (msum);
    return d == 0 ? sum : vec_L2Sqr_ref_T<float> (x, y, d, sum);
}

inline float vec_L2Sqr_128b_T (const float* x, const float* y, size_t d) {
    return vec_L2Sqr_fp_128b_T (x, y, d);
}

inline float vec_L2Sqr_128b_T (const bfp16_t* x, const bfp16_t* y,
        size_t d) {
    return vec_L2Sqr_fp_128b_T (x, y, d);
}

#endif

#ifdef __AVX2__

#ifndef USE_SIMD_128
#error "SIMD 256 must have SIMD 128 enabled"
#endif

#define USE_SIMD_256

inline __m256 _mm256_loadu_ps_T (const float* x) {
    return _mm256_loadu_ps (x);
}

inline __m256 _mm256_loadu_ps_T (const bfp16_t* x) {
    return
        _mm256_castsi256_ps (
            _mm256_unpacklo_epi16 (
                _mm256_setzero_si256 (),
                _mm256_insertf128_si256 (
                    _mm256_castsi128_si256 (
                        _mm_loadl_epi64 ((const __m128i*)x)),
                    _mm_loadl_epi64 ((const __m128i*)(x + 4)),
                    1)));
}

template <typename T>
float vec_IP_fp_256b_T (const T* x, const T* y, size_t d,
        __m256 msum = _mm256_setzero_ps ()) {
    while (d >= 8) {
        __m256 mx = _mm256_loadu_ps_T (x);
        x += 8;
        __m256 my = _mm256_loadu_ps_T (y);
        y += 8;
        msum = _mm256_add_ps (msum, _mm256_mul_ps (mx, my));
        d -= 8;
    }
    __m128 msum2 = _mm256_extractf128_ps (msum, 1);
    msum2 = _mm_add_ps (msum2, _mm256_extractf128_ps (msum, 0));
    return vec_IP_fp_128b_T (x, y, d, msum2);
}

inline float vec_IP_256b_T (const float* x, const float* y, size_t d) {
    return vec_IP_fp_256b_T (x, y, d);
}

inline float vec_IP_256b_T (const bfp16_t* x, const bfp16_t* y,
        size_t d) {
    return vec_IP_fp_256b_T (x, y, d);
}

template <typename T>
float vec_L2Sqr_fp_256b_T (const T* x, const T* y, size_t d,
        __m256 msum = _mm256_setzero_ps ()) {
    while (d >= 8) {
        __m256 mx = _mm256_loadu_ps_T (x);
        x += 8;
        __m256 my = _mm256_loadu_ps_T (y);
        y += 8;
        __m256 mdiff = _mm256_sub_ps (mx, my);
        msum = _mm256_add_ps (msum, _mm256_mul_ps (mdiff, mdiff));
        d -= 8;
    }
    __m128 msum2 = _mm256_extractf128_ps (msum, 1);
    msum2 = _mm_add_ps (msum2, _mm256_extractf128_ps (msum, 0));
    return vec_L2Sqr_fp_128b_T (x, y, d, msum2);
}

inline float vec_L2Sqr_256b_T (const float* x, const float* y, size_t d) {
    return vec_L2Sqr_fp_256b_T (x, y, d);
}

inline float vec_L2Sqr_256b_T (const bfp16_t* x, const bfp16_t* y,
        size_t d) {
    return vec_L2Sqr_fp_256b_T (x, y, d);
}

#endif

#if defined (__AVX512F__) && defined(__AVX512DQ__) && defined(__AVX512BW__) \
        && defined(__AVX512VL__)

#ifndef USE_SIMD_256
#error "SIMD 512 must have SIMD 256 enabled"
#endif
#define USE_SIMD_512

inline __m512 _mm512_loadu_ps_T (const float* x) {
    return _mm512_loadu_ps (x);
}

inline __m512 _mm512_loadu_ps_T (const bfp16_t* x) {
    return
    _mm512_castsi512_ps (
        _mm512_unpacklo_epi16 (
            _mm512_setzero_si512 (),
            _mm512_inserti64x4 (
                _mm512_castsi256_si512 (
                    _mm256_inserti32x4 (
                        _mm256_castsi128_si256 (
                            _mm_loadl_epi64 ((const __m128i*)x)),
                        _mm_loadl_epi64 ((const __m128i*)(x + 4)),
                        1)),
                _mm256_inserti32x4 (
                    _mm256_castsi128_si256 (
                        _mm_loadl_epi64 ((const __m128i*)(x + 8))),
                    _mm_loadl_epi64 ((const __m128i*)(x + 12)),
                    1),
                1)));
}

template <typename T>
float vec_IP_fp_512b_T (const T* x, const T* y, size_t d,
        __m512 msum = _mm512_setzero_ps ()) {
    while (d >= 16) {
        __m512 mx = _mm512_loadu_ps_T (x);
        x += 16;
        __m512 my = _mm512_loadu_ps_T (y);
        y += 16;
        msum = _mm512_add_ps (msum, _mm512_mul_ps (mx, my));
        d -= 16;
    }
    __m256 msum2 = _mm512_extractf32x8_ps (msum, 1);
    msum2 = _mm256_add_ps (msum2, _mm512_extractf32x8_ps (msum, 0));
    return vec_IP_fp_256b_T (x, y, d, msum2);
}

inline float vec_IP_512b_T (const float* x, const float* y, size_t d) {
    return vec_IP_fp_512b_T (x, y, d);
}

inline float vec_IP_512b_T (const bfp16_t* x, const bfp16_t* y,
        size_t d) {
    return vec_IP_fp_512b_T (x, y, d);
}

template <typename T>
float vec_L2Sqr_fp_512b_T (const T* x, const T* y, size_t d,
        __m512 msum = _mm512_setzero_ps ()) {
    while (d >= 16) {
        __m512 mx = _mm512_loadu_ps_T (x);
        x += 16;
        __m512 my = _mm512_loadu_ps_T (y);
        y += 16;
        __m512 mdiff = _mm512_sub_ps (mx, my);
        msum = _mm512_add_ps (msum, _mm512_mul_ps (mdiff, mdiff));
        d -= 16;
    }
    __m256 msum2 = _mm512_extractf32x8_ps (msum, 1);
    msum2 = _mm256_add_ps (msum2, _mm512_extractf32x8_ps (msum, 0));
    return vec_L2Sqr_fp_256b_T (x, y, d, msum2);
}

inline float vec_L2Sqr_512b_T (const float* x, const float* y, size_t d) {
    return vec_L2Sqr_fp_512b_T (x, y, d);
}

inline float vec_L2Sqr_512b_T (const bfp16_t* x, const bfp16_t* y,
        size_t d) {
    return vec_L2Sqr_fp_512b_T (x, y, d);
}

#endif

#if defined (USE_SIMD_512)

template <typename T>
inline float vec_IP_T (const T* x, const T* y, size_t d) {
    return vec_IP_512b_T (x, y, d);
}

template <typename T>
inline float vec_L2Sqr_T (const T* x, const T* y, size_t d) {
    return vec_L2Sqr_512b_T (x, y, d);
}

#elif defined (USE_SIMD_256)

template <typename T>
inline float vec_IP_T (const T* x, const T* y, size_t d) {
    return vec_IP_256b_T (x, y, d);
}

template <typename T>
inline float vec_L2Sqr_T (const T* x, const T* y, size_t d) {
    return vec_L2Sqr_256b_T (x, y, d);
}

#elif defined (USE_SIMD_128)

template <typename T>
inline float vec_IP_T (const T* x, const T* y, size_t d) {
    return vec_IP_128b_T (x, y, d);
}

template <typename T>
inline float vec_L2Sqr_T (const T* x, const T* y, size_t d) {
    return vec_L2Sqr_128b_T (x, y, d);
}

#else

template <typename T>
inline float vec_IP_T (const T* x, const T* y, size_t d) {
    return vec_IP_ref_T (x, y, d);
}

template <typename T>
inline float vec_L2Sqr_T (const T* x, const T* y, size_t d) {
    return vec_L2Sqr_ref_T (x, y, d);
}

#endif

}

#endif

#ifdef OPT_FLAT_DTYPE

#define FLAT_BATCH_THRESHOLD    4

namespace faiss {

//=================================KNN Routine================================

template <typename T, typename D>
void knn_less_better_alone_T (const T* x, const T* y, size_t d,
        size_t nx, size_t ny, float_maxheap_array_t* res, D& distance) {
    size_t k = res->k;
    size_t check_period = InterruptCallback::get_period_hint (ny * d);
    check_period *= omp_get_max_threads ();
    for (size_t i0 = 0; i0 < nx; i0 += check_period) {
        size_t i1 = std::min (i0 + check_period, nx);
#pragma omp parallel for
        for (size_t i = i0; i < i1; i++) {
            const T* x_i = x + i * d;
            const T* y_j = y;
            float* simi = res->get_val (i);
            int64_t* idxi = res->get_ids (i);
            maxheap_heapify (k, simi, idxi);
            for (size_t j = 0; j < ny; j++) {
                float dis = distance (i, j, x_i, y_j, d);
                if (dis < simi[0]) {
                    maxheap_pop (k, simi, idxi);
                    maxheap_push (k, simi, idxi, dis, j);
                }
                y_j += d;
            }
            maxheap_reorder (k, simi, idxi);
        }
        InterruptCallback::check ();
    }
}

template <typename T, typename D>
void knn_greater_better_alone_T (const T* x, const T* y, size_t d,
        size_t nx, size_t ny, float_minheap_array_t* res, D& distance) {
    size_t k = res->k;
    size_t check_period = InterruptCallback::get_period_hint (ny * d);
    check_period *= omp_get_max_threads ();
    for (size_t i0 = 0; i0 < nx; i0 += check_period) {
        size_t i1 = std::min (i0 + check_period, nx);
#pragma omp parallel for
        for (size_t i = i0; i < i1; i++) {
            const T* x_i = x + i * d;
            const T* y_j = y;
            float* simi = res->get_val (i);
            int64_t* idxi = res->get_ids (i);
            minheap_heapify (k, simi, idxi);
            for (size_t j = 0; j < ny; j++) {
                float dis = distance (i, j, x_i, y_j, d);
                if (dis > simi[0]) {
                    minheap_pop (k, simi, idxi);
                    minheap_push (k, simi, idxi, dis, j);
                }
                y_j += d;
            }
            minheap_reorder (k, simi, idxi);
        }
        InterruptCallback::check ();
    }
}

template <typename T>
inline void knn_inner_product_alone_T (const T* x, const T* y, size_t d,
        size_t nx, size_t ny, float_minheap_array_t* res) {

    struct IP {

        inline float operator () (size_t /*ix*/, size_t /*jy*/,
                const T* xi, const T* yj, size_t d) const {
            return vec_IP_T (xi, yj, d);
        }

    }
    distance;
    knn_greater_better_alone_T (x, y, d, nx, ny, res, distance);
}

template <typename T>
inline void knn_L2Sqr_alone_T (const T* x, const T* y, size_t d,
        size_t nx, size_t ny, float_maxheap_array_t* res) {

    struct L2Sqr {

        inline float operator () (size_t /*ix*/, size_t /*jy*/,
                const T* xi, const T* yj, size_t d) const {
            return vec_L2Sqr_T (xi, yj, d);
        }

    }
    distance;
    knn_less_better_alone_T (x, y, d, nx, ny, res, distance);
}

template <typename T>
inline void knn_L2Sqr_expand_alone_T (const T* x, const T* y, size_t d,
        size_t nx, size_t ny, float_maxheap_array_t* res,
        const float* y_norm) {

    struct L2SqrExpand {

        const float* y_norm_sqr;

        inline float operator () (size_t /*ix*/, size_t jy,
                const T* xi, const T* yj, size_t d) const {
            return y_norm_sqr[jy] - 2 * vec_IP_T (xi, yj, d);
        }

    }
    distance = {
        .y_norm_sqr = y_norm,
    };
    knn_less_better_alone_T (x, y, d, nx, ny, res, distance);
}

template <typename T>
inline void knn_inner_product_batch_T (const T* x, const T* y, size_t d,
        size_t nx, size_t ny, float_minheap_array_t* res) {
    knn_inner_product_alone_T (x, y, d, nx, ny, res);
}

template <typename T>
inline void knn_L2Sqr_batch_T (const T* x, const T* y, size_t d,
        size_t nx, size_t ny, float_maxheap_array_t* res) {
    knn_L2Sqr_alone_T (x, y, d, nx, ny, res);
}

template <typename T>
inline void knn_L2Sqr_expand_batch_T (const T* x, const T* y, size_t d,
        size_t nx, size_t ny, float_maxheap_array_t* res,
        const float* y_norm) {
    knn_L2Sqr_expand_alone_T (x, y, d, nx, ny, res, y_norm);
}

template <typename H, typename D>
void knn_batch_T (const float* x, const float* y,
        size_t d, size_t nx, size_t ny, H* heap, D& distance) {
    heap->heapify ();
    if (nx == 0 || ny == 0) {
        return;
    }
    float* distances = new float [nx * ny];
    distance (x, y, d, nx, ny, distances);
    heap->addn (ny, distances, 0, 0, nx);
    delete[] distances;
    InterruptCallback::check ();
    heap->reorder ();
}

inline void knn_inner_product_batch_T (const float* x, const float* y,
        size_t d, size_t nx, size_t ny, float_minheap_array_t* res) {

    struct IP {

        inline void operator () (const float* x, const float* y,
                size_t d, size_t nx, size_t ny, float* distances) const {
            cblas_sgemm (CblasRowMajor, CblasNoTrans, CblasTrans, nx, ny, d,
                    1.0f, x, d, y, d, 0.0f, distances, ny);
        }

    }
    distance;
    knn_batch_T (x, y, d, nx, ny, res, distance);
}

inline void knn_L2Sqr_expand_batch_T (const float* x, const float* y,
        size_t d, size_t nx, size_t ny, float_maxheap_array_t* res,
        const float* y_norm) {

    struct L2SqrExpand {

        const float* y_norm;

        inline void operator () (const float* x, const float* y,
                size_t d, size_t nx, size_t ny, float* distances) const {
            float* distances_i = distances;
            size_t step = ny * sizeof(float);
            for (size_t i = 0; i < nx; i++) {
                memcpy (distances_i, y_norm, step);
                distances_i += ny;
            }
            cblas_sgemm (CblasRowMajor, CblasNoTrans, CblasTrans, nx, ny, d,
                    -2.0f, x, d, y, d, 1.0f, distances, ny);
            }
    }
    distance = {
        .y_norm = y_norm,
    };
    knn_batch_T (x, y, d, nx, ny, res, distance);
}

template <typename T>
inline void knn_inner_product_T (const T* x, const T* y, size_t d,
        size_t nx, size_t ny, float_minheap_array_t* res) {
    if (nx < FLAT_BATCH_THRESHOLD) {
        knn_inner_product_alone_T (x, y, d, nx, ny, res);
    }
    else {
        knn_inner_product_batch_T (x, y, d, nx, ny, res);
    }
}

template <typename T>
inline void knn_L2Sqr_T (const T* x, const T* y, size_t d,
        size_t nx, size_t ny, float_maxheap_array_t* res) {
    if (nx < FLAT_BATCH_THRESHOLD) {
        knn_L2Sqr_alone_T (x, y, d, nx, ny, res);
    }
    else {
        knn_L2Sqr_batch_T (x, y, d, nx, ny, res);
    }
}

template <typename T>
inline void knn_L2Sqr_expand_T (const T* x, const T* y, size_t d,
        size_t nx, size_t ny, float_maxheap_array_t* res,
        const float* y_norm) {
    if (nx < FLAT_BATCH_THRESHOLD) {
        knn_L2Sqr_expand_alone_T (x, y, d, nx, ny, res, y_norm);
    }
    else {
        knn_L2Sqr_expand_batch_T (x, y, d, nx, ny, res, y_norm);
    }
}

#ifdef OPT_FLAT_MKL_PACK
inline void knn_L2Sqr_expand_pack (const float* x, const float* y, size_t d,
        size_t nx, size_t ny, float_maxheap_array_t* res,
        const float* y_norm) {

    struct L2SqrExpand {

        const float* y_norm;

        inline void operator () (const float* x, const float* y,
                size_t d, size_t nx, size_t ny, float* distances) const {
            float* distances_i = distances;
            size_t step = ny * sizeof(float);
            for (size_t i = 0; i < nx; i++) {
                memcpy (distances_i, y_norm, step);
                distances_i += ny;
            }
		    cblas_sgemm_compute (CblasRowMajor, CblasNoTrans, CblasPacked,
                    nx, ny, d, x, d, y, 0, 1.0f, distances, ny);
            }
    }
    distance = {
        .y_norm = y_norm,
    };
    knn_batch_T (x, y, d, nx, ny, res, distance);
}
#endif

}

#endif

#ifdef OPT_IVFFLAT_DTYPE

#define SCANNER_USE_BATCH       false

namespace faiss {

//===========================Inverted List Scanner============================

template <typename T>
class InvertedListScanner_T : public InvertedListScanner {

    using idx_t = InvertedListScanner::idx_t;

protected:
    size_t d;
    size_t code_size;
    bool store_pairs;
    const T* converted_x;
    idx_t list_no;

public:
    InvertedListScanner_T (size_t d, bool store_pairs):
            d (d), code_size (sizeof(T) * d), store_pairs (store_pairs),
            converted_x(nullptr), list_no(-1) {
    }

    virtual ~InvertedListScanner_T () {
        if (converted_x) {
            del_converted_x_T (d, converted_x);
        }
    }

    virtual void set_query (const float* query) override {
        if (converted_x) {
            del_converted_x_T (d, converted_x);
        }
        converted_x = convert_x_T<T> (d, query);
    }

    virtual void set_list (idx_t lidx, float) override {
        list_no = lidx;
    }

    virtual float distance_to_code (const uint8_t*) const override {
        FAISS_THROW_MSG ("not implemented");
    }

    virtual size_t scan_codes (size_t list_size, const uint8_t* codes,
            const idx_t* ids, float* simi, idx_t* idxi, size_t k)
            const = 0;

};

template <typename T, typename C, typename D>
class AloneInvertedListScanner_T : public InvertedListScanner_T<T> {

    using idx_t = InvertedListScanner::idx_t;
    using Scanner = InvertedListScanner_T<T>;

private:
    D* distance;

public:
    AloneInvertedListScanner_T (size_t d, bool store_pairs, D* distance):
            Scanner (d, store_pairs), distance (distance) {
    }

    virtual ~AloneInvertedListScanner_T () {
        delete distance;
    }

    virtual size_t scan_codes (size_t list_size, const uint8_t* codes,
            const idx_t* ids, float* simi, idx_t* idxi,
            size_t k) const override {
        size_t nup = 0;
        for (size_t i = 0; i < list_size; i++) {
            float dis = (*distance) (Scanner::list_no, i,Scanner::converted_x,
                    (const T*)codes, Scanner::d);
            codes += Scanner::code_size;
            if (C::cmp (simi[0], dis)) {
                heap_pop<C> (k, simi, idxi);
                int64_t id = Scanner::store_pairs ?
                        lo_build (Scanner::list_no, i) :
                        ids[i];
                heap_push<C> (k, simi, idxi, dis, id);
                nup++;
            }
        }
        return nup;
    }

};

template <typename T>
InvertedListScanner* get_IP_alone_scanner_T (size_t d, bool store_pairs) {

    struct IP {

        inline float operator () (size_t /*ilist*/, size_t /*jy*/,
                const T* x, const T* yj, size_t d) const {
            return vec_IP_T (x, yj, d);
        }

    }
    *distance = new IP;
    return new AloneInvertedListScanner_T<T, CMin<float, int64_t>, IP> (d,
            store_pairs, distance);
}

template <typename T>
InvertedListScanner* get_L2Sqr_alone_scanner_T (size_t d, bool store_pairs) {

    struct L2Sqr {

        inline float operator () (size_t /*ilist*/, size_t /*jy*/,
                const T* x, const T* yj, size_t d) const {
            return vec_L2Sqr_T (x, yj, d);
        }

    }
    *distance = new L2Sqr;
    return new AloneInvertedListScanner_T<T, CMax<float, int64_t>, L2Sqr> (d,
            store_pairs, distance);
}

template <typename T, typename TNorm>
InvertedListScanner* get_L2Sqr_expand_alone_scanner_T (size_t d,
        bool store_pairs, const TNorm y_norm) {

    struct L2SqrExpand {

        const TNorm y_norm;

        inline float operator () (size_t ilist, size_t jy,
                const T* x, const T* yj, size_t d) {
            return y_norm [ilist] [jy] - 2 * vec_IP_T (x, yj, d);
        }

    }
    *distance = new L2SqrExpand {
        .y_norm = y_norm,
    };
    return new AloneInvertedListScanner_T<T, CMax<float, int64_t>,
            L2SqrExpand> (d, store_pairs, distance);
}

template <typename T, typename C, typename D>
class BatchInvertedListScanner_T : public InvertedListScanner_T<T> {

    using idx_t = InvertedListScanner::idx_t;
    using Scanner = InvertedListScanner_T<T>;

private:
    D* distance;

public:
    BatchInvertedListScanner_T (size_t d, bool store_pairs, D* distance):
            Scanner (d, store_pairs), distance (distance) {
    }

    virtual ~BatchInvertedListScanner_T () {
        delete distance;
    }

    virtual size_t scan_codes (size_t list_size, const uint8_t* codes,
            const idx_t* ids, float* simi, idx_t* idxi,
            size_t k) const override {
        float* distances = new float [list_size];
        (*distance) (Scanner::converted_x, Scanner::list_no, list_size,
                (const T*)codes, Scanner::d, distances);
        size_t nup = 0;
        for (size_t i = 0; i < list_size; i++) {
            float dis = distances [i];
            if (C::cmp (simi[0], dis)) {
                heap_pop<C> (k, simi, idxi);
                int64_t id = Scanner::store_pairs ?
                        lo_build (Scanner::list_no, i) : ids[i];
                heap_push<C> (k, simi, idxi, dis, id);
                nup++;
            }
        }
        delete[] distances;
        return nup;
    }

};

template <typename T>
inline InvertedListScanner* get_IP_batch_scanner_T (size_t d, bool store_pairs,
        T*) {
    return get_IP_alone_scanner_T<T> (d, store_pairs);
}

inline InvertedListScanner* get_IP_batch_scanner_T (size_t d, bool store_pairs,
        float*) {

    struct IP {

        inline void operator () (const float* x, size_t /*ilist*/,
                size_t list_size, const float* y, size_t d, float* distances)
                const {
            cblas_sgemv (CblasRowMajor, CblasNoTrans, list_size, d, 1.0f,
                    y, d, x, 1, 0.0f, distances, 1);
        }

    }
    *distance = new IP;
    return new BatchInvertedListScanner_T<float, CMin<float, int64_t>, IP> (d,
            store_pairs, distance);
}

template <typename T>
inline InvertedListScanner* get_L2Sqr_batch_scanner_T (size_t d,
        bool store_pairs, T*) {
    return get_L2Sqr_alone_scanner_T<T> (d, store_pairs);
}

template <typename T, typename TNorm>
inline InvertedListScanner* get_L2Sqr_expand_batch_scanner_T (size_t d,
        bool store_pairs, const TNorm y_norm, T*) {
    return get_L2Sqr_expand_alone_scanner_T<T> (d, store_pairs,
            y_norm);
}

template <typename TNorm>
inline InvertedListScanner* get_L2Sqr_expand_batch_scanner_T (size_t d,
        bool store_pairs, const TNorm y_norm, float*) {

    struct L2SqrExpand {

        const TNorm y_norm;

        inline void operator () (const float* x, size_t ilist,
                size_t list_size, const float* y, size_t d, float* distances)
                const {
            memcpy (distances, &(y_norm [ilist] [0]),
                    list_size * sizeof(float));
            cblas_sgemv (CblasRowMajor, CblasNoTrans, list_size, d, -2.0f,
                    y, d, x, 1, 1.0f, distances, 1);
        }

    }
    *distance = new L2SqrExpand {
        .y_norm = y_norm,
    };
    return new BatchInvertedListScanner_T<float, CMax<float, int64_t>,
            L2SqrExpand> (d, store_pairs, distance);
}

template <typename T>
inline InvertedListScanner* get_IP_scanner_T (size_t d, bool store_pairs) {
    if (!SCANNER_USE_BATCH) {
        return get_IP_alone_scanner_T<T> (d, store_pairs);
    }
    else {
        return get_IP_batch_scanner_T (d, store_pairs, (T*)nullptr);
    }
}

template <typename T>
inline InvertedListScanner* get_L2Sqr_scanner_T (size_t d, bool store_pairs) {
    if (!SCANNER_USE_BATCH) {
        return get_L2Sqr_alone_scanner_T<T> (d, store_pairs);
    }
    else {
        return get_L2Sqr_batch_scanner_T (d, store_pairs, (T*)nullptr);
    }
}

template <typename T, typename TNorm>
inline InvertedListScanner* get_L2Sqr_expand_scanner_T (size_t d,
        bool store_pairs, const TNorm y_norm) {
    if (!SCANNER_USE_BATCH) {
        return get_L2Sqr_expand_alone_scanner_T<T> (d, store_pairs, y_norm);
    }
    else {
        return get_L2Sqr_expand_batch_scanner_T (d, store_pairs, y_norm,
                (T*)nullptr);
    }
}

}

#endif