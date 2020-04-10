#pragma once

#ifdef OPT_DTYPE_UTILS

#include <stdlib.h>
#include <immintrin.h>

#include <faiss/utils/Heap.h>

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

//============================Inner Product Function==========================

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
    if (d > 0) {
        sum += vec_IP_ref_T<float> (x, y, d);
    }
    return sum;
}

inline float vec_IP_128b_T (const float* x, const float* y, size_t d) {
    return vec_IP_fp_128b_T (x, y, d);
}

inline float vec_IP_128b_T (const bfp16_t* x, const bfp16_t* y,
        size_t d) {
    return vec_IP_fp_128b_T (x, y, d);
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
    return _mm256_castsi256_ps (
            _mm256_unpacklo_epi16 (_mm256_setzero_si256 (),
            _mm256_insertf128_si256 (_mm256_castsi128_si256 (
            _mm_loadl_epi64 ((const __m128i*)x)),
            _mm_loadl_epi64 ((const __m128i*)(x + 4)), 1)));
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

#endif

#if defined (USE_SIMD_256)

template <typename T>
inline float vec_IP_T (const T* x, const T* y, size_t d) {
    return vec_IP_256b_T (x, y, d);
}

#elif defined (USE_SIMD_128)

template <typename T>
inline float vec_IP_T (const T* x, const T* y, size_t d) {
    return vec_IP_128b_T (x, y, d);
}

#else

template <typename T>
inline float vec_IP_T (const T* x, const T* y, size_t d) {
    return vec_IP_ref_T (x, y, d);
}

#endif

//============================K Nearest Neighbor Routine======================

template <typename T, typename D>
void knn_less_better_T (const T* x, const T* y, size_t d,
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
void knn_greater_better_T (const T* x, const T* y, size_t d,
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
void knn_inner_product_T (const T* x, const T* y, size_t d,
        size_t nx, size_t ny, float_minheap_array_t* res) {
    struct IP {

        inline float operator ()(size_t, size_t, const T* xi, const T* yj,
                size_t d) {
            return vec_IP_T (xi, yj, d);
        }

    };
    IP ip;
    knn_greater_better_T (x, y, d, nx, ny, res, ip);
}

template <typename T>
void knn_L2sqr_T (const T* x, const T* y, const float* y_norm, size_t d,
        size_t nx, size_t ny, float_maxheap_array_t* res) {
    struct L2Sqr {

        const float* y_norm;

        inline float operator ()(size_t, size_t j, const T* xi, const T* yj,
                size_t d) {
            return y_norm[j] - 2 * vec_IP_T (xi, yj, d);
        }

    };
    L2Sqr l2sqr;
    l2sqr.y_norm = y_norm;
    knn_less_better_T (x, y, d, nx, ny, res, l2sqr);
}

}

#endif