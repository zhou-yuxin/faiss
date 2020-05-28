#pragma once

#include <cassert>

#include <stdint.h>
#include <immintrin.h>

namespace faiss {

struct DiscreteDistances {

    static float int8 (size_t d, const float* x, const int8_t* y) {
#ifdef  __SSE4_2__
#ifdef __AVX2__
#if defined(__AVX512F__) && defined(__AVX512DQ__)
        __m512 msum16 = _mm512_setzero_ps ();
        while (d >= 16) {
            __m512 mx = _mm512_loadu_ps (x);
            __m512 my = _mm512_cvtepi32_ps (_mm512_cvtepi8_epi32 (
                    _mm_loadu_si128 ((const __m128i*) (y))));
            __m512 mdiff = _mm512_sub_ps (mx, my);
            msum16 = _mm512_add_ps (msum16, _mm512_mul_ps (mdiff, mdiff));
            x += 16;
            y += 16;
            d -= 16;
        }
        __m256 msum8 = _mm512_extractf32x8_ps (msum16, 1);
        msum8 = _mm256_add_ps (msum8, _mm512_extractf32x8_ps (msum16, 0));
        if (d >= 8) {
#else
        __m256 msum8 = _mm256_setzero_ps ();
        while (d >= 8) {
#endif
            __m256 mx = _mm256_loadu_ps (x);
            __m256 my = _mm256_cvtepi32_ps (_mm256_cvtepi8_epi32 (
                    _mm_loadl_epi64 ((const __m128i*) (y))));
            __m256 mdiff = _mm256_sub_ps (mx, my);
            msum8 = _mm256_add_ps (msum8, _mm256_mul_ps (mdiff, mdiff));
            x += 8;
            y += 8;
            d -= 8;
        }
        __m128 msum4 = _mm256_extractf128_ps(msum8, 1);
        msum4 = _mm_add_ps (msum4, _mm256_extractf128_ps(msum8, 0));
        if (d >= 4) {
#else
        __m128 msum4 = _mm_setzero_ps ();
        while (d >= 4) {
#endif
            __m128 mx = _mm_loadu_ps (x);
            __m128 my = _mm_cvtepi32_ps (_mm_cvtepi8_epi32 (
                    _mm_castps_si128 (_mm_load1_ps ((const float*) (y)))));
            __m128 mdiff = _mm_sub_ps (mx, my);
            msum4 = _mm_add_ps (msum4, _mm_mul_ps (mdiff, mdiff));
            x += 4;
            y += 4;
            d -= 4;
        }
        msum4 = _mm_hadd_ps (msum4, msum4);
        msum4 = _mm_hadd_ps (msum4, msum4);
        float sum = _mm_cvtss_f32 (msum4);
#else
        float sum = 0.0f;
#endif
        for (size_t i = 0; i < d; i++) {
            float diff = x[i] - float (y[i]);
            sum += diff * diff;
        }
        assert (sum >= 0.0f);
        return sum;
    }

};

}