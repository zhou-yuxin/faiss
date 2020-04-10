#pragma once

#ifdef OPT_DTYPE_UTILS

#include <stdint.h>

namespace faiss {

class BFP16 {

private:
    union {
        uint16_t u16;
        struct {
            uint16_t mantissa : 7;
            uint16_t exponent : 8;
            uint16_t sign : 1;
        };
    }
    bfp16;

    typedef union {
        float value;
        struct {
            uint32_t mantissa : 23;
            uint32_t exponent : 8;
            uint32_t sign : 1;
        };
        uint16_t u16s[2];
    }
    FP32;

public:
    inline BFP16 () {
    }

    inline BFP16 (float x) {
        *this = x;
    }

    inline void operator = (float x) {
        FP32 fp32;
        fp32.value = x;
        if (fp32.u16s[0] & 0x8000) {
            fp32.u16s[0] = 0;
            fp32.value *= (129.0f / 128.0f);
        }
        bfp16.u16 = fp32.u16s[1];
    }

    inline operator float () const {
        FP32 fp32;
        fp32.u16s[0] = 0;
        fp32.u16s[1] = bfp16.u16;
        return fp32.value;
    }

};

typedef BFP16 bfp16_t;

}

#endif