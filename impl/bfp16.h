#pragma once

#include <stdint.h>

class BFP16 {

private:
    uint16_t storage;

    typedef union {
        float fp32;
        struct {
            uint32_t mantissa : 23;
            uint32_t exponent : 8;
            uint32_t sign : 1;
        };
        uint16_t u16s[2];
    } Formater;

public:
    inline BFP16 () {
    }

    inline BFP16 (float x) {
        *this = x;
    }

    inline void operator = (float x) {
        Formater f;
        f.fp32 = x;
        // if (f.u16s[0] & 0x8000) {
        //     Formater g;
        //     g.fp32 = x;
        //     g.mantissa = 0;
        //     g.fp32 /= 128.0f;
        //     f.fp32 += g.fp32;
        // }
        storage = f.u16s[1];
    }

    inline operator float () const {
        Formater f;
        f.u16s[0] = 0;
        f.u16s[1] = storage;
        return f.fp32;
    }

    inline uint16_t raw() const {
        return storage;
    }
};

typedef BFP16 bfp16_t;