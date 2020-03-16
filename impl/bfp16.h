#pragma once

#include <stdint.h>

class BFP16 {

private:
    uint16_t storage;

    struct Memory {
        union {
            float fp32;
            uint32_t u32;
            uint16_t u16s[2];
        };
    };

public:
    inline BFP16 () {
    }

    inline BFP16 (float x) {
        *this = x;
    }

    inline void operator = (float x) {
        Memory m;
        m.fp32 = x;
        if (m.u16s[0] & 0x8000) {
            Memory m2;
            m2.fp32 = x;
            m2.u32 &= 0xff800000;
            m2.fp32 /= 128.0f;
            m.fp32 += m2.fp32;
        }
        storage = m.u16s[1];
    }

    inline operator float () const {
        Memory m;
        m.u16s[0] = 0;
        m.u16s[1] = storage;
        return m.fp32;
    }

    inline uint16_t raw() const {
        return storage;
    }
};

typedef BFP16 bfp16_t;