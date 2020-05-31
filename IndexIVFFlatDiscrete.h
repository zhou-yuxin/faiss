#pragma once

#ifdef OPT_DISCRETIZATION

#include <string>

#include <faiss/Index.h>

namespace faiss {

struct IVFFlatDiscreteSpace;

struct IndexIVFFlatDiscrete: Index {

    Index* quantizer;
    bool own_quantizer;
    InvertedLists* ivlists;
    std::string disc_exp;
    IVFFlatDiscreteSpace* disc;
    size_t nprobe;
    size_t chunk_size;
    int parallel_mode;

    IndexIVFFlatDiscrete ();

    IndexIVFFlatDiscrete (Index* quantizer, size_t d, size_t nlist,
            MetricType metric = METRIC_L2,
            const char* disc_exp = "x1+0>>int8");

    ~IndexIVFFlatDiscrete ();

    void rebuild_discrete_space (const char* exp = "x1+0>>int8");

    void train (idx_t n, const float* y) override;

    void add (idx_t n, const float* y) override;

    void reset () override;

    void search (idx_t n, const float* x,
            idx_t k, float* distances, idx_t* labels) const override;

private:
    void parse_disc_exp (const char* exp, size_t nlist);

};

}

#endif