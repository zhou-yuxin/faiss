#ifdef OPT_DISCRETIZATION

#include <memory>
#include <vector>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#include <faiss/Clustering.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/IndexIVFFlatDiscrete.h>
#include <faiss/utils/DiscreteRefiner.h>
#include <faiss/utils/DiscreteDistances.h>

namespace faiss {

using idx_t = Index::idx_t;

struct ID {
    idx_t ilist;
    idx_t iy_in_list;
};

struct Fetcher : DiscreteRefiner<ID>::Fetcher {

    const InvertedLists* ivlists;

    Fetcher (const InvertedLists* ivlists) : ivlists (ivlists) {
    }

    idx_t get_label (const ID& id) const override {
        return ivlists->get_single_id (id.ilist, id.iy_in_list);
    }

    const float* get_vector (const ID& id) const override {
        return (float*) (ivlists->get_single_code (id.ilist,
                id.iy_in_list));
    }

};

struct IVFFlatDiscreteSpace {

    const size_t d;
    const size_t nlist;
    const float k;
    const float b;
    const float dis_scale;
    const float max_error;

    IVFFlatDiscreteSpace (size_t d, size_t nlist, float k, float b) :
            d (d), nlist (nlist), k (k), b (b), dis_scale (1.0f / k / k),
            max_error (0.5f * sqrtf (float (d)) * dis_scale) {
    }

    virtual ~IVFFlatDiscreteSpace () {
    }

    virtual void add (idx_t ilist, const float* y) = 0;

    virtual void reset () = 0;

    virtual void calculate (const float* x, idx_t ilist, size_t n,
            float* distances, ID* ids) const = 0;

};

struct Int8Space : IVFFlatDiscreteSpace {

    std::vector<int8_t>* lists;

    Int8Space (size_t d, size_t nlist, float k, float b) :
            IVFFlatDiscreteSpace (d, nlist, k, b),
            lists (new std::vector<int8_t> [nlist]) {
    }

    ~Int8Space () {
        delete[] lists;
    }

    void add (idx_t ilist, const float* y) override {
        std::vector<int8_t>* list = lists + ilist;
        for (size_t i = 0; i < d; i++) {
            list->emplace_back (roundf (y[i] * k + b));
        }
    }

    void reset () override {
        for (size_t i = 0; i < nlist; i++) {
            lists[i].clear ();
        }
    }

    void calculate (const float* x, idx_t ilist, size_t n,
            float* distances, ID* ids) const override {
        float* xt = new float [d];
        for (size_t i = 0; i < d; i++) {
            xt[i] = x[i] * k + b;
        }
        std::vector<int8_t>* list = lists + ilist;
        FAISS_ASSERT (n * d == list->size ());
        const int8_t* y = list->data ();
        for (size_t i = 0; i < n; i++) {
            distances[i] = DiscreteDistances::int8 (d, xt, y + i * d);
            FAISS_ASSERT (distances[i] >= 0.0f);
            ID* id = ids + i;
            id->ilist = ilist;
            id->iy_in_list = i;
        }
        delete[] xt;
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
    this->disc_exp.assign (disc_exp);
    parse_disc_exp (nlist);
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

void IndexIVFFlatDiscrete::rebuild_discrete_space () {
    size_t nlist = ivlists->nlist;
    parse_disc_exp (nlist);
    for (size_t i = 0; i < nlist; i++) {
        size_t list_size = ivlists->list_size (i);
        const float* y = (const float*) (ivlists->get_codes (i));
        for (size_t j = 0; j < list_size; j++) {
            disc->add (i, y);
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
        ivlists->add_entry (ilist, ntotal + i, (uint8_t*) (yi));
        disc->add (ilist, yi);
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
            size_t* list_sizes = new size_t [nprobe];
            #pragma omp for schedule (dynamic)
            for (idx_t i = 0; i < n; i++) {
                const float* xi = x + i * d;
                const idx_t* ilists_i = ilists + i * nprobe;
                size_t total_size = 0;
                for (size_t j = 0; j < nprobe; j++) {
                    size_t list_size = ivlists->list_size (ilists_i[j]);
                    list_sizes[j] = list_size;
                    total_size += list_size;
                }
                float* disc_distances = new float [total_size];
                ID* ids = new ID [total_size];
                float* disc_distances_j = disc_distances;
                ID* ids_j = ids;
                for (size_t j = 0; j < nprobe; j++) {
                    size_t list_size = list_sizes[j];
                    disc->calculate (xi, ilists_i[j], list_size,
                            disc_distances_j, ids_j);
                    disc_distances_j += list_size;
                    ids_j += list_size;
                }
                DiscreteRefiner<ID>::refine (d, xi, disc->max_error,
                        total_size, disc_distances, ids, &fetcher,
                        chunk_size >= k ? chunk_size : 4 * k, k,
                        distances + i * k, labels + i * k);
                delete[] disc_distances;
                delete[] ids;
            }
            delete[] list_sizes;
        }
    }
    else {
        FAISS_THROW_FMT ("unsupported parallel mode: %d", parallel_mode);
    }
}

void IndexIVFFlatDiscrete::parse_disc_exp (size_t nlist) {
    FAISS_THROW_IF_NOT_MSG (!disc, "discrete space is already built");
    const char* exp = disc_exp.data ();
    std::vector<char> type_buf;
    type_buf.resize (disc_exp.length ());
    char* type_exp = type_buf.data ();
    float k, b;
    if (sscanf (exp, "%[^:]:x%f+%f", type_exp, &k, &b) != 3) {
        FAISS_THROW_FMT ("invalid expression: '%s'", exp);
    }
    if (strcasecmp (type_exp, "int8") == 0) {
        disc = new Int8Space (d, nlist, k, b);
    }
    else if (strcasecmp (type_exp, "uint8") == 0) {
        disc = new Int8Space (d, nlist, k, b - 128.0f);
    }
    else {
        FAISS_THROW_FMT ("unsupported type: '%s'", type_exp);
    }
}

}

#endif