#pragma once

#ifdef OPT_DISCRETIZATION

#include <math.h>

#include <faiss/Index.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>

namespace faiss {

template <typename Tid>
struct DiscreteRefiner {

    using idx_t = Index::idx_t;

    struct Fetcher {

        virtual idx_t get_label (const Tid& id) const = 0;

        virtual const float* get_vector (const Tid& id) const = 0;

    };

    static size_t refine (size_t d, const float* x, float max_error,
            size_t n, float* disc_distances, Tid* ids, Fetcher* fetcher,
            size_t chunk_size, size_t k, float* distances, idx_t* labels) {
        using C = CMax<float, idx_t>;
        heap_heapify<C> (k, distances, labels);
        if (n <= k) {
            for (size_t i = 0; i < n; i++) {
                heap_push<C> (k, distances, labels,
                        fvec_L2sqr (x, fetcher->get_vector (ids[i]), d),
                        fetcher->get_label (ids[i]));
            }
            heap_reorder<C> (k, distances, labels);
            return n;
        }
        float dis_threshold = C::neutral ();
        size_t ncalculate = 0;
        while (n) {
            if (n > chunk_size) {
                select (disc_distances, ids, 0, n - 1, chunk_size);
            }
            else {
                chunk_size = n;
            }
            sort (chunk_size, disc_distances, ids);
            for (size_t i = 0; i < chunk_size; i++) {
                float disc_distance = sqrtf (disc_distances[i]);
                if (disc_distance - max_error >= dis_threshold) {
                    heap_reorder<C> (k, distances, labels);
                    return ncalculate;
                }
                float distance = fvec_L2sqr (x,
                        fetcher->get_vector (ids[i]), d);
                ncalculate++;
                if (C::cmp (distances[0], distance)) {
                    heap_pop<C> (k, distances, labels);
                    heap_push<C> (k, distances, labels, distance,
                            fetcher->get_label (ids[i]));
                    dis_threshold = sqrtf (distances[0]);
                }
            }
            disc_distances += chunk_size;
            ids += chunk_size;
            n -= chunk_size;
        }
        heap_reorder<C> (k, distances, labels);
        return ncalculate;
    }

    static void select (float* distances, Tid* ids,
            size_t left, size_t right, size_t k) {
        while (true) {
            size_t i = split (distances, ids, left, right);
            if (k == i || k == i + 1) {
                return;
            }
            else if (k < i) {
                right = i - 1;
            }
            else {
                left = i + 1;
            }
        }
    }

    static size_t split (float* distances, Tid* ids,
            size_t left, size_t right) {
        size_t origin_left = left;
        float rule_d = distances[left];
        Tid rule_i = ids[left];
        while (left < right) {
            while (left < right && rule_d <= distances[right]) {
                right--;
            }
            while (left < right && distances[left] <= rule_d) {
                left++;
            }
            float tmp_d = distances[right];
            distances[right] = distances[left];
            distances[left] = tmp_d;
            Tid tmp_i = ids[right];
            ids[right] = ids[left];
            ids[left] = tmp_i;
        }
        distances[origin_left] = distances[left];
        distances[left] = rule_d;
        ids[origin_left] = ids[left];
        ids[left] = rule_i;
        return left;
    }

    static void sort (size_t n, float* distances, Tid* ids) {
        using C = CMax<float, Tid>;
        for (size_t i = 0; i < n; i++) {
            heap_push<C> (i + 1, distances, ids, distances[i], ids[i]);
        }
        for (size_t i = 0; i < n; i++) {
            float distance = distances[0];
            Tid id = ids[0];
            size_t j = n - i;
            heap_pop<C> (j, distances, ids);
            distances[j -1] = distance;
            ids[j - 1] = id;
        }
    }

};

}

#endif