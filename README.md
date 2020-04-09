# Optimized Faiss

This is an optimized version of Faiss by Intel. It is based on open-sourced Faiss 1.6.3.

To build original Faiss with no optimization, just follow the original build way, like:

```
./configure --without-cuda
make clean
make
```

Up to now, features included are as followings:

## - IVFPQ Relayout

This feature changes the layout of PQ code in InvertedLists in IndexIVFPQ. The new layout not only improves the cache hit rate, but also enables compiler-level SIMD optimization.

To enable this feature, you should append `--enable-ivfpq-relayout` to `./configure`, such as:

```
./configure --without-cuda --enable-ivfpq-relayout
```

Then, an IndexIVFPQ instance can be set to use this feature by such code:
```
# in Python
ps = faiss.ParameterSpace ()
ps.set_index_parameter (index, 'ivfpq_relayout', 4)
```
or
```
#in C++
faiss::ParameterSpace ps;
ps.set_index_parameter (index, "ivfpq_relayout", 4);
```
where *ivfpq_relayout* means the group size of relayout. Although *ivfpq_relayout* can be any non-negative integer, but 4 is usually the best based on experience.