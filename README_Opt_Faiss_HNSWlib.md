# Introduction

This readme file includes introduction of how to build optimized faiss (based on Facebook open source Faiss) and faiss-benchmark (test tool developed by Intel). The output of compiling optimized faiss is a library, which can be tested using faiss-benchmark tool.

# Optimized Faiss (with HNSWLib)

This is an optimized version of Faiss by Intel. It is based on Faiss version 1.6.1, and includes implementation of HNSWLib feature:

Codes included by macro :

- "OPT_HNSWLIB"

## Required
​    gcc version 7 and above (recommended version gcc9.3)
​	kernel 4.2 and above

To build original Faiss with no optimization, just follow the original build way, like:

```
cd optimized-faiss
./configure --without-cuda
make clean
make
```

>(Optional if MKL is used) Maybe you need `export LDFLAGS=-L${MKL_LIB} (MKL_LIB is the folder like "/opt/intel/mkl/lib/intel64_lin") `  when  `./configure ...` fail.

Up to now, features included are as followings:


## HNSWlib (build and test)

```
./configure --without-cuda --enable-hnswlib (pure DDR version)

Confirm makefile.inc file, and modify if necessary
1.CPPFLAGS
	(index file on SSD)
	CPPFLAGS     = -DFINTEGER=int  -DOPT_HNSWLIB -fopenmp
2.CPUFLAGS
	CPUFLAGS     = -mpopcnt -msse4 -mavx2 -mavx512f -mavx512dq -mavx512vnni -mavx512vl -mavx512bw
```

```
# shell (build index and test, please refer to next section for faiss-benchmark)
1.	Input INT8 vector data, build INT8 index, and search with INT8 query
	export OMP_NUM_THREADS=32
	export LD_LIBRARY_PATH=../optimized-faiss
	./index build HNSWlibInt8-deep1B-32-500.idx HNSWlibInt8_32 ip efConstruction=500,verbose=1 deep1b_base_int8.fvecs 0 10000

	export OMP_NUM_THREADS=1
	./benchmark HNSWlibInt8-deep1B-32-500.idx deep1b_query_int8.fvecs deep1B_gt_ip_1k_int8.ivecs 100 99 efSearch=250/100x1x32

2. 	Input FP32 vector data, build FP32 index, and search with FP32 query
	export OMP_NUM_THREADS=32
	export LD_LIBRARY_PATH=../optimized-faiss
	./index build HNSWlibfp32-deep1B-32-500.idx HNSWlibFp32_32 ip efConstruction=500,verbose=1 deep1b_base_fp32.fvecs 0 10000

	export OMP_NUM_THREADS=1
	./benchmark HNSWlibfp32-deep1B-32-500.idx deep1b_query_fp32.fvecs deep1B_gt_ip_1k_fp32.ivecs 100 99 efSearch=250/100x1x32

3*. Input FP32 vector data, build INT16 index, and search with FP32 query
    export OMP_NUM_THREADS=32
	export LD_LIBRARY_PATH=../optimized-faiss
	./index build HNSWlibInt16-10M-32-500.idx HNSWlibInt16_32 ip efConstruction=500,scale=32768,verbose=1 deep10M.fvecs 0 10000

	export OMP_NUM_THREADS=1
	./benchmark HNSWlibInt16-10M-32-500.idx deep1b_query_fp32.fvecs groundtruth_10M_1k_ip.ivecs 100 99 efSearch=250,scale=32768/100x1x32
```

Note: Two parameters "scale" and "bias" are provided to set quantization factors (Y=Q[scale*x+bias]). They are used to quantize FP32 input data to INT16 or INT8 data. They can be used in both building index file or online search process. By default scale=1.0 and bias=0. In above example to convert FP32 to INT16, the values of Deep1B dataset range from -1.0 to 1.0, so we set scale=32768, in order to convert range from -32768 to 32767.

The key parameters during index building for different data types are listed:

| data type | index key        |
| --------- | ---------------- |
| FP32      | HNSWlibFp32_<M>  |
| INT16     | HNSWlibInt16_<M> |
| INT8      | HNSWlibInt8_<M>  |
| BFP16 *   | HNSWlibBfp16_<M> |

*Note: for data type of "BFP16", the input data format is FP32, and the vector data is saved as BFP16 in index. During searching, the vector data is read from index as BFP16, and converted to FP32 to compute distance.

## USE PMEM

~~~
1. Mount PMEM equipment (AD mode), For example:
	sudo mount /dev/pmem0 /mnt/dcpmm -o dax
2. Copy index file to pmem path (Index file is compatible with pure DDR version)
	cp deep_test_result/10M_HNSWlibFp32-32-500.idx /mnt/dcpmm/
3. Set environment variable (USE_PMEM=0: DDR, =1: PMEM)
	export USE_PMEM=1
	numactl -C 0-23 ./benchmark /mnt/dcpmm/10M_HNSWlibFp32-32-500.idx ./deep_data/deep1B_queries.fvecs ./deep_data/10M_gt_ip_1k.ivecs 100 99 efSearch=100/100x1x24
~~~



# faiss-benchmark

This is a [faiss](https://github.com/facebookresearch/faiss) test suite, which provides five common tools (subset, randset, index, groundtruth and benchmark) and a group test script.
Download via git:
```
git clone https://github.com/zhou-yuxin/faiss-benchmark
git reset --hard e33e557cfaf5a05e476ec2b888d66f47274c0190

Modify FAISS_DIR and PCM_DIR in Makefile to your local paths
```
## subset

This tool is used to extract a small dataset from a large dataset. The use method is as follows:

```
./subset <src> <dst> <n>
```
Among them, src is the file of large dataset, dst is the file of small dataset generated, and n is the number of extracted items. The tool selects n randomly from src, so the dst generated each time is different. Both src and dst can be bvecs, ivecs, fvecs and their gz packages. Subset automatically handles decompression and compression, as well as data type conversion.

Examples of use:
```
./subset bigann.bvecs.gz small.fvecs 1000
./subset sift1M_base.fvecs sift500K.fvecs.gz 500000
```

## randset

This tool is used to generate a random dataset. The use method is as follows:
```
./randset <dst> <dim> <n> <min> <max>
```
Where dst is the output file path, dim is the vector dimension, and n is the number of vectors. Min and max are the values of each dimension of the vector. Dst can be bvecs, ivecs, fvecs and their gz compressed packets. Randset automatically handles decompression and compression, as well as data type conversion.

Examples of use:
```
./randset rand10M.fvecs 128 10000000 -100.0 123.4
```

Note that empirically, the performance of the algorithm running on random dataset (such as rand1m) can represent the performance of the algorithm running on meaningful dataset (such as sift1m), but the recall rate will be much worse than that of meaningful dataset. Therefore, the application of randset is to test the performance of the algorithm on different dataset. Its recall rate is generally not of reference significance.

## index

The tool deals with index in two aspects:
1) Build index;
2) Estimate the memory size of index.

When used to build the index, the method is as follows:
```
./index build <fpath> <key> <metric> <parameters> <base> <train_ratio> <add_batch_size>
```
Where fpath is the storage path of the constructed index, and key is the type of index (such as "ivf1024, pq64", and the format is faiss:: index_ factory() is the same), metric is the distance type. Currently, it supports IP, L2 and the format of "raw:%d". Parameters are the parameters that need to be passed to the index (for example, "verb = 1, nprobe = 10", in the same format as faiss:: ParameterSpace), base is the file path of the entire dataset, train_ Ratio is a decimal between 0 and 1, which represents how much data is extracted from the base as the training dataset. add_ batch_ size is the number of vectors added each time through the add() interface. Like add_ batch_ size is equal to 1 to add one by one in order, generally speaking, add_ batch_ size can be appropriately larger (for example, 1000), because many index types have parallel acceleration for batch insertion. Like subset, base can be bvecs, ivecs, fvecs and their GZ packages. Index automatically handles decompression and compression, as well as data type conversion.

Examples of use:
```
./index build myindex.idx IVF1024,Flat l2 verbose=0 sift1M_base.fvecs 0.1 1000
```

When it is used to estimate the memory consumption of index, the method is as follows:
```
./index size <fpath>
```
Where fpath is the path to the index. This command outputs a value, in MB, of the memory occupied by index.

## groundtruth

This tool is used to calculate the groundtruth. The use method is as follows:
```
./groundtruth <gt> <base> <query> <metric> <top_n> <thread>
```
Among them, gt is the storage path of the generated groundtruth, base is the path of the entire dataset, query is the path of the query dataset, metric is the distance calculation method (currently supporting "L1" and "L2", i.e., Manhattan distance and European distance), top_n specifies the number of nearest neighbors, and thread is how many threads are used for parallel acceleration (does not affect the final result, only affects the speed). Base and query can be bvecs, ivecs, fvecss, and their gz packages, but gt must be ivecs or ivecs.gz 。

Examples of use:
```
./groundtruth sift1M_gt_1K.ivecs sift1M_base.fvecs sift1M_query.fvecs l2 1000 4
```

## benchmark

The above four tools are all auxiliary, and benchmark is the core. The use method is as follows:
```
./benchmark <index> <query> <gt> <top_n> <percentages> <cases>
```
Where index is the storage path of index, query is the path of query dataset, gt is the storage path of groundtruth, top_n is the number of nearest neighbors, percentages are percentiles separated by commas, and cases are test cases separated by semicolons. Similarly, query can be bvecs, ivecs, fvecss and their gz compression package. gt must be ivecs or ivecs.gz 。

As long as the value of top_n does not exceed top_n in gt,it is fine.For example, when groundtruth is used to generate sift1M_gt_1K.ivecs, the top_n passed in is 1000, which means that sift1M_gt_1K.ivecs contains 1000 nearest neighbors of each vector in sift1M_query.fvecs. Then, when sift1M_gt_1K.ivecs is passed to the benchmark tool as gt parameter, as long as top_n does not exceed 1000, benchmark will automatically intercept the specified top_n nearest neighbors.

The benchmark will output test results for each test case, and the format is as follows:
```
qps: 887.449
cpu-util: 4.10067
mem-r-bw: 7220.37
mem-w-bw: 16.0404
latency: best=3269 worst=7687 average=4506.26 P(50%)=4499 P(99%)=5599 P(99.9%)=5881
recall: best=1 worst=0.71 average=0.902705 P(50%)=0.9 P(99%)=0.81 P(99.9%)=0.77
```
They are QPS (requests per second), CPU utilization (for example, 4.10067 above is equivalent to 410.1% shown in the top command, that is, 4.1 processor cores are used on average), memory read bandwidth (MB/s), memory write bandwidth (MB/s), request latency Statistics (ms) and recall rate statistics. The statistics include the best case, worst case, and average, plus a number of user specified percentiles.

Percentages is the percentile specified by the user. If the user passes in "50, 99, 99.9", the statistics above will be obtained.
Cases are several test cases. The benchmark command can execute multiple test cases at a time, which can avoid repeated preparation work (such as loading index, query and groundtruth), thus greatly improving efficiency. The syntax of a single test case:
```
<parameters>/<loop>x<batch_size>x<thread_count>[:<cpu_list>]
```
Where parameters is a comma separated parameter list (in the same format as faiss:: ParameterSpace) to configure the index. For example, the meaning of "nprobe=64/1x1x8" means that the nprobe of index is set to 64, and then the test is executed with 8 threads and single batch. The first parameter loop after the '/' indicates that the loop is executed repeatedly with the same set of query dataset. For example, "/10x1x4" means using 4 threads and single batch to repeatedly query 10 times. Generally speaking, the first query may trigger a lot of initialization work, and repeated multiple times can flatten the effect. Case can add optional cpu_list, indicating which cores are each thread bound to. Cases are separated by semicolons.

Examples of use:
```
./benchmark myidex.idx sift1M_query.fvecs sift1M_gt_1K.ivecs 100 50,99,99.9 'nprobe=64/5x1x4;nprobe=128/1x1x8;nprobe=32,verbose=1/10x8x2:0,1'
```
Note that when using the shell, the semicolon is used by the shell as a separator for command parameters, so we need to wrap cases in quotation marks to avoid "over reading" of the shell.

>For error like "error while loading shared libraries: libmkl_intel_lp64.so", please `export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64/:$LD_LIBRARY_PATH

## Dependent software

1) zlib, Most Linux comes with it;
2) faiss, `git clone https://github.com/facebookresearch/faiss.git`;
3) pcm(It is used to obtain hardware information such as memory bandwidth), `git clone https://github.com/opcm/pcm.git`;

After modifying FAISS_DIR and PCM_DIR in Makefile, the above five executable files can be obtained by `make`.When you run index and benchmark,you need to load `libfaiss.so` dynamically,so you need to set LD_ LIBRARY_ PATH.

In addition, MSR will be accessed when the benchmark is running. To do this, the msr kernel module should be loaded with `sudo modprobe msr`, and then the benchmark can be run with root permission.

src/util/perfmon.h    line:25    add:

using namespace pcm;

## PCM
Faiss-benchmark is depended on PCM to monitor HW status.
```
git clone https://github.com/opcm/pcm
git checkout 201901
make
sudo modprobe msr
lsmod | grep -i msr
```

# Environment setup (Optional)
We can test or validate FAISS cases on python or C++ environment. It’s alternative.

(recommended)
```
export OMP_WAIT_POLICY=”PASSIVE”
export KMP_BLOCKTIME=0
export KMP_AFFINITY=”granularity=fine,compact,1,0”
export GOMP_SPINCOUNT=100
export OMP_NESTED=FALSE
sudo cpupower frequency-set -g performance 1>/dev/null
sudo cpupower idle-set -e 0 1>/dev/null
sudo cpupower idle-set -e 1 1>/dev/null
sudo cpupower idle-set -d 2 1>/dev/null
sudo cpupower idle-set -d 3 1>/dev/null
```


