# CORAL: Cross-modal Vector Retrieval via Incremental Graph Construction at Scale

CORAL is a novel GPU-accelerated graph-based vector index for scalable cross-modal retrieval, featuring hierarchical memory management that spans GPU, CPU, and disk.
CORAL incrementally incorporates the characteristics of query modal and timely stops index construction.
CORAL also supports modal-semantics-based vector insertion and topology-repairing deletion that restore node connectivity.

<img src="./README.assets/def.pdf" alt="image-20260506下午44123908" style="zoom:200%;" />

## Getting Started

Our experiments are conducted on a server running Ubuntu 24.04.1 LTS, equipped with one Intel(R) Xeon(R) Gold 5218 processor, 256 GB of main memory, and one NVIDIA A100 PCIe GPU with 40 GB of HBM2 device memory. All GPU kernels are implemented and compiled using CUDA Toolkit 12.8.

**File format**: All `bin` files follow the same format as big-ann competition, including vector data file and ground-truth (gt) data file. The vector data files begin with the number of vectors (uint32, 4 bytes), dimension (uint32, 4 bytes), and followed by the vector data. The gt data file begin with the number of queries (uint32, 4 bytes), the number of retrieved nearest neighbors (uint32, 4 bytes), and followed by the gt data (The ids of the nearest neighbors).

### 0. Prerequisite

```
cmake >= 4.1.1
g++ >= 13.3.0
cuda >= 12.8
CPU supports AVX-512

Python >= 3.13.7
Python package:
numpy
tqdm
torch
clip
```

```bash
sudo apt install libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev libmkl-full-dev
```

### 1. Data Preparation

The dataset information is as follows. $|D|$ is the size of the base dataset, and $|Q_b|$ is the size of the auxiliary query set.

|      Dataset       | Dim  | Metric | $|D|$ | $|Q_b|$  | Modalities <br />(base, query) |
| :----------------: | :--: | :----: | :---: | :------: | ------------------------------ |
| Text-to-Image(T2I) | 200  |   L2   | 1M~1B | 0.1M~30M | Image, Text                    |
|       WebVid       | 512  |   IP   | 2.5M  |    1M    | Video, Text                    |
|       LAION        | 512  |   IP   |  10M  |    1M    | Image, Text                    |
|        WIT         | 512  |   IP   |  1M   |   0.1M   | Image, Text                    |
|        CC3M        | 512  |   IP   |  1M   |   0.1M   | Image, Text                    |

We use `./data_prepare/prepare_data.sh` to prepare the train and test data of each dataset, including the base data file `base.fbin`, the query data file for index construction `query.train.fbin`, the query data file for test `query.fbin`, and the gt data file `query.gt.bin`.

For example, to prepare the data of 1M-scale T2I dataset:

```bash
./data_prepare/prepare_data.sh t2i-1M
```

The data files will be saved in the `./data` directory.

### 2. Compile and Build

```bash
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug && make -j
```

### 3. Index Construction

The index construction workflow is as follows.

<img src="./README.assets/base.pdf" alt="image-20260506下午44123908" style="zoom:200%;" />

To build the index:

-   `dataset`: The name of dataset.
-   `base_data_path`: the base data file.
-   `sampled_query_data_path`: the auxiliary query data file for index construction.
-   `config_file`: The JSON file used for configuring various parameters,  which can refer to the content in `./config.json`.
-   `k`: $k_b$ in the paper, representing the query coverage.
-   `max_degree`: The maximun outg-degree of each node in the graph.
-   `T`: Number of threads used for index construction.

```bash
dataset=t2i-1M
prefix=../data/${dataset}
cd build
./tests/test_build_pipeline \
		--dataset ${dataset} --data_type float --dist l2 \
		--base_data_path ${prefix}/base.fbin \
		--sampled_query_data_path ${prefix}/query.train.fbin \
		--config_file ./config.json \
		--k 100  --max_degree 32 -T 32
```

The index will be saved in `./indexes` directory.

### 4. Search

 The overview of the index search and update is as follows.

<img src="./README.assets/sid.pdf" alt="image-20260506下午44123908" style="zoom:200%;" />

To search on the index:

-   `k`: The number of nearest neighbors results for each query.
-   `L`: capacities of the beam list during the search phase.
-   `query_path`: the query data file for search.
-   `gt_path`: the gt data file for search evaluation.

```bash
dataset=t2i-1M
prefix=../data/${dataset}
cd build
./tests/test_search_pipeline \
            --dataset ${dataset} --data_type float --dist ${dist} \
            --base_data_path ${prefix}/base.fbin \
            --config_file ./config.json \
            --query_path ${prefix}/query.fbin \
            --gt_path ${prefix}/gt.bin \
            --L 256 512 \
            --k 100 -T 16
```

The evaluation results will be saved in `../evaluation` directory.

### 5. Update

For the index update experiment, the entire base dataset will be divided into two. The first half is used to build the initial index, and the second half is used to insert into the index. $1\%$ of the initial data size is regarded as the size of a piece of vector data. There are three modes of index update: each step (0) delete a piece of original vector data first, and then insert a piece of new vector data; (1) delete a piece of original vector data; (2) insert a piece of new vector data. The update continues until all the original data is deleted or all new data is inserted. After each update step, the index performance will be re-evaluated.

To compute the gt of index update:

-   `update_gt_file`: where to store the gt results.
-   `upd_mode`: The update mode, 0 for mode 0, 1 for mode 1, 2 for mode 2.

```bash
dataset=t2i-10M
prefix=../data/${dataset}
./tests/update_compute_gt \
		--dataset ${dataset} --dist l2 \
		--base_data_path ${prefix}/base.fbin \
		--query_path ${prefix}/query.fbin \
		--update_gt_file ${prefix}/update.gt.bin \
		--upd_mode 0 --k 100
```

To evaluate the index update:

```bash
./tests/test_update \
		--dataset ${dataset} --data_type float --dist ${dist} \
		--base_data_path ${data_path}/base.fbin \
		--sampled_query_data_path ${data_path}/query.train.fbin \
		--query_path ${data_path}/query.fbin \
		--update_gt_file ${data_path}/update.gt.bin \
		--config_file ${config_file} \
		--k 100  --max_degree 32 --upd_mode 0 --L 512
```

