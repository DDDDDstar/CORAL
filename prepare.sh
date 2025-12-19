#!/bin/bash

CE=0
prefix=../data/t2i-10M
dataset=t2i
topk=100
num_threads=16
# date=0829
iso_thres=0.7
deg_thres=10
recall_thres=99
recall=0.98
dist=l2
version=2
# suffix=_${dist}${version}_${recall_thres}_${date}
# suffix1=_${dist}${version}
# suffix2=_${dist}${version}_${date}

cd ~/pro/PIPEGPU
rm -r build
mkdir build
cp ./prepare.sh ./build/prepare.sh
cd build
/usr/bin/cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_TYPE=Debug -Wno-dev
# $CONDA_PREFIX/bin/cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_TYPE=Debug -Wno-dev
# cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_TYPE=Debug -DCUDAToolkit_ROOT=/usr/local/cuda-11.4 -Wno-dev 
make -j &> make.log


# 检查 make.log 文件是否存在
if [ ! -f "./make.log" ]; then
    echo "Error: ./make.log 文件不存在"
    exit 1
fi

# 使用 grep 检查文件中是否包含 "error" 字符串（区分大小写）
if grep -q "error" ./make.log; then
    echo "Fail"
else
    cuda-gdb --args ./tests/test_build_pipeline \
    --dataset ${dataset} --data_type float --dist ${dist} \
    --base_data_path ${prefix}/base.10M.fbin \
    --sampled_query_data_path ${prefix}/queries/query.train.10M.fbin \
    --recall_thres ${recall_thres} \
    --CE ${CE} --k ${topk} --M_sq 100 --M_pjbp 35 --L_pjpq 500 -T 64

#     nohup time ./tests/test_build_pipeline \
# --dataset ${dataset} --data_type float --dist ${dist} \
# --base_data_path ${prefix}/base.10M.fbin \
# --sampled_query_data_path ${prefix}/queries/query.train.10M.fbin \
# --recall_thres ${recall_thres} \
# --CE ${CE} --k ${topk} --M_sq 100 --M_pjbp 35 --L_pjpq 500 -T 64 &
fi