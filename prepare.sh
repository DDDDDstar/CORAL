#!/bin/bash

CE=0
prefix=../data/t2i-10M
topk=100
num_threads=16
date=0716
iso_thres=0.9
recall=0.98

cd ~/pro/PIPEGPU
rm -r build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_TYPE=Debug -Wno-dev 
# cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_TYPE=Debug -DCUDAToolkit_ROOT=/usr/local/cuda-11.4 -Wno-dev 
make -j &> make.log


# 检查 make.log 文件是否存在
if [ ! -f "./make.log" ]; then
    echo "Error: ./make.log 文件不存在"
    exit 1
fi

# 使用 grep 检查文件中是否包含 "error" 字符串（不区分大小写）
if grep -qi "error" ./make.log; then
    echo "Fail"
else
    echo "Success"
fi