# ./data/t2i-10M/query.train.10M.fbin overwrite the first 4 bytes by 10000000 as uint32

import os
import sys
import numpy as np

META_DTYPE = np.uint32
FLOAT_DTYPE = np.float32

def fit_meta_and_copy_data(src_file, dst_file, data_size, dim, start=0):
    meta_bytes = 2 * META_DTYPE().nbytes
    vector_bytes = dim * FLOAT_DTYPE().nbytes
    data_bytes = data_size * vector_bytes
    print(f"data_bytes: {data_bytes}\n")

    # 打开目标文件，写 meta_data
    with open(dst_file, 'w+b') as fd:
        fd.seek(0)
        # 写新的 meta
        fd.write(np.array([data_size, dim], dtype=META_DTYPE).tobytes())
        # 定位到 meta_data 后
        fd.seek(meta_bytes)
        # 打开源文件
        with open(src_file, 'rb') as fs:
            # 跳过源文件 meta
            src_offset = meta_bytes + start * vector_bytes
            fs.seek(src_offset)
            # 分块拷贝
            buf_size = 1024 * 1024 * 1024
            remaining = data_bytes
            while remaining > 0:
                to_read = min(buf_size, remaining)
                buf = fs.read(to_read)
                if not buf:
                    raise RuntimeError("Source file ended unexpectedly")
                fd.write(buf)
                remaining -= len(buf)
                print(f"Remaining: {remaining}\n")

if __name__ == '__main__':
    print("change_data_in_file")
    data_file_s = sys.argv[1]
    data_file_d = sys.argv[2]
    data_size = int(sys.argv[3])
    dim = int(sys.argv[4])
    if len(sys.argv) <= 5:
        start = 0
    else:
        start = int(sys.argv[5])
    print(f"COPY DATA:\ndata_file_s: {data_file_s}\ndata_file_d: {data_size}\ndata_size: {data_size}, dim: {dim}")
    fit_meta_and_copy_data(data_file_s, data_file_d, data_size, dim, start)
    print(f"copy finished")
