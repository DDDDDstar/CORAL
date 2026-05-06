import numpy as np
import os

path = "/data/wsx/newdata/laion-10M/"
datapath = path + 'data/'
indices = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]

# ---------- IMG ----------
img_out = open(path + "base.fbin", "wb")

total_pts = 0
dim = None

# 先统计点数 + 维度
for i in indices:
    arr = np.load(datapath + f"img_emb_{i}.npy", mmap_mode="r")
    if dim is None:
        dim = arr.shape[1]
    total_pts += arr.shape[0]

# 写 header
img_out.write(np.array([total_pts, dim], dtype=np.uint32).tobytes())

# 分块写数据
for i in indices:
    arr = np.load(datapath + f"img_emb_{i}.npy", mmap_mode="r")
    arr = arr.astype(np.float32, copy=False)
    arr.tofile(img_out)

img_out.close()

# ---------- TEXT ----------
txt_out = open(path + "query.train.fbin", "wb")

total_pts = 0
dim = None

for i in indices:
    arr = np.load(datapath + f"text_emb_{i}.npy", mmap_mode="r")
    if dim is None:
        dim = arr.shape[1]
    total_pts += arr.shape[0]

txt_out.write(np.array([total_pts, dim], dtype=np.uint32).tobytes())

for i in indices:
    arr = np.load(datapath + f"text_emb_{i}.npy", mmap_mode="r")
    arr = arr.astype(np.float32, copy=False)
    arr.tofile(txt_out)

txt_out.close()

# imgs_10M = np.array([])
# text_10M = np.array([])
# dim = 512
# path = "/data/wsx/newdata/laion-10M/"
# datapath = path + 'data/'
# for i in [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]:
#     # append np arrays 
#     img_name = datapath + f'img_emb_{i}.npy'
#     one_img = np.load(img_name)
#     # print(img_name)
#     # convert to float32
#     one_img = one_img.astype(np.float32) 
#     dim = one_img.shape[1]
#     imgs_10M = np.append(imgs_10M, one_img).astype(np.float32)
#     text_name = datapath + f'text_emb_{i}.npy'
#     one_text = np.load(text_name)
#     one_text = one_text.astype(np.float32)
#     text_10M = np.append(text_10M, one_text).astype(np.float32)
#     # print(one_text.shape)

# imgs_10M = imgs_10M.reshape(-1, dim)
# text_10M = text_10M.reshape(-1, dim)
# # print(text_10M.shape)

# f_img = open(path + 'base.10M.fbin', 'wb')
# f_txt = open(path + 'query.train.10M.fbin', 'wb')



# # save imgs_10M to f, write num points and dimension at first
# npts, dim = imgs_10M.shape
# f_img.write(np.array([npts, dim]).astype(np.uint32).tobytes())
# imgs_10M.tofile(f_img)
# f_img.close()


# # save text_10M to f, write num points and dimension at first
# npts, dim = text_10M.shape
# f_txt.write(np.array([npts, dim]).astype(np.uint32).tobytes())
# text_10M.tofile(f_txt)
# f_txt.close()

