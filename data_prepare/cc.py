#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import threading
import queue

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import clip
from datasets import load_dataset
from huggingface_hub import list_repo_files, hf_hub_download


# =========================
# FBIN Writer
# =========================
class FBinWriter:
    def __init__(self, path):
        self.f = open(path, "wb")
        self.count = 0
        self.dim = None
        np.array([0, 0], dtype=np.int32).tofile(self.f)

    def append(self, x):
        if x.shape[0] == 0:
            return

        if self.dim is None:
            self.dim = x.shape[1]
        elif self.dim != x.shape[1]:
            raise ValueError("dim mismatch")

        x.astype(np.float32).tofile(self.f)
        self.count += x.shape[0]

        pos = self.f.tell()
        self.f.seek(0)
        np.array([self.count, self.dim], dtype=np.int32).tofile(self.f)
        self.f.seek(pos)

    def close(self):
        self.f.close()


# =========================
# Decode
# =========================
def decode_train(ex):
    try:
        if "jpg" not in ex or "txt" not in ex:
            return None
        img = ex["jpg"]
        if not isinstance(img, Image.Image):
            return None
        txt = str(ex["txt"]).strip()
        if not txt:
            return None
        return img.convert("RGB"), txt
    except:
        return None


def decode_test(ex):
    try:
        if "txt" not in ex:
            return None
        txt = str(ex["txt"]).strip()
        if not txt:
            return None
        return txt
    except:
        return None


# =========================
# 下载线程
# =========================
def download_worker(files, q, tmp_dir, stop_event):
    for f in files:
        if stop_event.is_set():
            break
        try:
            path = hf_hub_download(
                "pixparse/cc3m-wds",
                filename=f,
                repo_type="dataset",
                local_dir=tmp_dir,
            )
            q.put((f, path))
        except:
            q.put((f, None))
    q.put((None, None))


# =========================
# 主流程
# =========================
@torch.no_grad()
def run(args):

    os.makedirs(args.output_dir, exist_ok=True)
    tmp_dir = os.path.join(args.output_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    model, preprocess = clip.load(args.clip_model, device=args.device)
    model.eval()

    files = list_repo_files("pixparse/cc3m-wds", repo_type="dataset")
    tar_files = [f for f in files if f.endswith(".tar") and ("train-0574" in f or "train-0575" in f  or "train-0570" in f)]
    print(len(tar_files))
    random.seed(42)
    selected = random.sample(tar_files, args.num_shards)

    q = queue.Queue(maxsize=8)
    stop_event = threading.Event()

    threading.Thread(
        target=download_worker,
        args=(selected, q, tmp_dir, stop_event),
        daemon=True
    ).start()

    # =========================
    # writers
    # =========================
    if args.mode == "train":
        base_writer = FBinWriter(os.path.join(args.output_dir, "base.fbin"))
        query_writer = FBinWriter(os.path.join(args.output_dir, "query.train.fbin"))
    else:
        query_writer = FBinWriter(os.path.join(args.output_dir, "query.fbin"))

    total = 0
    target = args.target_size if args.mode == "train" else args.test_size
    pbar = tqdm(total=target)

    while True:

        fname, tar_path = q.get()

        if total >= target:
            stop_event.set()
            break

        if fname is None:
            break

        if tar_path is None:
            continue

        try:
            ds = load_dataset(
                "webdataset",
                data_files=tar_path,
                split="train",
                streaming=True
            )

            batch_img, batch_txt = [], []

            for ex in ds:
                if total >= target:
                    break

                # =========================
                # TRAIN
                # =========================
                if args.mode == "train":
                    r = decode_train(ex)
                    if r is None:
                        continue
                    img, txt = r
                    batch_img.append(preprocess(img))
                    batch_txt.append(txt)

                # =========================
                # TEST（只文本）
                # =========================
                else:
                    txt = decode_test(ex)
                    if txt is None:
                        continue
                    batch_txt.append(txt)

                # =========================
                # batch forward
                # =========================
                if len(batch_txt) >= args.batch_size:

                    remain = target - total

                    if args.mode == "train":

                        images = torch.stack(batch_img).to(args.device)
                        text = clip.tokenize(batch_txt).to(args.device)

                        img_f = model.encode_image(images)
                        txt_f = model.encode_text(text)

                        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
                        txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)

                        img_np = img_f.cpu().numpy()[:remain]
                        txt_np = txt_f.cpu().numpy()[:remain]

                        base_writer.append(img_np)
                        query_writer.append(txt_np)

                        written = img_np.shape[0]

                    else:

                        text = clip.tokenize(batch_txt).to(args.device)
                        txt_f = model.encode_text(text)
                        txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)

                        txt_np = txt_f.cpu().numpy()[:remain]

                        query_writer.append(txt_np)

                        written = txt_np.shape[0]

                    total += written
                    pbar.update(written)

                    batch_img.clear()
                    batch_txt.clear()

        except Exception as e:
            print("[skip shard]", fname, e)

        try:
            os.remove(tar_path)
        except:
            pass

    pbar.close()

    if args.mode == "train":
        base_writer.close()
    query_writer.close()

    print("DONE")


# =========================
# CLI
# =========================
def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_shards", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clip_model", default="ViT-B/32")

    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--target_size", type=int, default=1000000)
    parser.add_argument("--test_size", type=int, default=10000)

    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()