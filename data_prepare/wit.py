#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import re
import json
import time
import shutil
import argparse
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import clip  # pip install git+https://github.com/openai/CLIP.git

from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
import pyarrow.parquet as pq


# =========================================================
# 1) Environment / proxy
# =========================================================
def setup_network_env(
    http_proxy: str = "",
    https_proxy: str = "",
    hf_token: str = "",
    hf_home: str = "",
):
    if http_proxy:
        os.environ["http_proxy"] = http_proxy
        os.environ["HTTP_PROXY"] = http_proxy
    if https_proxy:
        os.environ["https_proxy"] = https_proxy
        os.environ["HTTPS_PROXY"] = https_proxy

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    if hf_home:
        os.environ["HF_HOME"] = hf_home
        os.makedirs(hf_home, exist_ok=True)

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


# =========================================================
# 2) Basic utils
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: str, obj: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


class FileInterrupted(RuntimeError):
    pass


# =========================================================
# 3) .fbin writer / reader
# format:
#   int32 n
#   int32 d
#   float32 data[n * d]
# =========================================================
class FBinWriter:
    def __init__(self, path: str, append: bool = False):
        self.path = path
        self.count = 0
        self.dim = None

        if append and os.path.exists(path) and os.path.getsize(path) >= 8:
            self.f = open(path, "r+b")
            header = np.fromfile(self.f, dtype=np.int32, count=2)
            if len(header) != 2:
                raise RuntimeError(f"Invalid fbin header: {path}")
            self.count = int(header[0])
            self.dim = int(header[1])
            self.f.seek(0, os.SEEK_END)
        else:
            self.f = open(path, "wb")
            np.array([0, 0], dtype=np.int32).tofile(self.f)

    def append(self, x: np.ndarray):
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        if x.ndim != 2:
            raise ValueError(f"append expects 2D array, got shape={x.shape}")

        bs, d = x.shape
        if self.dim is None or self.dim == 0:
            self.dim = d
        elif self.dim != d:
            raise ValueError(f"inconsistent dim: got {d}, expected {self.dim}")

        x.tofile(self.f)
        self.count += bs

    def close(self):
        self.f.flush()
        self.f.seek(0)
        np.array([self.count, 0 if self.dim is None else self.dim], dtype=np.int32).tofile(self.f)
        self.f.close()


def read_fbin_header(path: str) -> Tuple[int, int]:
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=2)
    if len(header) != 2:
        raise RuntimeError(f"Invalid fbin header: {path}")
    return int(header[0]), int(header[1])


def copy_fbin_body(src_path: str, dst_fp):
    with open(src_path, "rb") as f:
        f.seek(8)
        shutil.copyfileobj(f, dst_fp, length=8 << 20)


# =========================================================
# 4) WIT schema compatibility
# =========================================================
def _first_non_empty(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        for x in v:
            if x is None:
                continue
            x = " ".join(str(x).split())
            if x:
                return x
        return ""
    x = " ".join(str(v).split())
    return x if x else ""


def get_wit_feature(example: Dict[str, Any], key: str):
    wf = example.get("wit_features", None)
    if isinstance(wf, dict) and key in wf:
        return wf.get(key)
    return example.get(key, None)


def get_language(example: Dict[str, Any]) -> str:
    return _first_non_empty(get_wit_feature(example, "language"))


def get_page_title(example: Dict[str, Any]) -> str:
    return _first_non_empty(get_wit_feature(example, "page_title"))


def get_page_url(example: Dict[str, Any]) -> str:
    return _first_non_empty(get_wit_feature(example, "page_url"))


def choose_text(example: Dict[str, Any]) -> str:
    candidates = [
        get_wit_feature(example, "caption_reference_description"),
        get_wit_feature(example, "caption_title_and_reference_description"),
        get_wit_feature(example, "caption_alt_text_description"),
        get_wit_feature(example, "caption_attribution_description"),
        get_wit_feature(example, "context_section_description"),
        get_wit_feature(example, "context_page_description"),
        get_wit_feature(example, "page_title"),
    ]
    for c in candidates:
        t = _first_non_empty(c)
        if t:
            return t
    return ""


def get_hf_image(example: Dict[str, Any]) -> Optional[Image.Image]:
    img = example.get("image", None)
    if img is None:
        return None

    if isinstance(img, Image.Image):
        return img.convert("RGB")

    if isinstance(img, dict):
        if img.get("bytes") is not None:
            try:
                return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
            except Exception:
                return None
        if img.get("path") is not None:
            try:
                return Image.open(img["path"]).convert("RGB")
            except Exception:
                return None

    try:
        return Image.fromarray(img).convert("RGB")
    except Exception:
        return None


# =========================================================
# 5) Manifest / file list
# =========================================================
def manifest_path(output_root: str) -> str:
    return os.path.join(output_root, "parquet_manifest.json")


def skipped_jsonl_path(output_root: str) -> str:
    return os.path.join(output_root, "skipped_files.jsonl")


def get_or_build_manifest(output_root: str, dataset_name: str, split: str) -> List[Dict[str, Any]]:
    mpath = manifest_path(output_root)
    if os.path.exists(mpath):
        return load_json(mpath)["files"]

    api = HfApi()
    repo_files = api.list_repo_files(repo_id=dataset_name, repo_type="dataset")

    pat = re.compile(rf"^data/{re.escape(split)}-\d{{5}}-of-\d{{5}}\.parquet$")
    parquet_files = sorted([f for f in repo_files if pat.match(f)])

    if not parquet_files:
        raise RuntimeError(f"No parquet files found for split={split} in dataset={dataset_name}")

    files = []
    for idx, fname in enumerate(parquet_files):
        files.append({
            "file_idx": idx,
            "filename": fname,
            "url": hf_hub_url(repo_id=dataset_name, filename=fname, repo_type="dataset"),
        })

    save_json(mpath, {
        "dataset_name": dataset_name,
        "split": split,
        "num_files": len(files),
        "files": files,
    })
    return files


# =========================================================
# 6) File shard state
# =========================================================
def file_shard_dir(root: str, file_idx: int) -> str:
    return os.path.join(root, "file_shards", f"file_{file_idx:05d}")


def file_shard_paths(root: str, file_idx: int) -> Dict[str, str]:
    d = file_shard_dir(root, file_idx)
    return {
        "dir": d,
        "base": os.path.join(d, "base.fbin"),
        "query": os.path.join(d, "query.fbin"),
        "meta": os.path.join(d, "pairs_meta.jsonl"),
        "summary": os.path.join(d, "summary.json"),
    }


def load_file_progress(root: str, file_idx: int) -> Tuple[int, bool]:
    p = file_shard_paths(root, file_idx)
    if not os.path.exists(p["summary"]):
        return 0, False
    try:
        summary = load_json(p["summary"])
    except Exception:
        return 0, False
    return int(summary.get("pairs_written", 0)), bool(summary.get("complete", False))


def is_file_complete(root: str, file_idx: int) -> bool:
    p = file_shard_paths(root, file_idx)
    if not (os.path.exists(p["base"]) and os.path.exists(p["query"]) and os.path.exists(p["summary"])):
        return False

    try:
        summary = load_json(p["summary"])
        bn, bd = read_fbin_header(p["base"])
        qn, qd = read_fbin_header(p["query"])
    except Exception:
        return False

    if not summary.get("complete", False):
        return False
    if bn != summary.get("pairs_written", -1):
        return False
    if qn != summary.get("pairs_written", -1):
        return False
    if bd <= 0 or qd <= 0:
        return False

    return True


def load_skipped_file_ids(output_root: str) -> set:
    path = skipped_jsonl_path(output_root)
    out = set()
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                out.add(int(obj["file_idx"]))
            except Exception:
                pass
    return out


# =========================================================
# 7) Parquet direct reader
# =========================================================
def iter_parquet_rows(local_parquet_path: str):
    """
    Direct parquet iteration via pyarrow.
    Avoids HF datasets schema-unification issues.
    """
    pf = pq.ParquetFile(local_parquet_path)

    for batch in pf.iter_batches():
        table = batch.to_pydict()
        if not table:
            continue
        keys = list(table.keys())
        n = len(table[keys[0]])
        for i in range(n):
            ex = {}
            for k in keys:
                ex[k] = table[k][i]
            yield ex


# =========================================================
# 8) Process one parquet file
# =========================================================
@torch.no_grad()
def process_one_parquet_file(
    output_root: str,
    file_idx: int,
    file_info: Dict[str, Any],
    global_target_size: int,
    global_written_before: int,
    model,
    preprocess,
    batch_size: int,
    device: str,
    dataset_name: str,
    language: Optional[str],
    save_meta: bool = True,
):
    p = file_shard_paths(output_root, file_idx)
    ensure_dir(p["dir"])

    already_written_in_file, already_complete = load_file_progress(output_root, file_idx)
    if already_complete:
        print(f"[resume] file shard {file_idx:05d} already complete, skip")
        return already_written_in_file

    append_mode = already_written_in_file > 0 and os.path.exists(p["base"]) and os.path.exists(p["query"])

    # download this parquet to local HF cache / local file
    local_parquet = hf_hub_download(
        repo_id=dataset_name,
        filename=file_info["filename"],
        repo_type="dataset",
    )

    base_writer = FBinWriter(p["base"], append=append_mode)
    query_writer = FBinWriter(p["query"], append=append_mode)
    meta_fp = open(p["meta"], "a", encoding="utf-8") if save_meta else None

    batch_images = []
    batch_texts = []
    batch_meta = []

    valid_seen_in_file = 0
    written_in_file = already_written_in_file
    scanned_in_file = 0
    t0 = time.time()

    remain_global = global_target_size - global_written_before
    if remain_global <= 0:
        return 0

    try:
        pbar = tqdm(
            total=remain_global,
            initial=already_written_in_file,
            desc=f"file {file_idx:05d}",
        )

        for ex in iter_parquet_rows(local_parquet):
            if written_in_file >= remain_global:
                break

            scanned_in_file += 1

            ex_lang = get_language(ex)
            if language is not None and ex_lang != language:
                continue

            text = choose_text(ex)
            if not text:
                continue

            image = get_hf_image(ex)
            if image is None:
                continue

            if valid_seen_in_file < already_written_in_file:
                valid_seen_in_file += 1
                continue

            batch_images.append(preprocess(image))
            batch_texts.append(text)
            batch_meta.append({
                "language": ex_lang,
                "image_url": ex.get("image_url", ""),
                "page_url": get_page_url(ex),
                "page_title": get_page_title(ex),
                "text": text,
            })
            valid_seen_in_file += 1

            if len(batch_images) >= batch_size:
                images = torch.stack(batch_images, dim=0).to(device, non_blocking=True)
                text_tokens = clip.tokenize(batch_texts, truncate=True).to(device, non_blocking=True)

                img_f = model.encode_image(images)
                txt_f = model.encode_text(text_tokens)

                img_f = img_f / img_f.norm(dim=-1, keepdim=True)
                txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)

                img_np = img_f.cpu().numpy().astype(np.float32)
                txt_np = txt_f.cpu().numpy().astype(np.float32)

                remain_file = remain_global - written_in_file
                if img_np.shape[0] > remain_file:
                    img_np = img_np[:remain_file]
                    txt_np = txt_np[:remain_file]
                    batch_meta = batch_meta[:remain_file]

                base_writer.append(img_np)
                query_writer.append(txt_np)

                if meta_fp is not None:
                    for item in batch_meta:
                        meta_fp.write(json.dumps(item, ensure_ascii=False) + "\n")

                written_in_file += img_np.shape[0]
                pbar.update(img_np.shape[0])

                batch_images.clear()
                batch_texts.clear()
                batch_meta.clear()

        if written_in_file < remain_global and len(batch_images) > 0:
            images = torch.stack(batch_images, dim=0).to(device, non_blocking=True)
            text_tokens = clip.tokenize(batch_texts, truncate=True).to(device, non_blocking=True)

            img_f = model.encode_image(images)
            txt_f = model.encode_text(text_tokens)

            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)

            img_np = img_f.cpu().numpy().astype(np.float32)
            txt_np = txt_f.cpu().numpy().astype(np.float32)

            remain_file = remain_global - written_in_file
            if img_np.shape[0] > remain_file:
                img_np = img_np[:remain_file]
                txt_np = txt_np[:remain_file]
                batch_meta = batch_meta[:remain_file]

            base_writer.append(img_np)
            query_writer.append(txt_np)

            if meta_fp is not None:
                for item in batch_meta:
                    meta_fp.write(json.dumps(item, ensure_ascii=False) + "\n")

            written_in_file += img_np.shape[0]
            pbar.update(img_np.shape[0])

        pbar.close()

    except Exception as e:
        err_msg = "".join(traceback.format_exception_only(type(e), e)).strip()

        partial_summary = {
            "file_idx": file_idx,
            "filename": file_info["filename"],
            "url": file_info["url"],
            "pairs_written": written_in_file,
            "scanned_examples": scanned_in_file,
            "language": language,
            "device": device,
            "base_fbin": p["base"],
            "query_train_fbin": p["query"],
            "meta_jsonl": p["meta"] if save_meta else "",
            "elapsed_sec": round(time.time() - t0, 3),
            "complete": False,
            "error": err_msg,
        }
        save_json(p["summary"], partial_summary)

        raise FileInterrupted(
            f"file {file_idx:05d} interrupted after writing {written_in_file:,} pairs: {err_msg}"
        )

    finally:
        base_writer.close()
        query_writer.close()
        if meta_fp is not None:
            meta_fp.close()

    summary = {
        "file_idx": file_idx,
        "filename": file_info["filename"],
        "url": file_info["url"],
        "pairs_written": written_in_file,
        "scanned_examples": scanned_in_file,
        "language": language,
        "device": device,
        "base_fbin": p["base"],
        "query_train_fbin": p["query"],
        "meta_jsonl": p["meta"] if save_meta else "",
        "elapsed_sec": round(time.time() - t0, 3),
        "complete": True,
    }
    save_json(p["summary"], summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return written_in_file


# =========================================================
# 9) Retry wrapper
# =========================================================
def process_one_parquet_file_with_retry(
    output_root: str,
    file_idx: int,
    file_info: Dict[str, Any],
    global_target_size: int,
    global_written_before: int,
    model,
    preprocess,
    batch_size: int,
    device: str,
    dataset_name: str,
    language: Optional[str],
    save_meta: bool = True,
    max_retry: int = 5,
    retry_sleep: int = 10,
    skip_bad_file: bool = True,
):
    last_err = None

    for attempt in range(max_retry + 1):
        try:
            if attempt > 0:
                sleep_s = retry_sleep * attempt
                print(f"[retry] file {file_idx:05d}, attempt {attempt}/{max_retry}, sleep {sleep_s}s")
                time.sleep(sleep_s)

            return process_one_parquet_file(
                output_root=output_root,
                file_idx=file_idx,
                file_info=file_info,
                global_target_size=global_target_size,
                global_written_before=global_written_before,
                model=model,
                preprocess=preprocess,
                batch_size=batch_size,
                device=device,
                dataset_name=dataset_name,
                language=language,
                save_meta=save_meta,
            )

        except FileInterrupted as e:
            last_err = e
            print(f"[retry] file {file_idx:05d} failed: {e}")
            if attempt >= max_retry:
                if skip_bad_file:
                    append_jsonl(skipped_jsonl_path(output_root), {
                        "file_idx": file_idx,
                        "filename": file_info["filename"],
                        "url": file_info["url"],
                        "reason": str(e),
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    })
                    print(f"[skip] file {file_idx:05d} skipped after {max_retry} retries")
                    return 0
                raise
            continue

    if skip_bad_file:
        append_jsonl(skipped_jsonl_path(output_root), {
            "file_idx": file_idx,
            "filename": file_info["filename"],
            "url": file_info["url"],
            "reason": str(last_err),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        print(f"[skip] file {file_idx:05d} skipped")
        return 0

    raise last_err


# =========================================================
# 10) Merge file shards
# =========================================================
def merge_file_shards(
    output_root: str,
    target_size: int,
    manifest_files: List[Dict[str, Any]],
    save_meta: bool = True,
):
    final_base = os.path.join(output_root, "base.fbin")
    final_query = os.path.join(output_root, "query.fbin")
    final_meta = os.path.join(output_root, "pairs_meta.jsonl")
    final_summary = os.path.join(output_root, "summary.json")

    total_n = 0
    base_d = None
    query_d = None
    selected_shards = []

    skipped = load_skipped_file_ids(output_root)

    for fi in manifest_files:
        file_idx = fi["file_idx"]
        if file_idx in skipped:
            continue

        p = file_shard_paths(output_root, file_idx)
        if not os.path.exists(p["summary"]):
            continue

        summary = load_json(p["summary"])
        if not summary.get("complete", False):
            continue

        bn, bd = read_fbin_header(p["base"])
        qn, qd = read_fbin_header(p["query"])

        if bn != qn:
            raise RuntimeError(f"base/query count mismatch in file shard {file_idx:05d}")

        if base_d is None:
            base_d = bd
        elif base_d != bd:
            raise RuntimeError(f"base dim mismatch in file shard {file_idx:05d}")

        if query_d is None:
            query_d = qd
        elif query_d != qd:
            raise RuntimeError(f"query dim mismatch in file shard {file_idx:05d}")

        take = min(bn, target_size - total_n)
        if take <= 0:
            break

        if take != bn:
            raise RuntimeError("Unexpected partial merge state; last file shard should already be truncated.")

        selected_shards.append((file_idx, p, bn))
        total_n += bn

        if total_n >= target_size:
            break

    if total_n != target_size:
        raise RuntimeError(f"merged total_n={total_n} != target_size={target_size}")

    with open(final_base, "wb") as f:
        np.array([target_size, base_d], dtype=np.int32).tofile(f)
        for _, p, _ in tqdm(selected_shards, desc="merge base"):
            copy_fbin_body(p["base"], f)

    with open(final_query, "wb") as f:
        np.array([target_size, query_d], dtype=np.int32).tofile(f)
        for _, p, _ in tqdm(selected_shards, desc="merge query"):
            copy_fbin_body(p["query"], f)

    if save_meta:
        with open(final_meta, "w", encoding="utf-8") as out_f:
            for _, p, _ in tqdm(selected_shards, desc="merge meta"):
                if os.path.exists(p["meta"]):
                    with open(p["meta"], "r", encoding="utf-8") as in_f:
                        shutil.copyfileobj(in_f, out_f, length=8 << 20)

    summary = {
        "target_size": target_size,
        "num_file_shards_used": len(selected_shards),
        "num_skipped_files": len(skipped),
        "base_fbin": final_base,
        "query_train_fbin": final_query,
        "meta_jsonl": final_meta if save_meta else "",
        "base_dim": base_d,
        "query_dim": query_d,
        "merged": True,
    }
    save_json(final_summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# =========================================================
# 11) Main orchestration
# =========================================================
def run_file_sharded_pipeline(
    output_root: str,
    target_size: int,
    clip_model: str,
    batch_size: int,
    device: str,
    dataset_name: str,
    split: str,
    language: Optional[str],
    save_meta: bool = True,
    merge_only: bool = False,
    file_retry: int = 5,
    retry_sleep: int = 10,
    skip_bad_file: bool = True,
):
    ensure_dir(output_root)
    ensure_dir(os.path.join(output_root, "file_shards"))

    manifest_files = get_or_build_manifest(output_root, dataset_name, split)

    if merge_only:
        merge_file_shards(output_root, target_size, manifest_files, save_meta=save_meta)
        return

    # CLIP 只加载一次
    model, preprocess = clip.load(clip_model, device=device)
    model.eval()

    skipped = load_skipped_file_ids(output_root)
    global_written = 0

    for fi in manifest_files:
        if global_written >= target_size:
            break

        file_idx = fi["file_idx"]

        if file_idx in skipped:
            print(f"[skip] file shard {file_idx:05d} already marked skipped")
            continue

        p = file_shard_paths(output_root, file_idx)
        if os.path.exists(p["summary"]):
            summary = load_json(p["summary"])
            pw = int(summary.get("pairs_written", 0))
            if summary.get("complete", False):
                global_written += pw
                print(f"[resume] file shard {file_idx:05d} complete, pairs={pw:,}, global_written={global_written:,}")
                continue
            else:
                # partial file shard
                global_written += pw
                print(f"[resume] file shard {file_idx:05d} partial, pairs={pw:,}, global_written={global_written:,}")

        if global_written >= target_size:
            break

        print(f"[run] processing file shard {file_idx:05d}, global_written={global_written:,}, target={target_size:,}")

        written_in_this_file = process_one_parquet_file_with_retry(
            output_root=output_root,
            file_idx=file_idx,
            file_info=fi,
            global_target_size=target_size,
            global_written_before=global_written,
            model=model,
            preprocess=preprocess,
            batch_size=batch_size,
            device=device,
            dataset_name=dataset_name,
            language=language,
            save_meta=save_meta,
            max_retry=file_retry,
            retry_sleep=retry_sleep,
            skip_bad_file=skip_bad_file,
        )

        global_written += written_in_this_file
        print(f"[run] file shard {file_idx:05d} done, global_written={global_written:,}")

    if global_written < target_size:
        raise RuntimeError(
            f"Only wrote {global_written:,}/{target_size:,} pairs in total. "
            f"You may need fewer target pairs or fewer skipped files."
        )

    print("[run] enough file shards done, start merge")
    merge_file_shards(output_root, target_size, manifest_files, save_meta=save_meta)


# =========================================================
# 12) CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--clip_model", type=str, default="ViT-B/32",
                        help='CLIP model name like "ViT-B/32" or local .pt path')
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--dataset_name", type=str, default="wikimedia/wit_base")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--target_size", type=int, default=1_000_000)

    parser.add_argument("--merge_only", action="store_true")

    parser.add_argument("--http_proxy", type=str, default="")
    parser.add_argument("--https_proxy", type=str, default="")
    parser.add_argument("--hf_token", type=str, default="")
    parser.add_argument("--hf_home", type=str, default="")

    parser.add_argument("--file_retry", type=int, default=5)
    parser.add_argument("--retry_sleep", type=int, default=10)
    parser.add_argument("--skip_bad_file", action="store_true")

    parser.add_argument("--no_meta", action="store_true")

    args = parser.parse_args()
    print(f"device = {args.device}")

    language = None if args.language.lower() in {"none", "all", "*"} else args.language

    setup_network_env(
        http_proxy=args.http_proxy,
        https_proxy=args.https_proxy,
        hf_token=args.hf_token,
        hf_home=args.hf_home,
    )

    try:
        run_file_sharded_pipeline(
            output_root=args.output_dir,
            target_size=args.target_size,
            clip_model=args.clip_model,
            batch_size=args.batch_size,
            device=args.device,
            dataset_name=args.dataset_name,
            split=args.split,
            language=language,
            save_meta=(not args.no_meta),
            merge_only=args.merge_only,
            file_retry=args.file_retry,
            retry_sleep=args.retry_sleep,
            skip_bad_file=args.skip_bad_file,
        )
    except FileInterrupted as e:
        print(f"[interrupt] {e}")
        print("[interrupt] safe to rerun the same command; it will resume from the current parquet file.")
        os._exit(2)


if __name__ == "__main__":
    main()