#!/usr/bin/env python3
"""Pack LRW mouth .npz files and annotation durations into 3 single-file LMDBs.

pip install lmdb tqdm numpy

Layout of <output-dir>/<split>.lmdb (read by lipreading/dataset.py with --data-format lmdb):
    b"v:<WORD>/<STEM>"  -> original .npz bytes, unchanged
    b"__index__"        -> JSON [[key, label_id, duration_seconds], ...]
    b"__labels__"       -> JSON list of words (label_id indexes it)
    b"__format__"       -> b"lrw-npz-v1"
"""
import argparse
import json
import os
import re
from pathlib import Path

import lmdb
from tqdm import tqdm

SPLITS = ("train", "val", "test")
DURATION_RE = re.compile(r"^Duration:\s*([0-9]+(?:\.[0-9]+)?)\s+seconds\s*$", re.I | re.M)


def arguments():
    p = argparse.ArgumentParser(description="Pack LRW .npz and .txt annotations as LMDB")
    p.add_argument("--mouths-dir", type=Path, required=True, help="ROOT/WORD/SPLIT/VIDEO.npz")
    p.add_argument("--annotations-dir", type=Path, required=True, help="ROOT/WORD/SPLIT/VIDEO.txt")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--labels-file", type=Path, default=None, help="Original label list, one word per line, in training order")
    p.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    p.add_argument("--batch-size", type=int, default=128, help="Samples per LMDB write transaction")
    return p.parse_args()


def get_labels(root, labels_file):
    if labels_file:
        labels = [line.strip() for line in labels_file.read_text().splitlines() if line.strip()]
    else:
        labels = sorted(p.name for p in root.iterdir() if p.is_dir())
        print("[NOTE] No --labels-file supplied; IDs use alphabetical order.")
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("Labels list is empty or contains duplicates")
    return labels


def read_duration(path):
    match = DURATION_RE.search(path.read_text(encoding="utf-8-sig"))
    if not match:
        raise ValueError(f"Missing/invalid Duration field: {path}")
    return float(match.group(1))


def collect(mouths, annotations, labels, split):
    entries = []
    total_bytes = 0
    for label_id, word in enumerate(labels):
        folder = mouths / word / split
        if not folder.is_dir():
            continue
        for npz_path in sorted(folder.glob("*.npz")):
            txt_path = annotations / word / split / f"{npz_path.stem}.txt"
            if not txt_path.is_file():
                raise FileNotFoundError(f"Annotation missing for {npz_path}: {txt_path}")
            key = f"{word}/{npz_path.stem}"
            entries.append((key, label_id, npz_path, txt_path))
            total_bytes += npz_path.stat().st_size
    return entries, total_bytes


def pack_split(split, entries, total_bytes, labels, output, batch_size):
    if not entries:
        print(f"[{split}] No samples found; skipping")
        return
    final_path = output / f"{split}.lmdb"
    build_path = output / f"{split}.building.lmdb"
    if final_path.exists() or build_path.exists():
        raise FileExistsError(f"Output exists: {final_path} or {build_path}. Move/remove it before rebuilding")

    # LMDB map_size is virtual address space ceiling, not actual disk usage on Linux.
    map_size = int(total_bytes * 1.3) + 512 * 1024 * 1024
    print(f"[{split}] {len(entries):,} samples, input {total_bytes / 2**30:.2f} GiB, map_size {map_size / 2**30:.2f} GiB")
    env = lmdb.open(
        str(build_path), subdir=False, map_size=map_size,
        max_dbs=1, writemap=False, map_async=False, lock=True,
    )
    index = []
    try:
        txn = env.begin(write=True)
        try:
            for i, (key, label_id, npz_path, txt_path) in enumerate(tqdm(entries, desc=split, unit="video")):
                duration = read_duration(txt_path)
                # Raw pre-compressed NPZ bytes: no decompression/recompression at packaging time.
                payload = npz_path.read_bytes()
                txn.put(b"v:" + key.encode("utf-8"), payload, overwrite=False)
                index.append([key, label_id, duration])
                if (i + 1) % batch_size == 0:
                    txn.commit()
                    txn = env.begin(write=True)
            # Index and label vocabulary are stored inside the same LMDB file.
            txn.put(b"__index__", json.dumps(index, separators=(",", ":")).encode("utf-8"))
            txn.put(b"__labels__", json.dumps(labels, separators=(",", ":")).encode("utf-8"))
            txn.put(b"__format__", b"lrw-npz-v1")
            txn.commit()
        except BaseException:
            txn.abort()
            raise
        env.sync()
    finally:
        env.close()

    os.replace(build_path, final_path)
    # A future writer may recreate the lock file; read-only Kaggle access uses lock=False.
    lock_path = Path(str(build_path) + "-lock")
    if lock_path.exists():
        lock_path.unlink()
    print(f"[{split}] Saved: {final_path} ({final_path.stat().st_size / 2**30:.2f} GiB)")


def main():
    args = arguments()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    for folder in (args.mouths_dir, args.annotations_dir):
        if not folder.is_dir():
            raise NotADirectoryError(folder)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = get_labels(args.mouths_dir, args.labels_file)
    for split in args.splits:
        entries, total_bytes = collect(args.mouths_dir, args.annotations_dir, labels, split)
        pack_split(split, entries, total_bytes, labels, args.output_dir, args.batch_size)
    print("Done.")


if __name__ == "__main__":
    main()
