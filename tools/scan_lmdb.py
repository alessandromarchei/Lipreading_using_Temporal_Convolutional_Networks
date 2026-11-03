#!/usr/bin/env python3
"""Check every record of the LMDB files and list the clips that cannot be trained on.

A record is bad when it is empty (a 0-byte .npz was packed), cannot be read by np.load, has no
'data' array, or its array is not uint8 [T, H, W] with T > 0 and H, W >= the 88 px crop.
With --face-list it also reports clips of the authors' list (lrw500_detected_face.csv) that are
not in the LMDB, and clips in the LMDB that are not in the list.

The bad clips are written to --out, one WORD/partition/CLIP per line. Pass that file to main.py
with --exclude-list to train without them.

    python tools/scan_lmdb.py --lmdb-dir /kaggle/input/<LMDB> --out /kaggle/working/bad_clips.txt \
        --face-list preprocessing/lrw500_detected_face.csv --workers 4
"""
import argparse
import collections
import io
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lipreading.dataset import _get_lmdb_env, LMDB_FORMAT  # noqa: E402

MIN_SIZE = 88  # CenterCrop / RandomCrop size in lipreading/dataloaders.py


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lmdb-dir', required=True)
    p.add_argument('--partitions', nargs='+', default=['train', 'val', 'test'])
    p.add_argument('--out', required=True, help='file to write the bad clips to')
    p.add_argument('--face-list', default=None, help='lrw500_detected_face.csv, to report missing clips')
    p.add_argument('--workers', type=int, default=os.cpu_count())
    return p.parse_args()


def check(task):
    path, keys = task
    bad, frames = [], collections.Counter()
    with _get_lmdb_env(path).begin() as txn:
        for key in keys:
            payload = txn.get(('v:' + key).encode('utf-8'))
            if payload is None:
                bad.append((key, 'record missing'))
                continue
            if len(payload) == 0:
                bad.append((key, 'empty record (0-byte .npz)'))
                continue
            try:
                with np.load(io.BytesIO(payload)) as npz:
                    data = npz['data']
            except Exception as e:  # noqa: BLE001 - every failure makes the clip unusable
                bad.append((key, f'unreadable: {type(e).__name__}: {e}'))
                continue
            if data.dtype != np.uint8 or data.ndim != 3 or data.shape[0] == 0 \
                    or data.shape[1] < MIN_SIZE or data.shape[2] < MIN_SIZE:
                bad.append((key, f'bad array {data.dtype}{data.shape}'))
                continue
            frames[data.shape[0]] += 1
    return bad, frames


def main():
    args = arguments()
    expected = collections.defaultdict(set)
    if args.face_list:
        for line in open(args.face_list).read().splitlines():
            word, partition, clip = line.split(',')[0].split('/')
            expected[partition].add(f'{word}/{clip}')

    out_lines = []
    for partition in args.partitions:
        path = os.path.join(args.lmdb_dir, f'{partition}.lmdb')
        with _get_lmdb_env(path).begin() as txn:
            assert txn.get(b'__format__') == LMDB_FORMAT, f'{path} is not a {LMDB_FORMAT!r} LMDB'
            keys = [k for k, _, _ in json.loads(txn.get(b'__index__'))]
        t = time.time()
        chunks = [(path, keys[i:i + 500]) for i in range(0, len(keys), 500)]
        bad, frames = [], collections.Counter()
        with Pool(args.workers) as pool:
            for b, f in pool.imap_unordered(check, chunks):
                bad += b
                frames += f
        print(f'[{partition}] {len(keys)} records checked in {time.time() - t:.0f} s, {len(bad)} bad')
        for key, reason in sorted(bad):
            print(f'    {partition}  {key}: {reason}')
        print(f'[{partition}] frames per clip: {dict(sorted(frames.items()))}')
        if args.face_list:
            missing = sorted(expected[partition] - set(keys))
            extra = sorted(set(keys) - expected[partition])
            print(f'[{partition}] vs {args.face_list}: {len(missing)} missing from the LMDB, {len(extra)} not in the list')
            for key in missing[:20]:
                print(f'    missing  {key}')
        for key, _ in sorted(bad):
            word, clip = key.split('/')
            out_lines.append(f'{word}/{partition}/{clip}')

    with open(args.out, 'w') as f:
        f.write(''.join(line + '\n' for line in out_lines))
    print(f'\n{len(out_lines)} bad clips written to {args.out}')


if __name__ == '__main__':
    main()
