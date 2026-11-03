#!/usr/bin/env python3
"""Check that --data-format lmdb gives the same samples as --data-format raw.

For every partition it checks:
  * the raw and LMDB sample sets are the same (and reports what is missing on either side)
  * every LMDB index entry has a record, with a consistent label id (--check-all-records)
  * on sampled clips: class index, instance id, word duration, decoded array (dtype, shape, values),
    the preprocessed tensor with the same RNG seed (train augmentations included), the boundary
    vector, and the collated batch.

Run from the repository root:
    python tools/check_lmdb_equivalence.py --raw-data-dir <MOUTHS> --annonation-direc <LRW> \
        --lmdb-dir <LMDB_DIR> --label-path labels/500WordsSortedList.txt
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lipreading.dataset import MyDataset, pad_packed_collate, _get_lmdb_env  # noqa: E402
from lipreading.dataloaders import get_preprocessing_pipelines  # noqa: E402


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--raw-data-dir', required=True, help='WORD/SPLIT/*.npz mouth ROIs (the original --data-dir)')
    p.add_argument('--annonation-direc', required=True, help='WORD/SPLIT/*.txt LRW annotations')
    p.add_argument('--lmdb-dir', required=True, help='folder with train.lmdb, val.lmdb, test.lmdb')
    p.add_argument('--label-path', default='./labels/500WordsSortedList.txt')
    p.add_argument('--partitions', nargs='+', default=['train', 'val', 'test'])
    p.add_argument('--num-samples', type=int, default=200, help='clips compared per partition, 0 = all')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--check-all-records', action='store_true', help='read every LMDB record, not only the sampled ones')
    return p.parse_args()


def rel_path(path):
    return '/'.join(path.split('/')[-3:])


def same_seed_call(fn, seed):
    random.seed(seed)
    np.random.seed(seed)
    return fn()


def check_partition(args, partition, preprocessing):
    problems = []
    kwargs = dict(modality='video', data_partition=partition, label_fp=args.label_path,
                  preprocessing_func=preprocessing[partition], data_suffix='.npz')
    raw = MyDataset(data_dir=args.raw_data_dir, annonation_direc=args.annonation_direc, data_format='raw', **kwargs)
    lmdb_ds = MyDataset(data_dir=args.lmdb_dir, data_format='lmdb', **kwargs)
    raw_b = MyDataset(data_dir=args.raw_data_dir, annonation_direc=args.annonation_direc, data_format='raw',
                      use_boundary=True, **kwargs)
    lmdb_b = MyDataset(data_dir=args.lmdb_dir, data_format='lmdb', use_boundary=True, **kwargs)

    raw_idx = {rel_path(v[0]): i for i, v in raw.list.items()}
    lmdb_idx = {v[0]: i for i, v in lmdb_ds.list.items()}
    only_raw = sorted(set(raw_idx) - set(lmdb_idx))
    only_lmdb = sorted(set(lmdb_idx) - set(raw_idx))
    print(f'[{partition}] raw {len(raw)} clips, lmdb {len(lmdb_ds)} clips, '
          f'only raw {len(only_raw)}, only lmdb {len(only_lmdb)}')
    if only_raw or only_lmdb:
        problems.append(f'sample sets differ: only raw {only_raw[:5]}..., only lmdb {only_lmdb[:5]}...')

    # -- index / record integrity
    env = _get_lmdb_env(lmdb_ds._lmdb_path)
    with env.begin() as txn:
        labels = json.loads(txn.get(b'__labels__'))
        index = json.loads(txn.get(b'__index__'))
        keys = [k for k, _, _ in index]
        if len(keys) != len(set(keys)):
            problems.append('duplicate keys in __index__')
        n_records = txn.stat()['entries'] - 3  # minus __index__, __labels__, __format__
        if n_records != len(index):
            problems.append(f'{n_records} records but {len(index)} index entries')
        label_order = [w for w in labels if w in lmdb_ds._labels]
        if [w for w in lmdb_ds._labels if w in labels] != label_order:
            print(f'[{partition}] NOTE: label order in LMDB differs from {args.label_path}; '
                  f'class ids are taken from {args.label_path}, as in the raw backend')
        if args.check_all_records:
            missing = [k for k in keys if txn.get(('v:' + k).encode('utf-8')) is None]
            print(f'[{partition}] checked all {len(keys)} records, {len(missing)} missing')
            if missing:
                problems.append(f'{len(missing)} index entries without record, e.g. {missing[:5]}')

    # -- per-sample comparison
    common = sorted(set(raw_idx) & set(lmdb_idx))
    rng = random.Random(args.seed)
    sample = common if args.num_samples == 0 or args.num_samples >= len(common) else rng.sample(common, args.num_samples)
    n_bad = 0
    for n, rel in enumerate(sample):
        i_r, i_l = raw_idx[rel], lmdb_idx[rel]
        f_r, f_l = raw.list[i_r][0], lmdb_ds.list[i_l][0]
        diffs = []
        if raw.list[i_r][1] != lmdb_ds.list[i_l][1]:
            diffs.append(f'label {raw.list[i_r][1]} vs {lmdb_ds.list[i_l][1]}')
        if raw.instance_ids[i_r] != lmdb_ds.instance_ids[i_l]:
            diffs.append('instance id')
        if raw._get_duration(f_r) != lmdb_ds._get_duration(f_l):
            diffs.append(f'duration {raw._get_duration(f_r)} vs {lmdb_ds._get_duration(f_l)}')
        a_r, a_l = raw.load_data(f_r), lmdb_ds.load_data(f_l)
        if a_r.dtype != a_l.dtype or a_r.shape != a_l.shape or not np.array_equal(a_r, a_l):
            diffs.append(f'decoded data {a_r.dtype}{a_r.shape} vs {a_l.dtype}{a_l.shape}')
        seed = args.seed * 100003 + n
        x_r, y_r = same_seed_call(lambda: raw[i_r], seed)
        x_l, y_l = same_seed_call(lambda: lmdb_ds[i_l], seed)
        if y_r != y_l or x_r.dtype != x_l.dtype or x_r.shape != x_l.shape or not np.array_equal(x_r, x_l):
            diffs.append(f'preprocessed sample {x_r.dtype}{x_r.shape} vs {x_l.dtype}{x_l.shape}')
        _, _, b_r = same_seed_call(lambda: raw_b[i_r], seed)
        _, _, b_l = same_seed_call(lambda: lmdb_b[i_l], seed)
        if not np.array_equal(b_r, b_l):
            diffs.append('boundary')
        if diffs:
            n_bad += 1
            problems.append(f'{rel}: ' + ', '.join(diffs))

    # -- collated batch with the same seed (sorting, padding, dtypes)
    batch_rel = sample[:8]
    if batch_rel:
        def batch(ds, idx):
            return pad_packed_collate([ds[idx[r]] for r in batch_rel])
        d_r, len_r, lab_r = same_seed_call(lambda: batch(raw, raw_idx), args.seed)
        d_l, len_l, lab_l = same_seed_call(lambda: batch(lmdb_ds, lmdb_idx), args.seed)
        if not (torch.equal(d_r, d_l) and len_r == len_l and torch.equal(lab_r, lab_l) and d_r.dtype == d_l.dtype):
            problems.append('collated batch differs')
    print(f'[{partition}] compared {len(sample)} clips, {n_bad} with differences')
    return problems


def main():
    args = arguments()
    preprocessing = get_preprocessing_pipelines('video')
    problems = {p: check_partition(args, p, preprocessing) for p in args.partitions}
    print()
    ok = True
    for partition, items in problems.items():
        if items:
            ok = False
            print(f'[{partition}] {len(items)} problem(s):')
            for item in items[:50]:
                print('   ', item)
        else:
            print(f'[{partition}] identical')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
