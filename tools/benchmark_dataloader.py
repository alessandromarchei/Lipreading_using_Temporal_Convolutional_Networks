#!/usr/bin/env python3
"""Benchmark the raw (.npz + .txt files) and LMDB data backends, CPU only.

Both backends read the same clips in the same order, with the training preprocessing of the
partition and the same RNG seed. Each (backend, workers) run is timed on a cold page cache
(the files are dropped from the cache with posix_fadvise first, when --drop-cache is set) and
then again on a warm cache.

Run from the repository root:
    python tools/benchmark_dataloader.py --raw-data-dir <MOUTHS> --annonation-direc <LRW> \
        --lmdb-dir <LMDB_DIR> --label-path labels/500WordsSortedList.txt --drop-cache
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lipreading.dataset as dataset_module  # noqa: E402
from lipreading.dataset import MyDataset, pad_packed_collate  # noqa: E402
from lipreading.dataloaders import get_preprocessing_pipelines  # noqa: E402


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--raw-data-dir', default=None)
    p.add_argument('--annonation-direc', default=None)
    p.add_argument('--lmdb-dir', default=None)
    p.add_argument('--label-path', default='./labels/500WordsSortedList.txt')
    p.add_argument('--partition', default='train', choices=['train', 'val', 'test'])
    p.add_argument('--workers', type=int, nargs='+', default=[0, 2, 4, 8])
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-batches', type=int, default=50, help='timed batches per run')
    p.add_argument('--warmup-batches', type=int, default=5, help='batches excluded from the throughput')
    p.add_argument('--drop-cache', action='store_true', help='evict the data files from the page cache before the cold run')
    p.add_argument('--pin-memory', action='store_true', help='pin batches (needs CUDA; off by default to stay CPU only)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--lmdb-readahead', choices=['off', 'on', 'both'], default='on',
                   help='LMDB readahead setting(s) to benchmark (the loader default is on)')
    return p.parse_args()


def evict(paths):
    """Drop clean pages of these files from the page cache (no root needed). Returns False if unsupported."""
    if not hasattr(os, 'posix_fadvise'):
        return False
    for path in paths:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    return True


def files_of(ds, rels):
    if ds._data_format == 'lmdb':
        return [ds._lmdb_path]
    out = []
    for rel in rels:
        out.append(os.path.join(ds._data_dir, rel))
        if ds._annonation_direc:
            out.append(os.path.splitext(os.path.join(ds._annonation_direc, rel))[0] + '.txt')
    return out


def run(ds, indices, args, workers):
    loader = DataLoader(Subset(ds, indices), batch_size=args.batch_size, shuffle=False,
                        collate_fn=pad_packed_collate, num_workers=workers, pin_memory=args.pin_memory,
                        persistent_workers=workers > 0)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    start = time.perf_counter()
    stamps = []
    n_samples = 0
    for data, lengths, labels in loader:
        stamps.append(time.perf_counter())
        if len(stamps) > args.warmup_batches:
            n_samples += len(labels)
    del loader  # shuts the persistent workers down
    first = stamps[0] - start
    timed = np.diff([stamps[args.warmup_batches - 1] if args.warmup_batches else start] + stamps[args.warmup_batches:])
    total = timed.sum()
    return {'first': first, 'lat': timed.mean(), 'bps': len(timed) / total, 'sps': n_samples / total}


def main():
    args = arguments()
    torch.set_num_threads(1)
    preprocessing = get_preprocessing_pipelines('video')[args.partition]
    common = dict(modality='video', data_partition=args.partition, label_fp=args.label_path,
                  preprocessing_func=preprocessing, data_suffix='.npz')
    backends = {}
    if args.raw_data_dir:
        t = time.perf_counter()
        ds = MyDataset(data_dir=args.raw_data_dir, annonation_direc=args.annonation_direc, data_format='raw', **common)
        backends['raw'] = (ds, time.perf_counter() - t, {'/'.join(v[0].split('/')[-3:]): i for i, v in ds.list.items()})
    if args.lmdb_dir:
        t = time.perf_counter()
        ds = MyDataset(data_dir=args.lmdb_dir, data_format='lmdb', **common)
        backends['lmdb'] = (ds, time.perf_counter() - t, {v[0]: i for i, v in ds.list.items()})
    assert backends, 'give --raw-data-dir and/or --lmdb-dir'

    # -- the same clips, in the same order, for every backend
    rels = sorted(set.intersection(*[set(b[2]) for b in backends.values()]))
    need = args.batch_size * (args.num_batches + args.warmup_batches)
    rels = random.Random(args.seed).sample(rels, min(need, len(rels)))
    assert len(rels) > args.batch_size * args.warmup_batches, 'not enough clips for the warm-up batches'

    print(f'partition {args.partition}, {len(rels)} clips, batch {args.batch_size}, '
          f'{args.warmup_batches} warm-up batches, torch {torch.__version__}, {os.cpu_count()} CPUs')
    for name, (ds, init, _) in backends.items():
        print(f'{name:5s} dataset init: {init:.3f} s ({len(ds)} clips in partition)')
    print()
    header = f'{"backend":7s} {"workers":>7s} {"cache":5s} {"1st batch s":>11s} {"batch ms":>9s} {"batch/s":>8s} {"sample/s":>9s}'
    print(header)
    print('-' * len(header))
    runs = []
    for name in backends:
        if name == 'lmdb':
            runs += [('lmdb', ra) for ra in {'off': [False], 'on': [True], 'both': [False, True]}[args.lmdb_readahead]]
        else:
            runs.append((name, None))
    for workers in args.workers:
        for name, readahead in runs:
            ds, _, index = backends[name]
            label = name if readahead is None else f'{name}{"+ra" if readahead else ""}'
            indices = [index[r] for r in rels]
            caches = ['cold', 'warm'] if args.drop_cache else ['warm']
            for cache in caches:
                # handles are reopened with this readahead; closing also unmaps the file so it can be evicted
                dataset_module._close_lmdb_envs()
                if readahead is not None:
                    dataset_module.LMDB_READAHEAD = readahead
                if cache == 'cold':
                    evict(files_of(ds, rels))
                else:  # read every benchmarked clip once, so the warm run hits the page cache
                    for i in indices:
                        ds.load_data(ds.list[i][0])
                        if ds._data_format == 'raw' and ds._annonation_direc:
                            ds._get_duration(ds.list[i][0])
                r = run(ds, indices, args, workers)
                print(f'{label:7s} {workers:7d} {cache:5s} {r["first"]:11.3f} {r["lat"] * 1e3:9.1f} '
                      f'{r["bps"]:8.2f} {r["sps"]:9.1f}')

if __name__ == '__main__':
    main()
