#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Cut LRW-style word clips out of the LRS3 pretrain mouth ROIs.

Every occurrence of a vocabulary word becomes a 29-frame (1.16 s) window of the mouth
ROI, centred on the word's midpoint, as in LRW. The clips are written in the layout
that lipreading/dataset.py (MyDataset) already reads:

    <data-out>/<WORD>/<partition>/<WORD>_<video>_<clip>_<start_frame>.npz   key 'data', uint8 [29, 96, 96]
    <ann-out>/<WORD>/<partition>/<same name>.txt                            line 5 = 'Duration: <word seconds>'

Partitions are assigned per video (train / val / test, see lrs3_words.video_partition), so no
speaker clip leaks across partitions. Each word is capped per partition with a seeded random
subsample: --max-train-per-word for train and --max-eval-per-word for val and test.

Example:
    python preprocessing/lrs3/build_word_clips.py \
        --labels labels/LRS3_500WordsSortedList.txt \
        --roi-root /scratch_hdd/LRS3/roi_root \
        --data-out /scratch_hdd/LRS3/word_clips/visual_data \
        --ann-out /scratch_hdd/LRS3/word_clips/annotations
"""

import os
import json
import random
import shutil
import argparse
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from lrs3_words import (WINDOW_FRAMES, read_word_alignment, is_candidate,
                        video_partition, window_start_frame)

_VOCAB = None
_PERCENTS = None


def load_args():
    parser = argparse.ArgumentParser(description='Build LRW-style word clips from LRS3 pretrain')
    parser.add_argument('--labels', required=True, help='label file (one word per line), e.g. labels/LRS3_500WordsSortedList.txt')
    parser.add_argument('--roi-root', default='/scratch_hdd/LRS3/roi_root',
                        help='output root of preprocess.py: processed/mouths/<split>/... and manifests/')
    parser.add_argument('--transcript-root', default=None, help='defaults to <roi-root>/transcripts')
    parser.add_argument('--split', default='pretrain', help='only pretrain has word alignments')
    parser.add_argument('--data-out', required=True, help='directory for the mouth clips (MyDataset --data-dir)')
    parser.add_argument('--ann-out', required=True, help='directory for the annotations (MyDataset --annonation-direc)')
    parser.add_argument('--max-train-per-word', type=int, default=250, help='train clips kept per word (250 fits the disk budget, see README)')
    parser.add_argument('--max-eval-per-word', type=int, default=50, help='val and test clips kept per word (LRW uses 50)')
    parser.add_argument('--val-percent', type=int, default=5, help='percentage of videos held out for val')
    parser.add_argument('--test-percent', type=int, default=5, help='percentage of videos held out for test')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 8)
    parser.add_argument('--overwrite', action='store_true',
                        help='delete existing --data-out and --ann-out before writing (needed to rebuild in place, '
                             'otherwise clips from an earlier run would stay in the training set)')
    args = parser.parse_args()
    if args.transcript_root is None:
        args.transcript_root = os.path.join(args.roi_root, 'transcripts')
    for folder in (args.data_out, args.ann_out):
        if os.path.isdir(folder) and os.listdir(folder):
            if not args.overwrite:
                raise SystemExit(f'{folder} is not empty; pass --overwrite to rebuild it')
            shutil.rmtree(folder)
    return args


def load_roi_frames(roi_root, split):
    """Map 'video/clip' -> number of mouth frames, for every ROI preprocess.py finished."""
    frames = {}
    manifest = Path(roi_root, 'manifests', 'preprocess_archive.jsonl')
    with open(manifest, encoding='utf-8') as handle:
        for line in handle:
            record = json.loads(line)
            if record.get('ok') and record.get('split') == split:
                frames[record['relative_mp4'][:-len('.mp4')]] = record['mouth_frames']
    return frames


def _init_scan(vocab, percents):
    global _VOCAB, _PERCENTS
    _VOCAB = vocab
    _PERCENTS = percents


def scan_roi(task):
    """Word occurrences that fit a window inside one ROI.

    Returns ([(word, partition, video, clip, start_frame, word_start, word_end)], n_out_of_range).
    """
    transcript_path, key, n_frames = task
    video, clip = key.split('/')
    partition = video_partition(video, *_PERCENTS)
    occurrences = []
    out_of_range = 0
    for word, start, end in read_word_alignment(transcript_path):
        if word not in _VOCAB or not is_candidate(word, start, end):
            continue
        first = window_start_frame(start, end)
        if first < 0 or first + WINDOW_FRAMES > n_frames:
            out_of_range += 1
            continue
        occurrences.append((word, partition, video, clip, first, start, end))
    return occurrences, out_of_range


def select_occurrences(occurrences, caps, seed):
    """Random per-(partition, word) subsample. Sorting keys keeps the draw reproducible."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for occurrence in occurrences:
        groups[(occurrence[1], occurrence[0])].append(occurrence)
    selected = []
    for key in sorted(groups):
        items = groups[key]
        cap = caps[key[0]]
        if len(items) > cap:
            items = rng.sample(items, cap)
        selected.extend(items)
    return selected


def write_clips(task):
    """Cut every selected window out of one sentence-level ROI and write it with its annotation."""
    mouth_path, items, data_out, ann_out = task
    mouths = np.load(mouth_path)['data']
    rows = []
    for word, partition, video, clip, first, start, end in items:
        window = mouths[first:first + WINDOW_FRAMES]
        if window.shape[0] != WINDOW_FRAMES:
            continue
        name = f'{word}_{video}_{clip}_{first:04d}'
        data_dir = Path(data_out, word, partition)
        ann_dir = Path(ann_out, word, partition)
        data_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(data_dir / f'{name}.npz', data=window.astype(np.uint8))
        with open(ann_dir / f'{name}.txt', 'w', encoding='utf-8') as handle:
            handle.write(f'Target: {word}\n')
            handle.write(f'Source: {video}/{clip}\n')
            handle.write(f'Start: {start:.3f}\n')
            handle.write(f'End: {end:.3f}\n')
            handle.write(f'Duration: {end - start:.3f}\n')
        rows.append((partition, word, video, clip, first, start, end, str(data_dir / f'{name}.npz')))
    return rows


def main():
    args = load_args()
    with open(args.labels, encoding='utf-8') as handle:
        vocab = set(line.strip() for line in handle if line.strip())
    percents = (args.val_percent, args.test_percent)
    frames = load_roi_frames(args.roi_root, args.split)
    print(f'ROIs with a successful preprocessing manifest entry: {len(frames):,}')
    print(f'Vocabulary: {len(vocab)} words')

    scan_tasks = [(str(Path(args.transcript_root, args.split, key + '.txt')), key, n)
                  for key, n in sorted(frames.items())]
    occurrences = []
    out_of_range = 0
    with Pool(args.workers, initializer=_init_scan, initargs=(vocab, percents)) as pool:
        for index, (found, skipped) in enumerate(pool.imap(scan_roi, scan_tasks, chunksize=64), 1):
            occurrences.extend(found)
            out_of_range += skipped
            if index % 20000 == 0:
                print(f'  scanned {index:,}/{len(scan_tasks):,} ROIs', flush=True)
    print(f'Word occurrences inside the window: {len(occurrences):,}  (outside the ROI, dropped: {out_of_range:,})')

    caps = {'train': args.max_train_per_word, 'val': args.max_eval_per_word, 'test': args.max_eval_per_word}
    selected = select_occurrences(occurrences, caps, args.seed)
    available = defaultdict(int)
    for occurrence in occurrences:
        available[occurrence[1]] += 1
    chosen = defaultdict(int)
    for occurrence in selected:
        chosen[occurrence[1]] += 1
    for partition in ('train', 'val', 'test'):
        print(f'  {partition:5s}: {available[partition]:,} occurrences, {chosen[partition]:,} selected (cap {caps[partition]}/word)')

    by_roi = defaultdict(list)
    for occurrence in selected:
        by_roi[(occurrence[2], occurrence[3])].append(occurrence)
    write_tasks = []
    for (video, clip), items in sorted(by_roi.items()):
        mouth_path = Path(args.roi_root, 'processed', 'mouths', args.split, video, clip + '.npz')
        write_tasks.append((str(mouth_path), items, args.data_out, args.ann_out))

    rows = []
    with Pool(args.workers) as pool:
        for index, written in enumerate(pool.imap_unordered(write_clips, write_tasks, chunksize=16), 1):
            rows.extend(written)
            if index % 5000 == 0:
                print(f'  wrote clips from {index:,}/{len(write_tasks):,} ROIs', flush=True)

    rows.sort()
    manifest_path = Path(args.data_out, 'word_clips.tsv')
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w', encoding='utf-8') as handle:
        handle.write('partition\tword\tvideo\tclip\tstart_frame\tword_start\tword_end\tpath\n')
        for partition, word, video, clip, first, start, end, path in rows:
            handle.write(f'{partition}\t{word}\t{video}\t{clip}\t{first}\t{start:.3f}\t{end:.3f}\t{path}\n')

    written_per_partition = defaultdict(int)
    words_without_train = set(vocab)
    for partition, word, *_ in rows:
        written_per_partition[partition] += 1
        if partition == 'train':
            words_without_train.discard(word)
    for partition in ('train', 'val', 'test'):
        print(f'Wrote {written_per_partition[partition]:,} {partition} clips')
    if words_without_train:
        print(f'WARNING: {len(words_without_train)} vocabulary words have no train clip, e.g. {sorted(words_without_train)[:5]}')
    print(f'Manifest: {manifest_path}')


if __name__ == '__main__':
    main()
