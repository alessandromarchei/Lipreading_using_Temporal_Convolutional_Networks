#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Choose the word vocabulary for LRS3 word-level lipreading.

Counts every alignable word in the pretrain transcripts that could fit a 1.16 s window,
keeps the K most frequent ones and writes an LRW-style label file (one word per line,
alphabetically sorted, as labels/500WordsSortedList.txt).

Example:
    python preprocessing/lrs3/build_vocab.py \
        --transcript-root /scratch_hdd/LRS3/organized/transcripts \
        --num-words 500 \
        --labels-out labels/LRS3_500WordsSortedList.txt \
        --stats-out labels/LRS3_word_counts.tsv
"""

import os
import argparse
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from lrs3_words import read_word_alignment, is_candidate


def count_transcript(path):
    return [word for word, start, end in read_word_alignment(path) if is_candidate(word, start, end)]


def load_args():
    parser = argparse.ArgumentParser(description='Choose the LRS3 word vocabulary')
    parser.add_argument('--transcript-root', required=True, help='directory holding <split>/<video>/<clip>.txt')
    parser.add_argument('--split', default='pretrain', help='LRS3 split with word alignments (only pretrain has them)')
    parser.add_argument('--num-words', type=int, default=500, help='vocabulary size K (500 matches LRW)')
    parser.add_argument('--labels-out', required=True, help='where to write the sorted label file')
    parser.add_argument('--stats-out', default=None, help='optional tsv with the counts of the top 2000 words')
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 8)
    return parser.parse_args()


def main():
    args = load_args()
    transcripts = sorted(Path(args.transcript_root, args.split).rglob('*.txt'))
    print(f'Transcripts in {args.split}: {len(transcripts):,}')

    counts = Counter()
    with Pool(args.workers) as pool:
        for index, words in enumerate(pool.imap_unordered(count_transcript, transcripts, chunksize=256), 1):
            counts.update(words)
            if index % 20000 == 0:
                print(f'  counted {index:,}/{len(transcripts):,} transcripts', flush=True)

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked) < args.num_words:
        raise SystemExit(f'Only {len(ranked)} distinct candidate words, cannot pick {args.num_words}')
    chosen = ranked[:args.num_words]
    vocabulary = sorted(word for word, _ in chosen)

    total = sum(counts.values())
    kept = sum(count for _, count in chosen)
    min_count = chosen[-1][1]
    print(f'Candidate word tokens: {total:,}   distinct words: {len(counts):,}')
    print(f'Top-{args.num_words} coverage: {kept / total:.1%} of tokens   '
          f'least frequent chosen word: {chosen[-1][0]} ({min_count:,} tokens)')
    print(f'Most frequent: {ranked[0][0]} ({ranked[0][1]:,})')

    Path(args.labels_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.labels_out, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(vocabulary) + '\n')
    print(f'Wrote {len(vocabulary)} labels to {args.labels_out}')

    if args.stats_out:
        vocab_set = set(vocabulary)
        with open(args.stats_out, 'w', encoding='utf-8') as handle:
            handle.write('rank\tword\ttokens\tin_vocab\n')
            for rank, (word, count) in enumerate(ranked[:2000], 1):
                handle.write(f'{rank}\t{word}\t{count}\t{int(word in vocab_set)}\n')
        print(f'Wrote counts to {args.stats_out}')


if __name__ == '__main__':
    main()
