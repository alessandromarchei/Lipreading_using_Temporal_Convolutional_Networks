"""Shared helpers for cutting LRW-style word clips out of LRS3 pretrain.

LRS3 pretrain transcripts carry a word alignment table after the plain text:

    Text:  HOWEVER WHEN I TURNED 15 1 2 ...
    Conf:  3

    WORD START END ASDSCORE
    HOWEVER 0.08 0.51 9.8
    WHEN 0.51 0.60 5.0

Times are seconds from the start of the clip. The mouth ROIs are at 25 fps, so
frame i covers t = i / 25 s. trainval and test transcripts have no alignment
table, so word-level clips can only be built from pretrain.
"""

import hashlib
import re

FPS = 25
WINDOW_FRAMES = 29                       # 1.16 s, the LRW clip length
CENTRE_INDEX = WINDOW_FRAMES // 2        # index of the word midpoint inside the window
MAX_WORD_SECONDS = WINDOW_FRAMES / FPS   # longer words cannot be centred in the window

WORD_RE = re.compile(r"^[A-Z]+(?:'[A-Z]+)?$")   # drops digits and punctuation tokens such as "15" or "-"
ALIGNMENT_HEADER = "WORD START END"


def read_word_alignment(transcript_path):
    """Return [(WORD, start, end), ...]; empty if the transcript has no alignment table."""
    words = []
    in_table = False
    with open(transcript_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not in_table:
                in_table = line.strip().startswith(ALIGNMENT_HEADER)
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                start, end = float(parts[1]), float(parts[2])
            except ValueError:
                continue
            words.append((parts[0].upper(), start, end))
    return words


def is_candidate(word, start, end):
    """A word can become a clip if it is alphabetic and short enough to fit the window."""
    return WORD_RE.match(word) is not None and 0.0 < (end - start) <= MAX_WORD_SECONDS


def video_partition(video_id, val_percent=5, test_percent=5):
    """Deterministic video-level split, so the clips of one video never cross partitions."""
    bucket = int(hashlib.md5(video_id.encode("utf-8")).hexdigest(), 16) % 100
    if bucket < test_percent:
        return "test"
    if bucket < test_percent + val_percent:
        return "val"
    return "train"


def window_start_frame(start, end):
    """First frame of the 29-frame window whose centre frame is the word midpoint."""
    centre_frame = int(round((start + end) / 2.0 * FPS))
    return centre_frame - CENTRE_INDEX
