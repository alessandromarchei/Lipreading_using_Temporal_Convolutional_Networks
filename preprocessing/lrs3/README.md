# LRS3 word-level lipreading data

LRW needs a license, so this folder builds an LRW-style word dataset from LRS3 pretrain instead.
The LRS3 pretrain transcripts carry word alignments (`WORD START END ASDSCORE`), which is what LRW-style
clips need. The output is read by the existing `MyDataset` with no change to the loader.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Mouth ROIs for every pretrain clip | [extract_mouth_rois.py](extract_mouth_rois.py) | `roi_root/processed/mouths/pretrain/<video>/<clip>.npz`, uint8 `[T, 96, 96]` |
| 2. Choose the vocabulary | [build_vocab.py](build_vocab.py) | [labels/LRS3_500WordsSortedList.txt](../../labels/LRS3_500WordsSortedList.txt) |
| 3. Cut 1.16 s word clips | [build_word_clips.py](build_word_clips.py) | `word_clips/visual_data/<WORD>/<split>/*.npz` and `word_clips/annotations/<WORD>/<split>/*.txt` |
| 4. Train | `main.py` with [configs/lrs3_resnet18_mstcn.json](../../configs/lrs3_resnet18_mstcn.json) | `train_logs/...` |

[lrs3_words.py](lrs3_words.py) holds the shared parsing, window and split rules.

### 1. Mouth ROIs

[extract_mouth_rois.py](extract_mouth_rois.py) is the repo copy of the LRS3 preprocessing script
(`/scratch_hdd/LRS3/preprocess.py`). It runs face landmarks on each frame, smooths them over 12 frames and crops a
96x96 grayscale mouth patch, the same convention as the LRW preprocessing in [preprocessing/](../).
The copy differs from the script it was derived from in these ways:

- **Batched GPU landmark path.** `face_alignment.get_landmarks_from_batch` runs the network once per image, so the
  "batch" is not a batch on the GPU. The new `_detect_landmarks_batched` runs the FAN network on chunks of 48 frames and
  reproduces the library's single-face crop and decode. With batch size 1 it matches the library exactly.
- **Landmark stride 2** (`--landmark-stride`, default 2). Every second frame gets a detection and the rest are linearly
  interpolated. After the 12-frame smoothing that costs 0.38 px on average and 0.81 px at most, against a 96 px crop.
  Use `--landmark-stride 1` for every frame, at roughly double the GPU time.
- **fp32 by default.** `--landmark-fp16` gave 2x speed but produced 30+ px landmark jumps on some frames. Those jumps
  broke the crops, with about 13% of clips failing. fp32 gives at most 4 px of drift.
- **`--skip-audio`.** FLAC audio is about 0.2 MB per clip. It is skipped to fit the disk, and it can be regenerated from
  the MP4s with ffmpeg when the audio branch is needed.

Run it (resumable: finished clips are skipped):

```Shell
python preprocessing/lrs3/extract_mouth_rois.py \
    --ssd-root /scratch_hdd/LRS3/roi_root \
    --hdd-root /scratch_hdd/LRS3/roi_archive \
    --mean-face preprocessing/20words_mean_face.npy \
    --splits pretrain --scratch-root /dev/shm/lrs3_scratch --skip-audio
```

`--ssd-root` must contain `av_media/` and `transcripts/`. `roi_root` holds symlinks to
`/scratch_hdd/LRS3/organized/av_media` and `.../transcripts`, so the outputs do not mix into `organized/`.
Do not pass `--archive-source`: it would move the source MP4s.

Failures are logged to `roi_root/manifests/preprocess_failures.tsv`. They are mostly videos where the landmark
detector returns points outside the frame, so the crop fails for every clip of that video. A few videos account for
most of the failures.

Throughput on the RTX 5060 Ti is about 55 clips per minute, roughly 32 hours for the 103,635 pretrain clips. The
GPU is the bottleneck.

### 2. Vocabulary

```Shell
python preprocessing/lrs3/build_vocab.py \
    --transcript-root /scratch_hdd/LRS3/organized/transcripts --num-words 500 \
    --labels-out labels/LRS3_500WordsSortedList.txt --stats-out labels/LRS3_word_counts_top500.tsv
```

- A word counts if it is alphabetic (an apostrophe is allowed) and lasts at most 1.16 s, the length of a window.
  Digits such as `15` and punctuation tokens are dropped.
- Over pretrain there are 3.38M candidate tokens with 46,486 distinct words. The top 500 cover 74% of them.
  The rarest kept word, `COMPLETELY`, has 699 tokens, so each word can supply the per-word training cap.
- K = 500 matches the LRW setup, so the numbers are comparable in shape. A larger K would push the rarest words below
  the cap and make the classes harder.

### 3. Word clips

```Shell
python preprocessing/lrs3/build_word_clips.py \
    --labels labels/LRS3_500WordsSortedList.txt \
    --roi-root /scratch_hdd/LRS3/roi_root \
    --data-out /scratch_hdd/LRS3/word_clips/visual_data \
    --ann-out /scratch_hdd/LRS3/word_clips/annotations \
    --max-train-per-word 250 --max-eval-per-word 50
```

- **Window.** As in LRW, each occurrence is 29 frames (1.16 s at 25 fps) centred on the word's midpoint. The clip is
  stored as uint8 `[29, 96, 96]` in `<WORD>/<split>/<WORD>_<video>_<clip>_<start_frame>.npz`.
- **Annotation.** `annotations/<WORD>/<split>/<same name>.txt`. Line 5 is `Duration: <word seconds>`, which
  `MyDataset` reads to crop the training clip to the word.
- **Dropped occurrences.** A word whose 29-frame window would fall outside the sentence is dropped. That is about
  13% of occurrences, mostly the first and last words of a sentence.
- **Splits.** LRS3 gives word alignments only for pretrain. Trainval and test transcripts are plain sentences, so the
  official test set cannot be used at word level without a forced aligner. Instead, pretrain is split by video
  (hash of the video id): 90% train, 5% val, 5% test. Clips of one video never cross partitions, but speakers can
  appear in more than one partition.
- **Caps.** Train: 250 clips per word (`--max-train-per-word`). Val and test: 50 per word, the LRW protocol. The
  sampling is random with a fixed seed.
- **Rebuilding.** The builder refuses a non-empty output folder unless `--overwrite` is passed. Otherwise clips from
  an earlier run would stay in the training set.

Expected size, with 500 words: about 125k train clips and about 25k each for val and test, roughly 34 GB in total
(about 190 KB per clip, compressed).

### 4. Training

```Shell
CUDA_VISIBLE_DEVICES=0 python main.py --modality video \
    --config-path configs/lrs3_resnet18_mstcn.json \
    --data-dir /scratch_hdd/LRS3/word_clips/visual_data \
    --annonation-direc /scratch_hdd/LRS3/word_clips/annotations \
    --label-path labels/LRS3_500WordsSortedList.txt \
    --num-classes 500
```

The model is the visual-only LRW architecture: Conv3D front end, ResNet-18, and a multi-scale TCN back end. The
config is the LRW config with the LRS3 name. Resume with `--model-path <ckpt> --init-epoch <n>`.

### Feature extraction for the audio-visual separator

The 512-d per-frame features come from the trunk before the TCN, at 25 fps:

```Shell
python main.py --modality video --config-path configs/lrs3_resnet18_mstcn.json --num-classes 500 \
    --data-dir /scratch_hdd/LRS3/word_clips/visual_data \
    --annonation-direc /scratch_hdd/LRS3/word_clips/annotations \
    --label-path labels/LRS3_500WordsSortedList.txt \
    --extract-feats --model-path <CKPT> \
    --mouth-patch-path <MOUTH-NPZ> --mouth-embedding-out-path <OUT-NPZ>
```

The dataset arguments are required because `main.py` builds the loaders before it checks for feature mode. The output
has shape `(1, T, 512)` for an input of `T` frames.

## Limitations

- The model is trained on noisy automatic alignments and sentences from pretrain. Expect lower accuracy than LRW,
  and there is no reference number to compare with.
- No LRS3 official test evaluation, as above.
- Word clips take about 34 GB and sentence ROIs about 135 GB on `/scratch_hdd`, so the disk budget is tight.
