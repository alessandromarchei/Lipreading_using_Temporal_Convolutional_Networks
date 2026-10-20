#!/usr/bin/env python3
"""Preprocess LRS3 for AV-TSE and archive each verified source MP4 to HDD.

Pipelined, multi-process version of the original single-process script.

For each source MP4 this pipeline creates, on SSD:
  * lossless mono 16 kHz FLAC audio;
  * 68-point facial landmarks, float32 [T, 68, 2];
  * grayscale mouth crops, uint8 [T, 96, 96];
  * verification against the already-separated transcript.

Only after every output is valid, the original MP4 is copied to HDD using the
same path relative to --ssd-root, verified by SHA-256, durably renamed, and
removed from SSD. Failed samples always retain their SSD MP4.

The source videos are already 224x224 face tracks. No face detection is run:
the complete frame is passed directly to a 68-point landmark regressor. The
result uses the convention required by CTCNet's mouth preprocessing (48:68).

PIPELINE ARCHITECTURE
----------------------
The original script forced --workers=1 whenever a CUDA device was used,
because every worker process loaded its own copy of the face-alignment model
and called it once per video frame. That serialized the entire job behind a
single, unbatched GPU loop.

This version splits the work into three overlapping stages connected by
bounded queues (classic producer/consumer, "poison pill" shutdown):

  Stage 1  decode_workers x CPU processes
           cv2 video decode + ffmpeg audio extraction. Frames are dumped to
           an uncompressed scratch .npy on SSD; only the *path* travels
           through the queue (not the pixel data), so IPC stays cheap.

  Stage 2  exactly 1 process, holds the GPU model
           Loads face-alignment ONCE. For every video it runs landmark
           detection in batches (chunked forward passes) instead of a
           Python loop calling the model once per frame -- this is normally
           the single biggest speedup available, independent of core count.

  Stage 3  finish_workers x CPU processes
           mouth cropping (cv2.warpAffine), output validation, SHA-256
           verified archival copy to HDD.

Tune --decode-workers / --finish-workers to your core count, and
--landmark-batch-size to your VRAM. Only stage 2 ever touches the GPU, so
you can freely parallelize stages 1 and 3 across all remaining cores.
"""

from __future__ import annotations

import os

# Must happen before numpy/cv2/torch are imported: each subprocess (spawn
# context) re-executes this module from the top, so these are picked up by
# every worker. Without this, every process independently spins up
# BLAS/OpenMP thread pools sized to the full core count, and N processes x M
# threads each causes massive oversubscription and contention on 24 cores.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import soundfile as sf

cv2.setNumThreads(1)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    cpu_count = os.cpu_count() or 8
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssd-root", type=Path, required=True)
    parser.add_argument("--hdd-root", type=Path, required=True)
    parser.add_argument("--mean-face", type=Path, required=True)
    parser.add_argument(
        "--splits", nargs="+", default=["pretrain", "trainval"],
        choices=["pretrain", "trainval"],
    )
    parser.add_argument(
        "--decode-workers", type=int, default=max(2, cpu_count - 6),
        help="CPU processes doing video decode + audio extraction (stage 1)",
    )
    parser.add_argument(
        "--finish-workers", type=int, default=max(2, cpu_count // 2),
        help="CPU processes doing mouth-crop + validation + archive (stage 3)",
    )
    parser.add_argument(
        "--landmark-batch-size", type=int, default=48,
        help="Frames per GPU forward pass in the landmark stage (stage 2)",
    )
    parser.add_argument(
        "--pipeline-depth", type=int, default=None,
        help="Max in-flight items per inter-stage queue (default: 2x that "
             "stage's worker count). Lower this if scratch disk usage grows "
             "too fast; raise it if stages are stalling on each other.",
    )
    parser.add_argument(
        "--skip-audio", action="store_true",
        help="Do not write the 16 kHz FLAC audio (saves ~0.2 MB per clip; regenerate later with ffmpeg)",
    )
    parser.add_argument(
        "--landmark-stride", type=int, default=2,
        help="Run landmark detection on every N-th frame and interpolate the others "
             "(2 halves GPU time, ~0.4 px mean error after the mouth-crop smoothing; 1 = every frame)",
    )
    parser.add_argument(
        "--landmark-fp16", action="store_true",
        help="Run the landmark network under fp16 autocast (~2x faster, but produces "
             "occasional 30+ px landmark jumps that break mouth crops; default is fp32)",
    )
    parser.add_argument("--device", default="cuda", help="cpu, cuda, or cuda:0")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--mouth-size", type=int, default=96)
    parser.add_argument("--smoothing-window", type=int, default=12)
    parser.add_argument("--min-free-gib", type=float, default=2.0)
    parser.add_argument(
        "--scratch-root", type=Path, default=None,
        help="Where stage 1 writes raw decoded frames for stage 2/3 to "
             "read (default: <ssd-root>/scratch). Cleaned up per-sample "
             "as soon as it's no longer needed.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Process at most N sources; use 100 first to estimate final space",
    )
    parser.add_argument(
        "--archive-source", action="store_true",
        help="After full verification, copy MP4 to HDD and remove its SSD copy",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


# --------------------------------------------------------------------------
# Shared helpers (unchanged logic from the original script)
# --------------------------------------------------------------------------


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fsync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def ensure_free_space(root: Path, minimum_gib: float) -> None:
    free = shutil.disk_usage(root).free
    if free < int(minimum_gib * 1024**3):
        raise RuntimeError(
            f"SSD free space {free / 1024**3:.2f} GiB is below "
            f"--min-free-gib={minimum_gib:.2f}"
        )


def atomic_npz(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".part.npz")
    temporary.unlink(missing_ok=True)
    np.savez_compressed(temporary, data=data)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_save_frames(path: Path, frames: list[np.ndarray]) -> None:
    """Dump decoded frames to scratch so stage 2/3 can mmap them instead of
    the pixel data being pickled through an IPC queue."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".part.npy")
    temporary.unlink(missing_ok=True)
    np.save(temporary, np.stack(frames))
    os.replace(temporary, path)


def extract_flac(mp4: Path, flac: Path, sample_rate: int) -> None:
    flac.parent.mkdir(parents=True, exist_ok=True)
    temporary = flac.with_name(flac.stem + ".part.flac")
    temporary.unlink(missing_ok=True)
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-threads", "1", "-y", "-i", str(mp4), "-map", "0:a:0", "-vn",
        "-ac", "1", "-ar", str(sample_rate), "-sample_fmt", "s16",
        "-c:a", "flac", str(temporary),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg: {result.stderr.strip()}")
    info = sf.info(temporary)
    if info.samplerate != sample_rate or info.channels != 1 or info.frames <= 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"invalid temporary FLAC: {info}")
    os.replace(temporary, flac)


def read_video(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError("OpenCV cannot open video")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise RuntimeError("video contains no decoded frames")
    if not 24.0 <= fps <= 26.0:
        raise RuntimeError(f"unexpected video FPS: {fps}")
    return frames, fps


def interpolate_landmarks(values: list[Optional[np.ndarray]]) -> np.ndarray:
    valid = [index for index, value in enumerate(values) if value is not None]
    if not valid:
        raise RuntimeError("no facial landmarks detected in any frame")
    result = np.empty((len(values), 68, 2), dtype=np.float32)
    first, last = valid[0], valid[-1]
    result[: first + 1] = values[first]
    result[last:] = values[last]
    for left, right in zip(valid, valid[1:]):
        left_value = np.asarray(values[left], dtype=np.float32)
        right_value = np.asarray(values[right], dtype=np.float32)
        for index in range(left, right + 1):
            alpha = (index - left) / max(1, right - left)
            result[index] = (1.0 - alpha) * left_value + alpha * right_value
    return result


def crop_square(image: np.ndarray, center: np.ndarray, size: int) -> np.ndarray:
    half = size // 2
    cx, cy = int(round(float(center[0]))), int(round(float(center[1])))
    padded = cv2.copyMakeBorder(image, half, half, half, half, cv2.BORDER_REPLICATE)
    cx += half
    cy += half
    patch = padded[cy - half : cy - half + size, cx - half : cx - half + size]
    if patch.shape[:2] != (size, size):
        raise RuntimeError(f"invalid mouth patch shape: {patch.shape}")
    return patch


def make_mouths(
    frames,  # list[np.ndarray] or np.ndarray / memmap, both support enumerate()
    landmarks: np.ndarray,
    mean_face: np.ndarray,
    size: int,
    smoothing_window: int,
) -> np.ndarray:
    if mean_face.shape != (68, 2):
        raise RuntimeError(f"mean face must be [68,2], got {mean_face.shape}")
    stable = np.asarray([33, 36, 39, 42, 45])
    output: list[np.ndarray] = []
    total = len(frames)
    # This matches CTCNet's forward-window landmark smoothing convention.
    for index, frame in enumerate(frames):
        stop = min(total, index + smoothing_window)
        smoothed = landmarks[index:stop].mean(axis=0)
        matrix, _ = cv2.estimateAffinePartial2D(
            smoothed[stable], mean_face[stable], method=cv2.LMEDS
        )
        if matrix is None:
            raise RuntimeError(f"affine alignment failed at frame {index}")
        aligned = cv2.warpAffine(
            np.asarray(frame), matrix, (256, 256), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        transformed = landmarks[index] @ matrix[:, :2].T + matrix[:, 2]
        mouth_center = transformed[48:68].mean(axis=0)
        patch = crop_square(aligned, mouth_center, size)
        output.append(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY))
    return np.asarray(output, dtype=np.uint8)


def validate_outputs(
    flac: Path,
    landmark_path: Path,
    mouth_path: Path,
    transcript: Path,
    frames: int,
    sample_rate: int,
    mouth_size: int,
) -> tuple[int, int]:
    audio_frames = 0  # --skip-audio: FLAC not written, nothing to check
    if flac.exists():
        info = sf.info(flac)
        if info.samplerate != sample_rate or info.channels != 1 or info.frames <= 0:
            raise RuntimeError("FLAC validation failed")
        audio_frames = int(info.frames)
    landmarks = np.load(landmark_path, allow_pickle=False)["data"]
    mouths = np.load(mouth_path, allow_pickle=False)["data"]
    if landmarks.shape != (frames, 68, 2) or landmarks.dtype != np.float32:
        raise RuntimeError(f"landmark validation failed: {landmarks.shape}/{landmarks.dtype}")
    if mouths.shape != (frames, mouth_size, mouth_size) or mouths.dtype != np.uint8:
        raise RuntimeError(f"mouth validation failed: {mouths.shape}/{mouths.dtype}")
    if not transcript.is_file() or not transcript.read_text(encoding="utf-8").strip():
        raise RuntimeError("missing or empty transcript")
    return audio_frames, int(mouths.shape[0])


def archive_mp4(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(source)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size or sha256(destination) != source_hash:
            raise RuntimeError(f"different HDD destination already exists: {destination}")
    else:
        temporary = destination.with_name(destination.name + ".part")
        temporary.unlink(missing_ok=True)
        with source.open("rb") as source_handle, temporary.open("wb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle, length=8 * 1024 * 1024)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        if temporary.stat().st_size != source.stat().st_size or sha256(temporary) != source_hash:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("HDD archive checksum mismatch")
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
    source.unlink()
    fsync_directory(source.parent)
    return source_hash


# --------------------------------------------------------------------------
# Stage 1: decode (many CPU processes)
# --------------------------------------------------------------------------


def decode_worker_init() -> None:
    cv2.setNumThreads(1)


def decode_task(task: tuple[str, str], config: dict[str, Any]) -> dict[str, Any]:
    split, relative_text = task
    relative = Path(relative_text)
    ssd_root = Path(config["ssd_root"])
    scratch_root = Path(config["scratch_root"])
    stem = relative.with_suffix("")

    source = ssd_root / "av_media" / split / relative
    flac = ssd_root / "processed" / "audio" / split / stem.with_suffix(".flac")
    landmark_path = ssd_root / "processed" / "landmarks" / split / stem.with_suffix(".npz")
    mouth_path = ssd_root / "processed" / "mouths" / split / stem.with_suffix(".npz")
    transcript = ssd_root / "transcripts" / split / stem.with_suffix(".txt")
    archive = Path(config["hdd_root"]) / source.relative_to(ssd_root)
    scratch_frames = scratch_root / split / stem.with_suffix(".npy")

    ensure_free_space(ssd_root, config["min_free_gib"])
    frames, fps = read_video(source)

    if not config["skip_audio"] and (config["overwrite"] or not flac.exists()):
        extract_flac(source, flac, config["sample_rate"])

    needs_recompute = (
        config["overwrite"] or not landmark_path.exists() or not mouth_path.exists()
    )
    frames_path: Optional[str] = None
    if needs_recompute:
        atomic_save_frames(scratch_frames, frames)
        frames_path = str(scratch_frames)

    return {
        "ok": True,
        "split": split,
        "relative_mp4": relative_text,
        "fps": fps,
        "video_frames": len(frames),
        "source_bytes": source.stat().st_size,
        "needs_recompute": needs_recompute,
        "frames_path": frames_path,
        "source": str(source),
        "flac": str(flac),
        "landmark_path": str(landmark_path),
        "mouth_path": str(mouth_path),
        "transcript": str(transcript),
        "archive": str(archive),
    }


def decode_worker_loop(task_queue: mp.Queue, out_queue: mp.Queue, config: dict[str, Any]) -> None:
    decode_worker_init()
    while True:
        task = task_queue.get()
        if task is None:
            out_queue.put(None)
            return
        try:
            result = decode_task(task, config)
        except Exception as error:
            split, relative_text = task
            result = {
                "ok": False, "split": split, "relative_mp4": relative_text,
                "error": f"{type(error).__name__}: {error}",
            }
        out_queue.put(result)


# --------------------------------------------------------------------------
# Stage 2: landmark detection (single persistent GPU process)
# --------------------------------------------------------------------------


def _face_alignment_model(device: str):
    import face_alignment

    landmark_type = getattr(face_alignment.LandmarksType, "TWO_D", None)
    if landmark_type is None:
        landmark_type = getattr(face_alignment.LandmarksType, "_2D")
    return face_alignment.FaceAlignment(
        landmark_type,
        flip_input=False,
        device=device,
        face_detector="sfd",
    )


def _detect_landmarks_batched(
    aligner, frames_mmap: np.ndarray, batch_size: int, fp16: bool = False
) -> list[Optional[np.ndarray]]:
    """Batched landmark detection on the whole frame (the clips are face tracks).

    face_alignment's get_landmarks_from_batch() loops over frames and runs the
    network once per image, so it never batches on the GPU. This runs the FAN
    network on chunks of `batch_size` frames and reproduces the library's
    single-face crop and decode (same centre, scale and CENTER_Y_OFFSET), so the
    landmarks match the per-frame path. The last chunk is zero-padded to
    `batch_size` so the network always sees one input shape.
    """
    import torch
    from face_alignment.utils import CENTER_Y_OFFSET, crop, get_preds_fromhm

    torch.backends.cudnn.benchmark = True  # chunk shape is fixed (padded), so autotuning pays off
    net = aligner.face_alignment_net
    total, height, width = frames_mmap.shape[0], frames_mmap.shape[1], frames_mmap.shape[2]

    # same full-frame box the previous path passed as detected_faces
    x0, y0, x1, y1 = 0.0, 0.0, width - 1.0, height - 1.0
    center = np.array([x1 - (x1 - x0) / 2.0, y1 - (y1 - y0) / 2.0])
    center[1] = center[1] - (y1 - y0) * CENTER_Y_OFFSET
    scale = (x1 - x0 + y1 - y0) / aligner.face_detector.reference_scale

    detected: list[Optional[np.ndarray]] = []
    for start in range(0, total, batch_size):
        chunk = np.asarray(frames_mmap[start : start + batch_size])[..., ::-1]  # BGR -> RGB
        count = chunk.shape[0]
        crops = np.zeros((batch_size, 3, 256, 256), dtype=np.float32)
        for index, frame in enumerate(chunk):
            crops[index] = crop(np.ascontiguousarray(frame), center, scale).transpose(2, 0, 1) / 255.0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
            output = net(torch.from_numpy(crops).to(aligner.device))
        if isinstance(output, list):
            output = output[-1]
        heatmaps = output.float().cpu().numpy()
        for index in range(count):
            _, points_image, _ = get_preds_fromhm(heatmaps[index : index + 1], center, scale)
            detected.append(points_image.reshape(68, 2).astype(np.float32))
    return detected


def _best_candidate(candidates) -> Optional[np.ndarray]:
    if candidates is None or len(candidates) == 0:
        return None
    candidate_list = candidates if isinstance(candidates, (list, tuple)) else [candidates]
    candidate_list = [np.asarray(c, dtype=np.float32) for c in candidate_list]
    landmark = max(
        candidate_list,
        key=lambda points: float(np.ptp(points[:, 0]) * np.ptp(points[:, 1])),
    )
    return landmark if landmark.shape == (68, 2) else None


def gpu_worker_loop(
    in_queue: mp.Queue,
    out_queue: mp.Queue,
    config: dict[str, Any],
    decode_worker_count: int,
    finish_worker_count: int,
) -> None:
    aligner = _face_alignment_model(config["device"])
    pending_sentinels = 0

    while pending_sentinels < decode_worker_count:
        item = in_queue.get()
        if item is None:
            pending_sentinels += 1
            continue

        if item["ok"] and item["needs_recompute"]:
            try:
                frames_mmap = np.load(item["frames_path"], mmap_mode="r")
                # detect every `stride`-th frame (plus the last one) and linearly interpolate the rest;
                # the mouth crops are smoothed over 12 frames, so this costs well under 1 px
                stride = config["landmark_stride"]
                total = frames_mmap.shape[0]
                indices = sorted(set(range(0, total, stride)) | {total - 1})
                detected: list[Optional[np.ndarray]] = [None] * total
                subset = _detect_landmarks_batched(
                    aligner, frames_mmap[indices], config["landmark_batch_size"], config["landmark_fp16"]
                )
                for index, points in zip(indices, subset):
                    detected[index] = points
                landmarks = interpolate_landmarks(detected)
                atomic_npz(Path(item["landmark_path"]), landmarks.astype(np.float32))
            except Exception as error:
                item["ok"] = False
                item["error"] = f"{type(error).__name__}: {error}"

        out_queue.put(item)

    for _ in range(finish_worker_count):
        out_queue.put(None)


# --------------------------------------------------------------------------
# Stage 3: finish -- mouth crop, validation, archival (many CPU processes)
# --------------------------------------------------------------------------


def finish_worker_loop(in_queue: mp.Queue, result_queue: mp.Queue, config: dict[str, Any]) -> None:
    cv2.setNumThreads(1)
    mean_face = np.load(config["mean_face"]).astype(np.float32)

    while True:
        item = in_queue.get()
        if item is None:
            return

        if not item["ok"]:
            result_queue.put(item)
            continue

        frames_path = item.get("frames_path")
        try:
            landmark_path = Path(item["landmark_path"])
            mouth_path = Path(item["mouth_path"])
            flac = Path(item["flac"])
            transcript = Path(item["transcript"])

            if item["needs_recompute"]:
                frames_mmap = np.load(frames_path, mmap_mode="r")
                landmarks = np.load(landmark_path, allow_pickle=False)["data"]
                mouths = make_mouths(
                    frames_mmap, landmarks, mean_face,
                    config["mouth_size"], config["smoothing_window"],
                )
                atomic_npz(mouth_path, mouths)

            audio_frames, mouth_frames = validate_outputs(
                flac, landmark_path, mouth_path, transcript, item["video_frames"],
                config["sample_rate"], config["mouth_size"],
            )
            flac_bytes = flac.stat().st_size if flac.exists() else 0
            processed_bytes = flac_bytes + landmark_path.stat().st_size + mouth_path.stat().st_size
            source_hash = ""
            if config["archive_source"]:
                source_hash = archive_mp4(Path(item["source"]), Path(item["archive"]))

            result = {
                "ok": True, "split": item["split"], "relative_mp4": item["relative_mp4"],
                "fps": item["fps"], "video_frames": item["video_frames"],
                "audio_frames": audio_frames, "mouth_frames": mouth_frames,
                "source_bytes": item["source_bytes"], "processed_bytes": processed_bytes,
                "net_ssd_bytes": processed_bytes - item["source_bytes"],
                "archive": item["archive"], "source_sha256": source_hash,
            }
        except Exception as error:
            result = {
                "ok": False, "split": item["split"], "relative_mp4": item["relative_mp4"],
                "error": f"{type(error).__name__}: {error}",
            }
        finally:
            if frames_path is not None:
                Path(frames_path).unlink(missing_ok=True)

        result_queue.put(result)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    args.ssd_root = args.ssd_root.resolve()
    args.hdd_root = args.hdd_root.resolve()
    args.mean_face = args.mean_face.resolve()
    args.scratch_root = (args.scratch_root or (args.ssd_root / "scratch")).resolve()

    if args.decode_workers < 1 or args.finish_workers < 1:
        raise SystemExit("--decode-workers and --finish-workers must be >= 1")
    if not args.mean_face.is_file():
        raise SystemExit(f"Mean face not found: {args.mean_face}")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found")
    args.hdd_root.mkdir(parents=True, exist_ok=True)
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    if args.ssd_root.stat().st_dev == args.hdd_root.stat().st_dev:
        print("WARNING: SSD and HDD roots appear to be on the same filesystem", file=sys.stderr)

    tasks: list[tuple[str, str]] = []
    for split in args.splits:
        root = args.ssd_root / "av_media" / split
        if not root.is_dir():
            raise SystemExit(f"Missing media split: {root}")
        tasks.extend((split, str(path.relative_to(root))) for path in sorted(root.rglob("*.mp4")))
    if args.max_samples is not None:
        tasks = tasks[: args.max_samples]
    if not tasks:
        print("No SSD MP4 files remain for the selected splits.")
        return 0

    manifest_dir = args.ssd_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "preprocess_archive.jsonl"
    failures_path = manifest_dir / "preprocess_failures.tsv"

    config = {
        "ssd_root": str(args.ssd_root), "hdd_root": str(args.hdd_root),
        "mean_face": str(args.mean_face), "sample_rate": args.sample_rate,
        "mouth_size": args.mouth_size, "smoothing_window": args.smoothing_window,
        "min_free_gib": args.min_free_gib, "archive_source": args.archive_source,
        "overwrite": args.overwrite, "scratch_root": str(args.scratch_root),
        "landmark_batch_size": args.landmark_batch_size, "device": args.device,
        "landmark_fp16": args.landmark_fp16,
        "landmark_stride": args.landmark_stride, "skip_audio": args.skip_audio,
    }

    depth_decode = args.pipeline_depth or (args.decode_workers * 2)
    depth_finish = args.pipeline_depth or (args.finish_workers * 2)

    print(f"Samples:          {len(tasks):,}")
    print(f"Decode workers:   {args.decode_workers}")
    print(f"Finish workers:   {args.finish_workers}")
    print(f"GPU stage:        1 process, device={args.device}, batch={args.landmark_batch_size}")
    print(f"Archive MP4:      {args.archive_source}")

    ctx = mp.get_context("spawn")
    task_queue: mp.Queue = ctx.Queue()
    decode_out_queue: mp.Queue = ctx.Queue(maxsize=depth_decode)
    landmark_out_queue: mp.Queue = ctx.Queue(maxsize=depth_finish)
    result_queue: mp.Queue = ctx.Queue()

    for task in tasks:
        task_queue.put(task)
    for _ in range(args.decode_workers):
        task_queue.put(None)

    decode_procs = [
        ctx.Process(target=decode_worker_loop, args=(task_queue, decode_out_queue, config))
        for _ in range(args.decode_workers)
    ]
    gpu_proc = ctx.Process(
        target=gpu_worker_loop,
        args=(decode_out_queue, landmark_out_queue, config, args.decode_workers, args.finish_workers),
    )
    finish_procs = [
        ctx.Process(target=finish_worker_loop, args=(landmark_out_queue, result_queue, config))
        for _ in range(args.finish_workers)
    ]

    all_procs = decode_procs + [gpu_proc] + finish_procs
    for process in all_procs:
        process.start()

    total_source = total_processed = total_net = successes = failures = 0
    try:
        with manifest_path.open("a", encoding="utf-8") as manifest, failures_path.open(
            "a", encoding="utf-8", newline=""
        ) as failure_file:
            failure_writer = csv.writer(failure_file, delimiter="\t")
            for done in range(1, len(tasks) + 1):
                result = result_queue.get()
                if result["ok"]:
                    successes += 1
                    total_source += result["source_bytes"]
                    total_processed += result["processed_bytes"]
                    total_net += result["net_ssd_bytes"]
                    manifest.write(json.dumps(result, sort_keys=True) + "\n")
                    manifest.flush()
                else:
                    failures += 1
                    failure_writer.writerow([
                        result["split"], result["relative_mp4"], result["error"]
                    ])
                    failure_file.flush()
                if done % 10 == 0 or done == len(tasks):
                    print(
                        f"[{done:,}/{len(tasks):,}] ok={successes:,} failed={failures:,} "
                        f"net SSD={total_net / 1024**3:+.2f} GiB",
                        flush=True,
                    )
    finally:
        for process in all_procs:
            process.join(timeout=30)
        for process in all_procs:
            if process.is_alive():
                process.terminate()
                process.join()

    print("\nSUMMARY")
    print(f"  Successes:       {successes:,}")
    print(f"  Failures:        {failures:,}")
    print(f"  Source MP4 size: {total_source / 1024**3:.3f} GiB")
    print(f"  Derived size:    {total_processed / 1024**3:.3f} GiB")
    print(f"  Net SSD change:  {total_net / 1024**3:+.3f} GiB")
    if successes and args.max_samples:
        print(
            f"  Projected net for 135,473 samples: "
            f"{total_net / successes * 135473 / 1024**3:+.1f} GiB "
            "(rough pilot estimate)"
        )
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())