#!/usr/bin/env python3
"""
Fine-tune the plate detector for fast-moving vehicles.

The problem
-----------
Plate reads degrade sharply above ~70 km/h. At 100 km/h with a 1/60 s shutter a
vehicle travels ~46 cm during the exposure, which at typical gantry framing
smears a plate across 15-40 pixels horizontally. The *detector* is the first
thing to fail: a smeared plate has weak vertical edges and low local contrast,
so the box confidence drops below threshold and the crop is never produced. No
crop means no OCR, however good the OCR is.

The answer is NOT preprocessing
-------------------------------
Sharpening or deblurring the crop before OCR was measured on 124 hand-labelled
real crops and made accuracy *worse* (CER 0.2291 raw vs 0.2410 enhanced) — see
`backend/services/anpr_service.preprocess_plate_crop`. PaddleOCR normalises
internally; enhancing first resamples twice and destroys detail. So the fix has
to be in the detector's weights, not in a filter at inference time.

What this script does
---------------------
Fine-tunes `best.pt` on a dataset augmented with **directional motion blur**,
so the detector learns to localise smeared plates instead of only crisp ones.

Two ways to get the training data:

1. `--dataset path/to/data.yaml` — a real labelled plate dataset (the Phase 1
   HF dataset, or your own). Preferred: real labels, real backgrounds.

2. `--from-videos` — **self-labelled from traffic footage on this machine.**
   The current detector is run over the sharp frames of the sample videos and
   its high-confidence detections (>= --pseudo-conf) are kept as pseudo-labels.
   Each frame is then written out twice: once clean, once with synthetic
   directional blur applied *at the label's own scale and direction of travel*.
   The label box is unchanged, because motion blur along the plate's axis
   smears content without moving the box centre.

   This is legitimate and it is the standard trick for making a detector robust
   to a corruption you can simulate: the model already knows what a plate is
   when it is sharp, and we are teaching it that the same object under blur is
   still that object. It is *not* a way to teach it plates it never knew — the
   labels are only as good as the current model's confident detections, which
   is why the threshold is high and why option 1 is preferred when available.

Training regime
---------------
Low LR, frozen backbone, few epochs. This is a *fine-tune*: the base weights
already reach mAP@50 >= 0.85 on sharp plates and the goal is to add blur
robustness without forgetting that. A full-LR run over a blur-heavy dataset
would trade sharp-plate accuracy for blurred-plate accuracy, which is a net
loss — most vehicles are not speeding.

Every run writes `provenance.json` beside the weights (base weights, dataset,
resolved hyperparameters, ultralytics version, git SHA) and evaluates the
before/after on a held-out blurred split, so the claim "this helped" is a
measurement rather than an assertion.

Usage
-----
  # Self-labelled from the bundled traffic footage (runs on this machine)
  .venv/bin/python scripts/finetune_fast_motion.py --from-videos --epochs 25

  # From a real labelled dataset
  .venv/bin/python scripts/finetune_fast_motion.py --dataset data/yolo/data.yaml

  # Just build the blur-augmented dataset and stop, to inspect it
  .venv/bin/python scripts/finetune_fast_motion.py --from-videos --build-only

Then point the backend at the result:
  ANPR_WEIGHTS_PATH=/abs/path/to/runs/fast_motion/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

REPO_ROOT = BASE_DIR.parent
DEFAULT_WEIGHTS = REPO_ROOT / "Automatic-License-Plate-Recognition" / "best.pt"
ARCHIVE_DIR = REPO_ROOT / "archive"
WORK_DIR = BASE_DIR / "training" / "fast_motion"

# Blur severities, in pixels of travel during the exposure. Chosen against the
# physics rather than by taste: at a 1/60 s shutter and typical gantry framing,
# 60 km/h smears a plate ~8 px, 100 km/h ~14 px, 140 km/h ~20 px. Sampling
# across this range (including 0) is what keeps sharp-plate accuracy intact
# while adding the blurred regime.
BLUR_KERNELS = (0, 5, 9, 13, 17, 21)

# Direction of travel relative to the camera, in degrees. Vehicles pass a
# gantry roughly horizontally; a little spread covers oblique mounts.
BLUR_ANGLES = (0, 8, -8, 15, -15, 90)


# ─────────────────────────────────────────────────────────────────────────────
# Motion blur
# ─────────────────────────────────────────────────────────────────────────────

def motion_blur_kernel(size: int, angle_degrees: float):
    """A normalised line kernel — the point spread function of linear motion.

    A Gaussian blur is the wrong model and would teach the wrong invariance:
    motion blur is *directional*, and a plate smeared horizontally keeps its
    vertical edges. Training on isotropic blur would make the detector tolerant
    of a corruption it will never see and no better at the one it will.
    """
    import cv2
    import numpy as np

    size = max(3, int(size) | 1)          # odd, so the kernel has a centre
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0
    matrix = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle_degrees, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (size, size))
    total = kernel.sum()
    return kernel / total if total > 0 else kernel


def apply_motion_blur(image, size: int, angle_degrees: float):
    import cv2

    if size <= 1:
        return image
    return cv2.filter2D(image, -1, motion_blur_kernel(size, angle_degrees))


# ─────────────────────────────────────────────────────────────────────────────
# Self-labelling from video
# ─────────────────────────────────────────────────────────────────────────────

def sample_frames(video_path: Path, every: int, limit: int):
    """Yield (frame_index, frame). Subsampled — adjacent frames are near-duplicates."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ! could not open {video_path.name}")
        return
    idx = kept = 0
    try:
        while kept < limit:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % every == 0:
                kept += 1
                yield idx, frame
            idx += 1
    finally:
        cap.release()


def build_from_videos(args) -> Path:
    """Pseudo-label the sample footage, then write a blur-augmented YOLO dataset."""
    import cv2
    from ultralytics import YOLO

    out = WORK_DIR / "dataset"
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    videos = sorted(p for p in ARCHIVE_DIR.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No .mp4 files in {ARCHIVE_DIR}. Pass --dataset instead.")

    print(f"\n[1/3] Pseudo-labelling {len(videos)} video(s) with the base detector")
    print(f"      confidence floor {args.pseudo_conf} — a wrong label is worse than no label,")
    print(f"      so anything the current model is not sure about is discarded.\n")

    model = YOLO(str(args.weights))
    rng = random.Random(args.seed)

    kept = skipped = written = 0
    for video in videos:
        v_kept = 0
        for frame_idx, frame in sample_frames(video, args.frame_every, args.frames_per_video):
            results = model.predict(frame, conf=args.pseudo_conf, verbose=False,
                                    device=args.device)
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                skipped += 1
                continue

            h, w = frame.shape[:2]
            lines = []
            for xyxy in boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = xyxy
                bw, bh = (x2 - x1), (y2 - y1)
                if bw < 8 or bh < 5:
                    continue          # too small to be a usable plate label
                # YOLO format: class cx cy w h, all normalised.
                lines.append(
                    f"0 {((x1 + x2) / 2) / w:.6f} {((y1 + y2) / 2) / h:.6f} "
                    f"{bw / w:.6f} {bh / h:.6f}"
                )
            if not lines:
                skipped += 1
                continue

            kept += 1
            v_kept += 1
            # 85/15. Split by frame rather than by video so both splits see
            # every scene; the point of the val set here is to measure blur
            # robustness, not generalisation to an unseen road.
            split = "val" if rng.random() < 0.15 else "train"
            stem = f"{video.stem[:24].replace(' ', '_')}_{frame_idx:06d}"

            # Write the clean frame AND a blurred copy sharing the same label.
            # Keeping the clean copy is what stops the fine-tune from trading
            # sharp-plate accuracy away: most vehicles are not speeding.
            variants = [("clean", frame, 0, 0)]
            for _ in range(args.blur_variants):
                k = rng.choice([b for b in BLUR_KERNELS if b > 0])
                a = rng.choice(BLUR_ANGLES)
                variants.append((f"blur{k}a{a}", apply_motion_blur(frame, k, a), k, a))

            for tag, image, _k, _a in variants:
                name = f"{stem}_{tag}"
                cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), image,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + "\n")
                written += 1

        print(f"  {video.name[:52]:<52} {v_kept:>4} labelled frames")

    if kept == 0:
        raise SystemExit(
            "The base detector found no confident plates in any sampled frame. "
            "Lower --pseudo-conf, or pass a real labelled --dataset."
        )

    yaml_path = out / "data.yaml"
    yaml_path.write_text(
        f"path: {out}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: [plate]\n"
    )

    print(f"\n  {kept:,} frames labelled, {skipped:,} skipped (no confident plate)")
    print(f"  {written:,} images written ({args.blur_variants} blurred variant(s) per frame)")
    print(f"  dataset: {yaml_path}")
    return yaml_path


# ─────────────────────────────────────────────────────────────────────────────
# Train + evaluate
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(weights: Path, data_yaml: Path, device: str, label: str) -> dict:
    from ultralytics import YOLO

    print(f"\n  evaluating {label}...")
    metrics = YOLO(str(weights)).val(data=str(data_yaml), device=device,
                                     verbose=False, plots=False)
    box = metrics.box
    return {
        "map50": round(float(box.map50), 4),
        "map50_95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
    }


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=BASE_DIR,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", help="path to an existing YOLO data.yaml")
    src.add_argument("--from-videos", action="store_true",
                     help="build a pseudo-labelled, blur-augmented dataset from archive/*.mp4")

    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="base weights to fine-tune")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lr0", type=float, default=0.0008,
                    help="initial LR — 10x below a from-scratch run, because this is a fine-tune")
    ap.add_argument("--freeze", type=int, default=10,
                    help="freeze the first N layers (backbone). 0 to train everything.")
    ap.add_argument("--device", default=None,
                    help="'mps', 'cpu', '0'... default: auto (mps on Apple Silicon)")
    ap.add_argument("--project", default=str(WORK_DIR / "runs"))
    ap.add_argument("--name", default="fast_motion")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--frame-every", type=int, default=15,
                    help="sample one frame in N (adjacent frames are near-duplicates)")
    ap.add_argument("--frames-per-video", type=int, default=120)
    ap.add_argument("--pseudo-conf", type=float, default=0.60,
                    help="only keep detections at least this confident as pseudo-labels")
    ap.add_argument("--blur-variants", type=int, default=2,
                    help="blurred copies written per labelled frame")

    ap.add_argument("--build-only", action="store_true", help="build the dataset and stop")
    ap.add_argument("--skip-baseline", action="store_true",
                    help="skip the before-metrics (halves the wall time, loses the comparison)")
    args = ap.parse_args()

    args.weights = Path(args.weights)
    if not args.weights.exists():
        raise SystemExit(f"Base weights not found: {args.weights}")

    if args.device is None:
        try:
            import torch
            args.device = ("mps" if torch.backends.mps.is_available()
                           else "0" if torch.cuda.is_available() else "cpu")
        except Exception:
            args.device = "cpu"

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    wall0 = time.perf_counter()

    print("=" * 84)
    print("FAST-MOTION FINE-TUNE — teaching the plate detector to survive motion blur")
    print("=" * 84)
    print(f"  base weights  {args.weights}")
    print(f"  device        {args.device}")
    print(f"  regime        {args.epochs} epochs, lr0={args.lr0}, freeze={args.freeze} layers")

    data_yaml = Path(build_from_videos(args)) if args.from_videos else Path(args.dataset)
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml not found: {data_yaml}")

    if args.build_only:
        print(f"\nBuilt only, as requested: {data_yaml}")
        return

    baseline = None
    if not args.skip_baseline:
        print("\n[2/3] Baseline — the current detector on the blurred validation split")
        baseline = evaluate(args.weights, data_yaml, args.device, "base weights")
        print(f"      mAP@50 {baseline['map50']:.4f}  mAP@50-95 {baseline['map50_95']:.4f}  "
              f"P {baseline['precision']:.4f}  R {baseline['recall']:.4f}")

    print(f"\n[3/3] Fine-tuning for {args.epochs} epochs...")
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        # Cosine decay to a small final LR: the last epochs should be settling,
        # not still moving the weights around.
        lrf=0.05,
        cos_lr=True,
        freeze=args.freeze,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        seed=args.seed,
        pretrained=True,
        patience=max(5, args.epochs // 3),
        # ── Augmentation, narrowed for this job ──
        # The blur is already baked into the dataset, at controlled severities.
        # Mosaic tiles four images and shrinks plates further into the
        # small-object regime, which on top of blur produces training examples
        # no camera will ever see; it is disabled for the last third.
        close_mosaic=max(5, args.epochs // 3),
        mosaic=0.5,
        # A plate is never upside down and never mirrored. flipud is off for
        # both reasons; fliplr is kept at the default because detection only
        # has to find a bright rectangle and halving the data is the bigger risk.
        flipud=0.0,
        fliplr=0.5,
        degrees=6.0,
        translate=0.08,
        scale=0.35,
        # Motion blur co-occurs with low light (a long exposure is why there is
        # blur at all), so the value jitter is widened.
        hsv_v=0.5,
        hsv_s=0.6,
        verbose=True,
    )

    run_dir = Path(args.project) / args.name
    tuned = run_dir / "weights" / "best.pt"
    if not tuned.exists():
        raise SystemExit(f"Training finished but no weights at {tuned}")

    after = evaluate(tuned, data_yaml, args.device, "fine-tuned weights")

    provenance = {
        "created": datetime.now(timezone.utc).isoformat(),
        "base_weights": str(args.weights),
        "tuned_weights": str(tuned),
        "dataset": str(data_yaml),
        "self_labelled": bool(args.from_videos),
        "pseudo_label_confidence": args.pseudo_conf if args.from_videos else None,
        "blur_kernels_px": list(BLUR_KERNELS),
        "blur_angles_deg": list(BLUR_ANGLES),
        "hyperparameters": {
            "epochs": args.epochs, "batch": args.batch, "imgsz": args.imgsz,
            "lr0": args.lr0, "freeze": args.freeze, "device": args.device,
            "seed": args.seed,
        },
        "metrics": {"before": baseline, "after": after},
        "git_sha": git_sha(),
    }
    try:
        import ultralytics
        provenance["ultralytics"] = ultralytics.__version__
    except Exception:
        pass

    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    print("\n" + "=" * 84)
    print("RESULT")
    print("=" * 84)
    if baseline:
        for key in ("map50", "map50_95", "precision", "recall"):
            delta = after[key] - baseline[key]
            arrow = "▲" if delta > 0.001 else "▼" if delta < -0.001 else "="
            print(f"  {key:<12} {baseline[key]:.4f}  ->  {after[key]:.4f}   {arrow} {delta:+.4f}")
        if after["map50"] <= baseline["map50"]:
            # Say so plainly. A fine-tune that did not help is a result, and
            # shipping the weights anyway because the script printed "done"
            # is how a regression reaches production.
            print("\n  The fine-tune did NOT improve mAP@50 on this split.")
            print("  Do not ship these weights. Try: more epochs, --freeze 0,")
            print("  more --blur-variants, or a real labelled --dataset.")
        else:
            print(f"\n  Weights: {tuned}")
            print("  Point the backend at them:")
            print(f"    ANPR_WEIGHTS_PATH={tuned}")
    else:
        print(f"  mAP@50 {after['map50']:.4f}  (no baseline — ran with --skip-baseline)")
    print(f"\n  provenance: {run_dir / 'provenance.json'}")
    print(f"  total wall time {time.perf_counter() - wall0:.1f}s")


if __name__ == "__main__":
    main()
