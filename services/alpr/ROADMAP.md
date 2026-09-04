# ALPR — Automatic License Plate Recognition

End-to-end pipeline: video in → plates detected, read, validated, deduplicated → **one row per vehicle in an Excel log**.

---

## Locked decisions

| Decision | Choice | Why |
|---|---|---|
| Detector | YOLO (Ultralytics), **trained from scratch by us** | Portfolio differentiator; a pretrained plate model would make Phases 1–3 a download |
| OCR | PaddleOCR (recognition head only) | Best accuracy/effort on tilted, low-res crops; trivial to install on Colab Linux |
| Plate formats | **India + Germany**, pluggable validators | Format-aware correction is the single biggest accuracy win after detection |
| Compute | **Google Colab T4** for everything GPU | No local CPU training |
| Dataset | **Built from scratch** (Phase 1) | No reuse of prior Drive data |
| Code home | `src/alpr/` installable package, notebooks are thin drivers | Logic stays testable and diffable; notebooks stay reviewable |
| Artifacts | Weights + dataset → Hugging Face Hub | Colab sessions are ephemeral; Drive mounts rot |

### Where each thing runs

Colab is the GPU, not the whole system. The split:

- **Colab T4** — dataset prep, detector training, OCR benchmarking, batch inference over video files, evaluation. Phases 1–8.
- **Local (your M4)** — git, editing, tests, and *only* the live camera modes.

**One honest constraint up front:** live webcam and RTSP **cannot run on Colab**. Colab has no path to your Mac's camera (only a slow JS screenshot bridge) and cannot reach an RTSP camera on your LAN — it's a VM in a Google datacenter. Those two inputs are inherently local, so they land in Phase 9. That does *not* mean CPU: YOLO-nano on the M4 runs on the Metal/MPS GPU at real-time framerates. Nothing heavy is ever asked of your CPU.

---

## Architecture

```
video/camera ──▶ FrameSource ──▶ Detector ──▶ Tracker ──▶ per-track crop buffer
                 (file/cam/rtsp)   (YOLO)     (ByteTrack)          │
                                                                   ▼
   Excel log ◀── Deduplicator ◀── Validator ◀── Voter ◀────────── OCR
   (openpyxl)    (cooldown)       (IN / DE)   (multi-frame)   (PaddleOCR)
```

**The two design choices that make or break this**, both often skipped in tutorial ALPR builds:

1. **Track-level aggregation, not frame-level.** A naive pipeline OCRs every frame and logs `MH12AB1234` thirty times per second. We assign each vehicle a track ID, buffer its plate crops across frames, and emit **one voted result per track**. This is also free accuracy — a plate misread at frame 5 is outvoted by frames 6–40.
2. **Format-aware correction.** OCR confuses `0/O`, `1/I`, `8/B`, `5/S`. Blind correction hurts. Position-aware correction against a known plate grammar (`MH12AB1234` → positions 0–1 must be letters, 2–3 digits) fixes exactly the errors that are fixable and leaves the rest alone.

---

## Phases

Each phase is one branch → one PR → merged to `main`. Exit criteria are things that can be *checked*, not felt.

### Phase 0 — Foundation
Repo scaffolding, installable package, CI, and the Colab bridge.

- `pyproject.toml` (hatchling, `src/alpr`), ruff + pytest config
- `.github/workflows/ci.yml` — ruff + pytest on push
- `notebooks/00_colab_bootstrap.ipynb` — clones the repo, `pip install -e .`, asserts `nvidia-smi` shows a T4
- Push to GitHub as its own repo (matching the `llm-finetuning-medqa` pattern)

**Exit:** CI green; the bootstrap notebook runs top-to-bottom on a fresh T4 runtime and prints the GPU name.

### Phase 1 — Dataset, from scratch
The phase that actually determines final accuracy. Budget real time here.

- Decide the image sources (open-license imagery, own footage, or both — decided at phase start)
- Annotate plate bounding boxes in YOLO format
- Deterministic, **grouped** train/val/test split — frames from one video must never straddle the split, or val accuracy is a lie
- `alpr.data` loaders + a stats report (count, resolution, plates/image, region balance IN vs DE)
- Publish to HF Hub as a versioned dataset

**Exit:** ≥1 000 annotated plates, IN/DE both represented, split verified leak-free by a test, dataset loads in Colab in one line.

> **Note on German plates:** a plate is personal data under GDPR. Own-footage capture in public is fine for private research, but the dataset must not be published with identifiable context (faces, locations). Phase 1 includes a blur-and-strip step before any upload.

### Phase 2 — Train the detector (T4)
- `alpr.train` — YOLO training entrypoint, config-driven
- `notebooks/02_train_detector.ipynb` — thin driver on T4
- Augmentation tuned for plates: rotation, perspective, motion blur, brightness — *not* vertical flip
- Checkpoints → HF Hub each run

**Exit:** trained weights on the Hub; training curve logged; run reproducible from a committed config.

### Phase 3 — Detection evaluation
- `alpr.detect` — clean inference API over the trained weights
- mAP@50 / mAP@50-95 on the held-out test split, plus a small-object slice (plates < 32 px)
- Failure gallery: worst 20 predictions rendered to disk

**Exit:** mAP@50 ≥ 0.85 on test, with the failure gallery reviewed. Below that, iterate Phase 1/2 rather than pushing forward — OCR cannot read a plate the detector missed.

### Phase 4 — OCR stage
- `alpr.ocr` — PaddleOCR wrapper over plate crops
- Crop preprocessing: perspective de-skew, upscale, contrast normalize
- Benchmark on cropped ground-truth plates, ablating each preprocessing step so we keep only what earns its place

**Exit:** character-error-rate measured on the test split; the preprocessing ablation table is in the repo.

### Phase 5 — Country-aware validation
- `alpr.plates` — `PlateFormat` interface, `IndiaFormat` + `GermanyFormat`
- Grammar-constrained correction (position-aware confusable mapping)
- India: state-code table; Germany: district-prefix table, EU-band handling
- Confidence score combining OCR confidence and grammar fit

**Exit:** unit tests covering valid, invalid, and confusable-corrupted plates for both countries; measured CER improvement over raw Phase 4 output.

### Phase 6 — Tracking and multi-frame voting
- `alpr.track` — ByteTrack assignment of stable IDs
- Per-track crop buffer; character-level weighted vote across frames
- Emit once per track, at track end or after N confident reads

**Exit:** on a test clip with a known vehicle count, emitted-event count equals vehicle count (no duplicates, no misses).

### Phase 7 — Excel logging and the end-to-end pipeline
- `alpr.excel` — openpyxl writer, append-safe, atomic
- Columns: timestamp, plate, country, confidence, track ID, source, frame #, crop path
- Handles the two real failure modes: file locked by an open Excel window, and interrupted runs (append without rewriting the workbook)
- `alpr.pipeline` + CLI: `python -m alpr run --source video.mp4 --out log.xlsx`

**Exit:** one command turns an mp4 into a populated `.xlsx`; interrupting mid-run and resuming loses no rows.

### Phase 8 — Evaluation and reporting
- End-to-end metrics: plate-level precision/recall/F1 against a ground-truth log
- Annotated output video (boxes, IDs, read text) for the README
- README with results table, sample output, architecture diagram

**Exit:** reproducible end-to-end numbers on a held-out clip, in the README.

### Phase 9 — Live modes (local, MPS)
The only phase that runs off Colab.

- `FrameSource` implementations for webcam and RTSP, with reconnect + frame-drop handling
- Export weights to a format that runs fast on Metal
- Live overlay window; measured FPS

**Exit:** ≥15 FPS sustained on the M4 from webcam, logging live to Excel.

---

## Workflow

Git and Antigravity IDE locally, Colab for GPU:

1. Branch locally, write code in `src/alpr/`, run tests
2. Push branch → CI runs
3. In Colab: `!git clone`, checkout branch, `pip install -e .`, run the notebook driver on T4
4. Artifacts (weights, metrics) → HF Hub; results committed back
5. PR → review → merge to `main`

Notebooks stay thin on purpose. Anything with logic worth testing belongs in `src/alpr/`, so Colab never becomes the place where the real code lives.
