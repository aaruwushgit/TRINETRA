# ALPR — Automatic License Plate Recognition

Detects license plates in video, reads them, validates them against Indian and German plate
grammars, and logs **one deduplicated row per vehicle** to an Excel workbook.

[![CI](https://github.com/fayazhussain2821/Automatic-License-Plate-Recognition/actions/workflows/ci.yml/badge.svg)](https://github.com/fayazhussain2821/Automatic-License-Plate-Recognition/actions/workflows/ci.yml)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20model-alpr--plate--detector-yellow)](https://huggingface.co/Babblu2821/alpr-plate-detector)

---

## Results

**Plate detector** — YOLOv8s, 100 epochs on a Colab T4 (1.25 h), evaluated on a held-out test
split of 465 images the model never saw.

| Metric | Test split |
|---|---|
| mAP@50 | **0.9921** |
| mAP@50-95 | 0.8377 |
| Precision | 0.9816 |
| Recall | **0.9917** |
| Inference | 4.7 ms/image (T4) |

**Recall matters more than precision here.** A plate the detector misses can never be read by
OCR — that error is unrecoverable. A false positive produces a crop, OCR emits noise, and the
plate grammar rejects it. The error profile is the right way round: **1 missed plate against 23
false positives** across the whole test split.

The trained detector is published at **[Babblu2821/alpr-plate-detector](https://huggingface.co/Babblu2821/alpr-plate-detector)**:

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

model = YOLO(hf_hub_download("Babblu2821/alpr-plate-detector", "best.pt"))
```

![Training curves](results/results.png)

Full run artifacts, including the exact Ultralytics arguments, are in [`results/`](results/).

### The test set was audited for leakage

A perceptual-hash audit (`alpr.dupes`) found **5.8% of test images had a near-duplicate in
train** — consecutive video frames that Roboflow names `dayride_type1_001-mp4-t-1062`, a pattern
the grouped-splitting logic did not recognise as frames of one clip.

Re-scoring on only the **438 uncontaminated** test images:

| | Full test (465) | Uncontaminated (438) |
|---|---|---|
| Recall | 0.9979 | **0.9978** |
| Precision | 0.9545 | **0.9556** |
| F1 | 0.9757 | **0.9762** |

**The leak was not carrying the score.** Removing every memorized image moved recall by 0.0001,
and precision improved. The grouping was fixed so future splits cluster duplicates.

*(mAP was not recomputed on the subset — the audit compares precision/recall at a fixed
confidence threshold, not the PR-curve integral.)*

### Detection by plate size

| Ground-truth plate width | n | Precision | Recall |
|---|---|---|---|
| tiny (<32 px) | 8 | 1.0000 | 1.0000 |
| small (32–64 px) | 64 | 0.9000 | 0.9844 |
| medium (64–128 px) | 191 | 0.9598 | 1.0000 |
| large (≥128 px) | 190 | 0.9845 | 1.0000 |

Small plates were expected to cap end-to-end accuracy. They do not — every plate under 32 px was
found. The single miss sits in the 32–64 px band.

---

## Reading plates: measured, not asserted

Neither source dataset ships plate *text*, so 124 test-split crops were hand-labelled
(`alpr label`, sampled across size buckets so the number is not flattered by easy plates).

### The grammars work — where they apply

| | CER | exact match |
|---|---|---|
| Raw OCR | 0.2291 | 30.6% |
| **+ plate grammar** | **0.1693** | **43.5%** |

Grammar-constrained correction cuts CER by **26%** and lifts exact-match accuracy by **13
points**. The grammar recognised 69 of 124 reads and altered 35 of them.

Splitting by region shows *why*, and it is the more honest number:

| | CER raw | CER + grammar | exact raw | exact + grammar |
|---|---|---|---|---|
| **Indian plates** (65) | 0.2067 | **0.1138** | 36.9% | **61.5%** |
| **European plates** (59) | 0.2658 | 0.2605 | 23.7% | 23.7% |

**Indian accuracy rises from 37% to 62%. European accuracy does not move at all.**

That is not a defect in the method — it is the method working exactly as designed and being
limited by scope. The grammars model India and Germany. The European dataset is *pan*-European:
Polish, Norwegian and other plates whose formats nothing here describes. A grammar cannot correct
toward a rule it does not have.

The lesson generalises: grammar-constrained correction is worth roughly **+25 points of accuracy
on plates whose format you have modelled, and nothing at all on plates you have not**.

### Preprocessing was removed, because it made things worse

The obvious idea — upscale small crops, normalize contrast — was measured against a control:

| variant | CER | exact match |
|---|---|---|
| **raw (control)** | **0.2291** | 30.6% |
| upscale only | 0.2301 | 29.8% |
| gray + contrast | 0.2380 | 30.6% |
| upscale + gray + contrast | 0.2410 | 30.6% |
| + sharpen | 0.2500 | 27.4% |

Every step made it worse, and the original default was the second-worst setting available.
PaddleOCR already resizes and normalizes each crop to its own input specification, so
preprocessing first resamples twice and destroys detail the model would have used.
`Preprocess()` now applies nothing by default.

An earlier ablation on *synthetic* renders showed no effect either way and was reported as such.
Only real crops showed the harm — which is the argument for labelling real data rather than
trusting a clean proxy.

### End to end: where the losses actually happen

Running the whole pipeline — detect → crop → read → validate — over the same 124 labelled plates,
with every plate attributed to the stage that lost it:

| | count | share |
|---|---|---|
| correct | 72 | 58.1% |
| **detection miss** | **0** | **0%** |
| grammar rejected | 30 | 24.2% |
| OCR error | 21 | 16.9% |
| grammar corrupted a correct read | 1 | 0.8% |

End-to-end accuracy **58.1%**, precision **76.6%** of the 94 plates it chose to log.

**Detection lost nothing.** Not one of the 124 plates was missed — consistent with the detector's
0.998 recall. Every remaining failure is a reading failure, so more detector training would buy
exactly zero.

Split by region, the pattern from the OCR section repeats and sharpens:

| | correct | rejected | OCR error | corrupted |
|---|---|---|---|---|
| **Indian** (65) | 67.7% | 13.8% | 18.5% | 0% |
| **European** (59) | 47.5% | 32.2% | 18.6% | 1.7% |
| — of which **Polish** (18) | **72.2%** | 11.1% | 16.7% | 0% |

### Adding one grammar was worth more than any model change

The first end-to-end run scored **42.7%**, with European plates at 15.3% and two thirds of them
*refused rather than misread* — the grammars modelled India and Germany while the dataset is full
of Polish plates. So Poland was modelled next. Nothing about the detector or the OCR engine
changed:

| | end-to-end | European | Polish | precision |
|---|---|---|---|---|
| India + Germany | 42.7% | 15.3% | — | 71.6% |
| + Polish standard | 50.0% | 30.5% | 50.0% | 71.3% |
| + Polish individual plates | **58.1%** | **47.5%** | **72.2%** | **76.6%** |

**+15 points end to end, and precision rose too** — usually those trade against each other.

Polish plates now score *higher than Indian ones*, and the reason is in the format: Poland bans
**B, D, I, O and Z from the series** precisely because they resemble digits. The country removed
the ambiguity at the design stage, so a `B` there is not a probable `8` but a certain one. That
is the cleanest case in this project of a real-world format encoding exactly the constraint an
OCR pipeline needs.

Modelling individual (vanity) plates mattered for a reason measurement found rather than
foresight: every corrupted read was a 1-letter Polish plate that a *single* edit pushed into the
2-letter standard shape — `P74103` → `PT4103`, `W2515T` → `WZ515T`. The grammar was inventing a
standard plate out of a valid individual one. Corruptions fell from 3 to 1.

### Two bugs this found

**Indian plates may have one to four trailing digits, not exactly four.** The grammar demanded
four and silently rejected `KL54H369`, `TN58AM1` and `KL7BZ99` — all real. Unit tests missed it
because they were written from the same assumption as the code. Loosening it recovered 3 plates
(40.3% → 42.7% end to end, Indian 63.1% → 67.7%) at the cost of ~2 points of precision, and of
some correction power: `MH12ABB234` is now a legitimate reading, so the grammar no longer repairs
that `B` as an `8`. Both halves of that trade are in the test suite.

**Preprocessing was hurting**, as the ablation above shows. Removed.

---

## Pipeline

```
source ──▶ FrameSource ──▶ Detector ──▶ Tracker ────────▶ per-track crop buffer
(file/img/dir/cam/rtsp)    (YOLO)       (IoU)                       │
                                                                    ▼
   Excel log ◀── Deduplicator ◀── Validator ◀── Voter ◀─────────── OCR
   (openpyxl)    (cooldown)       (IN / DE)     (multi-frame)      (PaddleOCR)
```

Three design decisions shape everything else.

**Results are emitted per track, not per frame.** A vehicle visible for 40 frames produces one
row, voted across all 40 reads — not 40 rows of varying quality.

**Detection is region-agnostic; reading is not.** A plate detector mostly learns "small bright
rectangle on a vehicle" and transfers across countries, so it trains on whatever is openly
licensed. The India/Germany specificity lives in the grammars, which need no training data.

**Correction is grammar-constrained.** OCR confuses `0/O`, `1/I`, `5/S`, `8/B`. Rewriting those
blindly makes accuracy *worse* — it breaks every plate that legitimately contains a zero.
Correcting against a grammar is safe, because the grammar says which positions hold digits:

```
MH12A81234   ->  MH 12 AB 1234   (India, 1 fix)
0L01CAB1234  ->  DL 01 CAB 1234  (India, 1 fix)
DAXYI23      ->  DA-XY 123       (Germany, 1 fix)
M-AB 123E    ->  M-AB 123E       (München, electric)
hello        ->  rejected
```

That last line matters: `hello` uppercases to `HELLO`, and `O→0` turns it into `HEL-L 0`, a
structurally valid German plate. A confidence floor rejects it — a false plate in the log is
worse than a missed one.

---

## Dataset

3,105 images / 3,273 plates, built from two Roboflow Universe datasets, both **CC BY 4.0**:

| Source | Images | Licence |
|---|---|---|
| [European License Plates](https://universe.roboflow.com/e-hh49k/european-license-plates-tjviy) | 1,455 | CC BY 4.0 |
| [Indian License Plate](https://universe.roboflow.com/nivu/indian-license-plate-knte7) | 1,650 | CC BY 4.0 |

Split 2174 / 466 / 465, **grouped, region-stratified and deterministic**:

- **Grouped** — frames from one video are near-duplicates; a per-image split evaluates the model
  on pictures it has memorized.
- **Stratified** — an unlucky split can leave test almost entirely Indian, resting the European
  number on a handful of images.
- **Deterministic** — `blake2b` of the group key, not `hash()`, which Python salts per process.

The upstream train/valid/test split is deliberately discarded: it is random per image, so
near-duplicate shots of one scene straddle it.

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
```

```bash
alpr env      # interpreter, Colab and GPU status
pytest        # 485 tests
```

GPU work runs on **Google Colab (T4)** — every notebook is self-contained and rebuilds whatever
it needs, because Colab's free tier gives one session and `/content` does not survive it.

| Notebook | Does |
|---|---|
| [`01_build_dataset`](notebooks/01_build_dataset.ipynb) | download → manifest → split → YOLO export |
| [`02_train_detector`](notebooks/02_train_detector.ipynb) | train on T4 |
| [`03_evaluate_detector`](notebooks/03_evaluate_detector.ipynb) | test mAP, size slices, failure gallery |

Credentials come from Colab's secret store, never from a cell. A `gitleaks` pre-commit hook and a
CI test both refuse credential-shaped strings in the repo.

---

## Status

| Phase | | |
|---|---|---|
| 0 | Foundation — package, CI, Colab bridge | ✅ |
| 1 | Dataset — ingest, grouped split, export | ✅ |
| 2 | Detector training | ✅ |
| 3 | Detection evaluation + failure gallery | ✅ |
| 4 | OCR (PaddleOCR) + CER ablation | ✅ |
| 5 | Plate grammars (India, Germany) | ✅ |
| 6 | Tracking + multi-frame voting | ✅ |
| 7 | Excel logging + deduplication | ✅ |
| 8 | End-to-end evaluation | ✅ |
| 9 | Live webcam / RTSP (Apple Silicon, MPS) | ✅ |

See [ROADMAP.md](ROADMAP.md) for the full plan and each phase's exit criteria.

**Phase 4 note:** neither source dataset ships plate *text*, so 124 test crops were
hand-labelled with `alpr label` to make CER measurable. See
[Reading plates](#reading-plates-measured-not-asserted).

### Live performance

Measured end to end on an M4 MacBook Air, detection on Metal/MPS and recognition on CPU:

| | |
|---|---|
| Detection | 34.6 fps (29 ms/frame) |
| OCR per plate crop | 41.1 ms |
| Full pipeline, `ocr_every=1` | 14.3 fps |
| Full pipeline, `ocr_every=3` | **23.5 fps** |

Recognition costs more than detection, which is why OCR does not run on every frame. A track
does not need forty reads — voting is decisive with a handful — so reading each track once every
three frames keeps the vote strong and the loop real-time.

```bash
alpr run --source 0 --weights best.pt --out plates.xlsx     # webcam
alpr run --source rtsp://camera.local/stream --region DE    # network camera
alpr run --source clip.mp4 --ocr-every 1                    # offline, most accurate
alpr run --source photo.jpg                                 # one still image
alpr run --source photos/                                   # a folder of stills
```

### Stills are not video, and score lower

A single image is accepted, but it is worth knowing what it gives up.

Tracking and voting exist because consecutive video frames show the *same* vehicle, so forty
imperfect reads collapse into one confident answer. A photograph offers exactly one read. There
is nothing to vote on, and the number you get is **raw OCR accuracy** rather than the voted
figure quoted above.

Two things change automatically for stills. The confirmation thresholds drop to 1, because
`min_hits=3` would otherwise confirm no track at all and log nothing. And the tracker is rebuilt
between images — across unrelated photographs it would otherwise link two different cars that
happen to sit in similar positions and vote their plates together.

---

## Layout

| Path | Contents |
|---|---|
| `src/alpr/` | The package — all logic worth testing lives here |
| `notebooks/` | Thin Colab drivers; no business logic |
| `configs/` | Training config, committed for reproducibility |
| `tests/` | 485 tests, run in CI without a GPU |

Notebooks stay thin on purpose: anything worth testing belongs in `src/alpr/`, so Colab never
becomes the place where the real code lives.

---

## Attribution

Derived from two Roboflow Universe datasets, both CC BY 4.0:

- European License Plates — https://universe.roboflow.com/e-hh49k/european-license-plates-tjviy
- Indian License Plate (NIVU) — https://universe.roboflow.com/nivu/indian-license-plate-knte7

## License

[MIT](LICENSE).
