---
license: mit
tags:
  - object-detection
  - license-plate-detection
  - ultralytics
  - yolov8
  - alpr
library_name: ultralytics
pipeline_tag: object-detection
---

# ALPR plate detector (YOLOv8s)

Single-class license plate detector, trained as Phase 2 of
[fayazhussain2821/Automatic-License-Plate-Recognition](https://github.com/fayazhussain2821/Automatic-License-Plate-Recognition)
— an end-to-end pipeline that
detects plates in video, reads them, validates them against Indian and German plate grammars, and
logs one deduplicated row per vehicle to an Excel workbook.

## Results

Evaluated on a held-out test split of 465 images the model never saw.

| Metric | Test |
|---|---|
| mAP@50 | **0.9921** |
| mAP@50-95 | 0.8377 |
| Precision | 0.9816 |
| Recall | **0.9917** |
| Inference | 4.7 ms/image (T4) · 29 ms/frame (Apple M4, MPS) |

**Recall is the metric that matters for a plate pipeline.** A plate the detector misses can never
be read downstream — that error is unrecoverable. A false positive produces a crop, OCR emits
noise, and a grammar check rejects it. The error profile is the right way round: **1 missed plate
against 23 false positives** across the whole test split.

### Detection by plate size

| Ground-truth width | n | Precision | Recall |
|---|---|---|---|
| tiny (<32 px) | 8 | 1.0000 | 1.0000 |
| small (32–64 px) | 64 | 0.9000 | 0.9844 |
| medium (64–128 px) | 191 | 0.9598 | 1.0000 |
| large (≥128 px) | 190 | 0.9845 | 1.0000 |

Small plates were expected to cap accuracy. They do not — every plate under 32 px was found.

### The test split was audited for leakage

A perceptual-hash audit found **5.8% of test images had a near-duplicate in train** (consecutive
video frames whose filenames the grouped-splitting logic did not recognise as frames of one clip).
Re-scoring on only the 438 uncontaminated images:

| | Full test (465) | Uncontaminated (438) |
|---|---|---|
| Recall | 0.9979 | 0.9978 |
| Precision | 0.9545 | 0.9556 |

The leak was **not** carrying the score. Grouping was fixed so future splits cluster duplicates.

## Training

| | |
|---|---|
| Architecture | YOLOv8s (11.1M parameters) |
| Epochs | 100 |
| Image size | 640 |
| Hardware | Google Colab T4, 1.25 h |
| Data | 3,105 images / 3,273 plates, split 2174/466/465 |

Augmentation was tuned for plates rather than for COCO: **no vertical flip** (a plate is never
upside down), and rotation and perspective *enabled* (Ultralytics defaults both to zero, which
under-trains the variation this task actually has).

Full arguments are in
[`results/args.yaml`](https://github.com/fayazhussain2821/Automatic-License-Plate-Recognition/blob/main/results/args.yaml).
Ultralytics changes augmentation defaults between minor releases, so reproducing this needs the
arguments *and* the version that consumed them.

## Usage

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict("car.jpg", conf=0.25)
```

Or through the project's pipeline, which adds tracking, OCR, plate-grammar validation and Excel
logging:

```bash
alpr run --source 0 --weights best.pt --out plates.xlsx
```

## Training data and attribution

Derived from two Roboflow Universe datasets, **both CC BY 4.0**:

- [European License Plates](https://universe.roboflow.com/e-hh49k/european-license-plates-tjviy) — 1,455 images
- [Indian License Plate (NIVU)](https://universe.roboflow.com/nivu/indian-license-plate-knte7) — 1,650 images

## Limitations

**It only finds plates. It does not read them.** Recognition is a separate stage in the pipeline.

**Boxes are axis-aligned**, so a plate photographed obliquely comes out as a rectangle around a
slanted plate. Downstream OCR cannot un-skew it without the four corners, which this model does
not predict.

**The training data is European and Indian.** Plates from other regions may work — a detector
largely learns "small bright rectangle on a vehicle" — but that is untested here.

**Test images averaged 119×40 px plates.** Performance on much smaller plates, heavy motion blur
or night footage is not characterised by these numbers.

## Intended use and responsible use

Built as a portfolio and learning project. Automatic plate recognition is surveillance
technology, and a plate is **personal data** — under the GDPR in the EU, and under comparable
regimes elsewhere. Deploying it against public traffic engages legal obligations around lawful
basis, retention, and notice that are the deployer's responsibility, not the model's.

Reasonable uses: research, private property you control, and datasets you have the right to
process. This model should not be used to track individuals.
