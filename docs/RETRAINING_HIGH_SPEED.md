# Retraining the detector and the OCR layer for high-speed vehicles

Step-by-step guide. Two independent models have to be retrained, and they fail
for different reasons, so they need different data and different procedures:

| Layer | Model | Where it lives | Retraining status |
|---|---|---|---|
| Plate **detection** | YOLOv8s, 1 class (`plate`) | `services/alpr/best.pt`, trained via `services/alpr/src/alpr/train.py` | **A pipeline already exists** — `services/backend/scripts/finetune_fast_motion.py`. It has been run once. Needs its stale paths fixed and a better dataset. |
| Plate **recognition (OCR)** | PaddleOCR PP-OCR recogniser (pretrained, off the shelf) | `services/alpr/src/alpr/ocr.py` | **No retraining path exists at all.** Has to be built. Section 3 does that. |

Before touching either, read section 0 — the physics decides your
hyperparameters, and there is one measured negative result that will otherwise
cost you a week.

---

## 0. Why high speed breaks the pipeline (and what does *not* fix it)

At a 1/60 s shutter, a vehicle at 100 km/h travels ~46 cm during the exposure.
At typical gantry framing that smears the plate **15–40 px horizontally**.
Consequences, in order:

1. **The detector fails first.** A smeared plate has weak vertical edges and
   low local contrast. Box confidence drops under `ANPR_CONFIDENCE` (0.25,
   `services/backend/backend/config.py:38`) and **no crop is produced**. No
   crop means no OCR, however good the OCR is. Fix the detector first — OCR
   improvements on frames that are never cropped are worth exactly zero.
2. **OCR degrades second.** Characters merge along the direction of travel:
   `8`→`B`, `0`→`D`, `1`→`I`/`7`.
3. **Voting partly rescues it.** `services/alpr/src/alpr/vote.py` votes
   *per character* across a track, weighted by confidence, so a plate no single
   frame read correctly can still come out right. A fast vehicle is in frame for
   fewer frames, so there is less to vote over — which is why raising detector
   recall pays twice: more crops *and* a stronger vote.

### The one thing that does not work: preprocessing the crop

Do not spend time on sharpening, deblurring, CLAHE or upscaling before OCR.
This was measured on 124 hand-labelled real crops and it made accuracy
**worse**: CER 0.2291 raw vs 0.2410 enhanced. PaddleOCR normalises internally,
so enhancing first resamples twice and destroys detail. That result is why
every field of `Preprocess` in `services/alpr/src/alpr/ocr.py:78` defaults to
off, and why `preprocess_plate_crop` in
`services/backend/backend/services/anpr_service.py:226` is not in the hot path.

**The fix has to be in the weights, not in a filter at inference time.**

---

## 1. Fix the existing fast-motion pipeline before you run it

`services/backend/scripts/finetune_fast_motion.py` was written and run against
an older directory layout (the repo was de-submoduled and services moved under
`services/`). Three things are stale and will fail or silently mislead:

**1a. The default base-weights path no longer exists.**
`scripts/finetune_fast_motion.py:96` resolves
`REPO_ROOT / "Automatic-License-Plate-Recognition" / "best.pt"`. That directory
is gone; the weights are now `services/alpr/best.pt`.

```bash
cd /Users/aarushsharma/Documents/SIH/services/backend && sed -i '' 's|REPO_ROOT / "Automatic-License-Plate-Recognition" / "best.pt"|REPO_ROOT / "alpr" / "best.pt"|' scripts/finetune_fast_motion.py
```

**1b. `REPO_ROOT` and `ARCHIVE_DIR` are wrong by one level.**
`BASE_DIR` is `services/backend`, so `REPO_ROOT` is `services/`, and
`ARCHIVE_DIR = REPO_ROOT / "archive"` points at `services/archive` — which does
not exist. The footage is at the repo root, `archive/`. Change `REPO_ROOT` to
`BASE_DIR.parent.parent`.

**1c. The committed `data.yaml` has an absolute path into a deleted directory.**
`training/fast_motion/dataset/data.yaml` says
`path: /Users/.../SIH/vehicle-intelligence-backend/training/fast_motion/dataset`.
Rewrite it (`--from-videos` regenerates it correctly, but if you reuse the
existing dataset you must fix it by hand), and delete the stale
`labels/train.cache` / `labels/val.cache` — Ultralytics will happily reuse a
cache keyed to the old paths.

**1d. Delete the stale label caches whenever you change the dataset.**

```bash
rm -f /Users/aarushsharma/Documents/SIH/services/backend/training/fast_motion/dataset/labels/*.cache
```

### What the previous run actually achieved

From `training/fast_motion/runs/fast_motion/provenance.json`:

| | mAP@50 | mAP@50-95 | Precision | Recall |
|---|---|---|---|---|
| before | 0.9570 | **0.8273** | 0.9162 | 0.9106 |
| after | **0.9878** | 0.7753 | 0.9633 | 0.9497 |

Read this honestly. **mAP@50 and recall went up; mAP@50-95 went down by 5.2
points.** The model got better at *finding* blurred plates and worse at
*localising them tightly*. For this pipeline that is probably a net win — the
crop is padded 8% anyway (`ocr.py:60`) so a loose box still feeds OCR — but it
is a real regression and it has a likely cause: the run stopped at **epoch 15
of 30** (`results.csv` ends at 15, patience 10), and the labels were the base
model's own pseudo-labels, which are themselves loosely localised. Section 2
addresses both.

---

## 2. Retraining the detector

### Step 2.1 — Decide your data source

`--from-videos` self-labels the 8 clips in `archive/` with the current detector
at conf ≥ 0.60, then writes each frame clean plus N blurred copies sharing the
same label (motion blur along the plate's axis smears content without moving the
box centre, so the label stays valid). This is a legitimate and standard trick
for a corruption you can *simulate*, but note its ceiling: **the labels are only
as good as the current model's confident detections.** It cannot teach the model
plates it never knew, and it inherits the base model's localisation sloppiness —
which is exactly the mAP@50-95 regression above.

Preferred, if you can get it: a real labelled plate dataset via `--dataset`.
Then the blur is applied to *human* boxes and localisation does not degrade.
The `alpr` package has the tooling to build one (`alpr.data.ingest`,
`alpr.data.split`, `alpr.data.export`, driven by
`services/alpr/notebooks/01_build_dataset.ipynb`).

**Recommendation: do both.** Fine-tune on a real dataset with synthetic blur;
use `--from-videos` only to add your own camera's backgrounds.

### Step 2.2 — Inspect the dataset before spending GPU time

```bash
cd /Users/aarushsharma/Documents/SIH/services/backend && .venv/bin/python scripts/finetune_fast_motion.py --from-videos --build-only
```

Then actually open a dozen images from `training/fast_motion/dataset/images/train/`.
You are checking two things: that the blur looks like motion blur and not mush,
and that the boxes are on plates. The filenames encode the augmentation
(`..._blur13a-15.jpg` = 13 px kernel, −15°), so you can find the extremes.

If the 21 px variants are unreadable even to you, drop the top severity — a
detector trained on inputs from which no plate is recoverable learns to fire on
smears, which costs you precision in production.

### Step 2.3 — Tune the blur ladder to *your* cameras

`BLUR_KERNELS = (0, 5, 9, 13, 17, 21)` at `finetune_fast_motion.py:104` is
derived from 1/60 s shutter + typical gantry framing: 60 km/h ≈ 8 px,
100 km/h ≈ 14 px, 140 km/h ≈ 20 px. **This is the single most important thing
to change for a real deployment.** Compute your own ladder:

```
smear_px = (speed_kmh / 3.6) * shutter_seconds * (image_width_px / scene_width_m)
```

A 1/250 s shutter on a highway camera gives ~3.4 px at 100 km/h and the whole
existing ladder is training for a corruption you will never see. Measure your
cameras' shutter and framing, then set `BLUR_KERNELS` from that formula.

`BLUR_ANGLES = (0, 8, -8, 15, -15, 90)` — the `90` entry is vertical smear,
which only happens on a pitching mount or a vehicle crossing perpendicular to
the optical axis. Drop it for a gantry-only deployment.

### Step 2.4 — Run the fine-tune

```bash
cd /Users/aarushsharma/Documents/SIH/services/backend && .venv/bin/python scripts/finetune_fast_motion.py --from-videos --epochs 40 --batch 8 --blur-variants 3
```

The regime (defaults, all deliberate):

- `--lr0 0.0008` — 10× below a from-scratch run, with `lrf=0.05` and `cos_lr`.
  This is a *fine-tune*: the base weights already do sharp plates well and the
  goal is to add blur robustness **without forgetting that**. A full-LR run on a
  blur-heavy set trades sharp accuracy for blurred accuracy, which is a net loss
  — most vehicles are not speeding.
- `--freeze 10` — backbone frozen. Blur robustness is mostly a
  neck/head problem. If after two runs the blurred slice still lags, try
  `--freeze 6`; that is the correct knob, not the learning rate.
- `mosaic=0.5`, off for the last third. Mosaic shrinks objects into the
  small-object regime plates already occupy; on top of blur it manufactures
  examples no camera will produce.
- `flipud=0.0` — a plate is never upside down. `fliplr=0.5` is kept
  because detection only has to find a bright rectangle and halving the data is
  the bigger risk.

**Changes to make relative to the previous run:**
1. Raise `--epochs` to 40 and let it actually finish — the last run stopped at
   epoch 15 with mAP@50 still climbing (0.9782 → 0.9807 → 0.9836 on epochs
   13/14/15). Patience is `max(5, epochs//3)`; at 40 epochs that is 13.
2. `--blur-variants 3` instead of 2, but keep the clean copy — the 1:N
   clean:blurred ratio is what protects sharp-plate accuracy.
3. Do **not** pass `--skip-baseline`. The before/after comparison is the only
   thing that makes "this helped" a measurement instead of an assertion.

On Apple Silicon `--device mps` is auto-detected. The previous run took ~1136 s
for 15 epochs on 688 train / 131 val images; budget ~50 min for 40 epochs.

### Step 2.5 — Judge the result on the right metric

Every run writes `provenance.json` beside the weights. Accept the new weights if:

- **mAP@50 up** on the blurred val split — this is the metric that matters,
  because it measures *did we find the plate at all*, which is the failure mode.
- **Recall up** — a missed plate is unrecoverable; a loose box is not.
- **mAP@50-95 down by no more than ~5 points.** Beyond that the boxes are
  getting loose enough that the 8% crop padding stops saving you, and OCR will
  start seeing bumper.
- **Sharp-plate performance not regressed.** This is the check the script does
  *not* do for you, and you must do it: validate the new weights against the
  original sharp dataset, not just the blurred one.

```bash
.venv/bin/python -c "
from ultralytics import YOLO
YOLO('../backend/training/fast_motion/runs/fast_motion/weights/best.pt').val(data='<your sharp data.yaml>')"
```

### Step 2.6 — Deploy the weights

Nothing reads the run directory automatically. Point the backend at it:

```bash
echo 'ANPR_WEIGHTS_PATH=/Users/aarushsharma/Documents/SIH/services/backend/training/fast_motion/runs/fast_motion/weights/best.pt' >> /Users/aarushsharma/Documents/SIH/services/backend/.env
```

For Docker, edit the `ANPR_WEIGHTS_PATH` env in `docker-compose.yml` (currently
`/app/alpr/best.pt` on both the `backend` and `kafka_consumer` services — change
both) and make sure the run directory is copied in by
`services/backend/Dockerfile`.

### Step 2.7 — Lower the confidence floor, and know what it costs

Even a retrained detector will be less confident on smears. `ANPR_CONFIDENCE`
defaults to 0.25. Dropping it to ~0.15 recovers marginal fast plates. This is
safe *here* specifically because three filters sit downstream and will discard
the extra garbage: per-character voting (`vote.py`), the country grammar
(`plates/india.py`), and the duplicate cooldown (`dedup.py`). A false detection
that produces no grammar-valid plate never reaches the database.

---

## 3. Retraining the OCR layer

**There is currently no OCR training path in this repo.** `PlateReader`
(`services/alpr/src/alpr/ocr.py:201`) wraps a pretrained PaddleOCR recogniser
and nothing fine-tunes it. This section builds that path.

Do section 2 first. Ordering is not a preference: OCR is only ever asked about
crops the detector produced, so an OCR fine-tune on crops from the *old*
detector is trained on the wrong distribution.

### Step 3.0 — Consider not retraining at all

Three cheaper interventions to exhaust first, in order of payoff. Measure each
with `alpr.cer` before moving on:

1. **Widen the vote.** `DEFAULT_MIN_READS = 2` in `vote.py:34`. A fast vehicle
   yields fewer frames, so raising the per-frame *sample rate* on high-speed
   approaches buys more reads to vote over. This is free accuracy — pure
   aggregation of information the pipeline already has.
2. **Extend the grammar corrections.** `services/alpr/src/alpr/plates/correct.py`
   maps confusable characters against the Indian plate grammar
   (`plates/india.py`). Motion blur has a *characteristic* confusion set along
   the horizontal axis (`8`↔`B`, `0`↔`D`, `1`↔`I`↔`7`, `6`↔`G`). Adding those
   pairs is a 20-line change with no training at all. Verify with
   `cer.grammar_gain` — a grammar that "corrects" a correct read has made things
   worse, and that function is there to catch it.
3. **Confidence-weight the vote by blur.** Reads from frames with high local
   gradient energy are more trustworthy; feed that into `Read.confidence`.

Only if CER is still unacceptable is a rec-model fine-tune worth its cost.

### Step 3.1 — Build a labelled crop set (tooling already exists)

`services/alpr/src/alpr/label.py` is exactly this tool: `extract_crops` pulls
plate crops out of frames, `build_page` writes a browser labelling page, and
`load_labels` reads the results back. Use it, driven by `alpr label`:

```bash
cd /Users/aarushsharma/Documents/SIH/services/alpr && .venv/bin/alpr label --help
```

Target **≥1500 crops**, and this is the part you cannot shortcut: they must be
*real fast-vehicle crops*, stratified by measured smear. Synthetic blur works
for the detector (it only has to localise a rectangle) and does **not** work for
recognition — real motion blur co-occurs with rolling-shutter skew, sensor
noise, low light and JPEG artefacts, and a recogniser trained on clean
`cv2.filter2D` output learns the filter, not the world. Roughly:

- 40% clean / low smear (≤5 px) — the anti-forgetting anchor
- 40% moderate (6–15 px)
- 20% severe (>15 px), only where a human can still read the plate

Hold out 20% as a test split **before** you train, and never look at it until
step 3.4.

### Step 3.2 — Establish the baseline you will be judged against

```bash
cd /Users/aarushsharma/Documents/SIH/services/alpr && .venv/bin/python -m pytest tests/test_cer.py -q
```

Then score the current pretrained recogniser on your held-out split using
`alpr.cer.score` / `alpr.cer.compare`. Record **both** CER and exact-match
accuracy. Exact match is the number that matters operationally: a plate with one
wrong character is a wrong plate, not a 90%-right one — `cer.py` says so in its
own docstring and it is correct.

Report per-smear-bucket, not just in aggregate. An aggregate CER that improves
while the severe bucket regresses means you have optimised the wrong thing.

### Step 3.3 — Fine-tune the PP-OCR recogniser

The `paddleocr` pip package is inference-only. Training needs the PaddleOCR
**source** repo and a config YAML:

```bash
cd /private/tmp/claude-501/-Users-aarushsharma-Documents-SIH/4a3ceee2-67ec-4cc7-ab45-caeb3b57cfd4/scratchpad && git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git
```

Convert your labelled crops to PP-OCR rec format — one `label.txt` with
`relative/image/path\tPLATETEXT` per line, plus train/val lists. Then start from
a pretrained `PP-OCRv5` (or v4) rec checkpoint and configure:

- **`Global.pretrained_model`** — the downloaded rec checkpoint. Never train
  from scratch on 1500 crops; you will get a model that memorises them.
- **`Global.character_dict_path`** — a **custom dict of exactly
  `0-9` + `A-Z`** (36 symbols). This is the highest-leverage single change in the
  whole OCR track and it costs nothing: the stock dict carries thousands of CJK
  and punctuation symbols that an Indian plate can never contain, and every one
  of them is a way for a blurred character to be misread. Set
  `Global.use_space_char: false` too.
- **`Optimizer.lr.learning_rate`** — ~1e-4, cosine, with warmup. Same fine-tune
  logic as the detector: do not un-learn clean plates.
- **`Train.dataset.transforms.RecAug`** — enable, but **do not** add synthetic
  Gaussian blur. Your data is already the real corruption. Isotropic blur trains
  an invariance to something that never occurs.
- **`Architecture`** — leave alone. Changing the backbone invalidates the
  pretrained checkpoint, which is the only reason 1500 samples is enough.

Then export to inference format (`tools/export_model.py`) and, because
`ocr.py:243` falls back to the `onnxruntime` engine on arm64 (paddle's native
`paddle_static` engine segfaults loading PIR-format models there — see
`_paddle_static_engine_is_unreliable`), **also export to ONNX** with
`paddle2onnx` if you are deploying on Apple Silicon or Graviton. Skipping this
is the mistake that will cost you an afternoon.

### Step 3.4 — Wire the custom recogniser in

`PlateReader.__init__` (`ocr.py:201`) forwards `**kwargs` to `PaddleOCR`, so the
plumbing is already there. Pass the custom model directory and dict:

```python
reader = PlateReader(
    text_recognition_model_dir="/abs/path/to/exported/rec_plate",
    text_rec_score_thresh=0.3,
)
```

Two follow-ups that are easy to forget:

1. Add these as **settings**, not literals — `ANPR_REC_MODEL_DIR` alongside
   `ANPR_WEIGHTS_PATH` in `backend/config.py`, threaded through
   `anpr_service.reader` (`anpr_service.py:172`). Otherwise the Docker path and
   the local path diverge and only one of them works.
2. **Re-run the `Preprocess` ablation** (`cer.ablate`). The finding that
   preprocessing hurts was measured against the *stock* recogniser, which
   normalises internally. A custom-trained recogniser has different
   normalisation statistics and the answer could genuinely flip. Do not carry
   the old conclusion forward on trust — it is cheap to re-measure.

### Step 3.5 — Accept or reject

Accept the new recogniser only if, on the held-out split:

- exact-match accuracy up in the moderate and severe buckets, **and**
- exact-match accuracy in the clean bucket down by **< 1 point**.

A recogniser that reads speeders better and ordinary traffic worse is a net loss
in every deployment, because ordinary traffic is almost all of the traffic.

---

## 4. End-to-end verification

Retraining is not done until the whole pipeline is measured, because the
detector and the recogniser interact: looser boxes change what OCR sees.

```bash
cd /Users/aarushsharma/Documents/SIH/services/alpr && .venv/bin/python -m pytest -q
```

Then run the end-to-end evaluation — `services/alpr/src/alpr/endtoend.py`
classifies each plate into outcome buckets (missed by detector / detected but
misread / correct), which is the only view that tells you *which layer* is still
the bottleneck. If the dominant bucket is still "missed by detector", go back to
section 2 and stop tuning OCR.

Finally, in the running system, use the sandbox at `/app/test` to upload a
high-speed clip and watch it process, and check the benchmark suite at
`/app/benchmarks` for the FPS cost — a heavier recogniser reduces throughput,
and `services/backend/backend/services/benchmark_service.py` is what quantifies
whether you can still keep up with the camera count you have projected.

---

## 5. Checklist

- [ ] Fix `REPO_ROOT`, `DEFAULT_WEIGHTS`, `ARCHIVE_DIR` in `finetune_fast_motion.py` (§1a–1b)
- [ ] Fix the absolute `path:` in `training/fast_motion/dataset/data.yaml`; delete `*.cache` (§1c–1d)
- [ ] Recompute `BLUR_KERNELS` from your cameras' shutter + framing (§2.3)
- [ ] `--build-only`, then eyeball the dataset (§2.2)
- [ ] Fine-tune 40 epochs, keep the baseline comparison (§2.4)
- [ ] Validate against the **sharp** dataset too — the check the script omits (§2.5)
- [ ] Set `ANPR_WEIGHTS_PATH` locally **and** in both docker-compose services (§2.6)
- [ ] Exhaust vote-widening + grammar corrections before OCR training (§3.0)
- [ ] Label ≥1500 **real** fast-vehicle crops, stratified, 20% held out (§3.1)
- [ ] Baseline CER *and* exact-match, per smear bucket (§3.2)
- [ ] Fine-tune PP-OCR rec with a **36-character custom dict** (§3.3)
- [ ] Export ONNX as well as paddle inference format if on arm64 (§3.3)
- [ ] Make the rec model dir a setting, not a literal (§3.4)
- [ ] Re-run the `Preprocess` ablation against the new recogniser (§3.4)
- [ ] End-to-end outcome buckets + benchmark FPS (§4)
