# Repository Audit — Automatic-License-Plate-Recognition

Audit date: 2026-08-12
Commit audited: `6bffdc6` ("Name the photograph each still-image row came from (#28)"), branch `main`, working tree clean.
Scope: read-only. No source file was modified; this document is the only file added.

Verification performed during the audit:

- `pytest` run locally against `.venv` — **518 tests collected, 518 passed, 0 failed** (GPU/weights/camera-marked tests excluded by the default `addopts`).
- Selected runtime behaviours were confirmed by executing the package (region filtering in `alpr.plates.parse`, CLI argument defaults) rather than by reading alone. Those are flagged as *verified* below.
- Every claim about "not wired in" was checked with a repo-wide grep across `src/`, `tests/` and `notebooks/`.

---

# Repository Structure

```
.
├── .github/workflows/ci.yml        # the only CI workflow
├── .pre-commit-config.yaml         # ruff, nbstripout, gitleaks, hygiene hooks
├── pyproject.toml                  # hatchling, src-layout, extras: ocr / live / dev
├── README.md                       # 362 lines — the real project documentation
├── ROADMAP.md                      # 147 lines — 10 phases with exit criteria
├── configs/detector.yaml           # committed training config (71 lines, heavily commented)
├── data/README.md                  # scratch space, fully gitignored
├── notebooks/                      # 4 Colab drivers (phases 0–3 ONLY)
│   ├── 00_colab_bootstrap.ipynb
│   ├── 01_build_dataset.ipynb
│   ├── 02_train_detector.ipynb
│   └── 03_evaluate_detector.ipynb
├── results/                        # Phase 2 run artifacts + MODEL_CARD.md
│   ├── MODEL_CARD.md  args.yaml  results.csv  results.png  confusion_matrix.png
├── src/alpr/                       # 32 files, 6,633 lines — all the logic
│   ├── __init__.py                 # phase→module map, __version__ = 0.1.0
│   ├── build.py                    # one-call dataset construction (Roboflow sources)
│   ├── cer.py                      # edit distance, CER reports, ablation harness
│   ├── cli.py                      # argparse: env / train / run / label / fetch-data
│   ├── dedup.py                    # cross-pass cooldown deduplicator
│   ├── detect.py                   # Ultralytics wrapper, Detection, iou, device select
│   ├── dupes.py                    # dHash near-duplicate / leakage audit
│   ├── endtoend.py                 # Phase 8 failure-attribution evaluation
│   ├── env.py                      # Colab/GPU probing, credential fetch
│   ├── evaluate.py                 # Phase 3 matching, size slices, failure gallery
│   ├── excel.py                    # JSONL write-ahead log + openpyxl workbook
│   ├── label.py                    # crop sampling + self-contained HTML labelling page
│   ├── ocr.py                      # PaddleOCR TextRecognition wrapper, Preprocess
│   ├── pipeline.py                 # the orchestrator
│   ├── sources.py                  # video / camera / RTSP / stills frame sources
│   ├── track.py                    # greedy-IoU tracker
│   ├── train.py                    # TrainConfig, provenance, validate_dataset, train
│   ├── viewer.py                   # OpenCV annotation, window, mp4 writer
│   ├── vote.py                     # per-character confidence-weighted voting
│   ├── data/                       # schema, ingest, manifest, split, export, stats
│   └── plates/                     # base, correct, india, germany, poland
├── tests/                          # 25 files, 4,645 lines, 518 tests
├── ui/__init__.py                  # "# UI Package"      — empty placeholder
├── agent/__init__.py               # "# Agent Package"   — empty placeholder
└── mcp_servers/__init__.py         # "# MCP Servers Package" — empty placeholder
```

Test-to-source ratio is 0.70:1 by line count, with a test module per source module — an unusually disciplined structure for a portfolio project.

**There is no `ui/`, `agent/` or `mcp_servers/` implementation.** All three are single-comment placeholder packages, not referenced anywhere in `src/`, `tests/` or `pyproject.toml`. The only user interfaces that exist are the `alpr` CLI, the OpenCV viewer window, and the generated `label.html` page.

Git history: 61 commits, one merged branch per roadmap phase plus targeted `fix/*` branches. History is clean and each phase is traceable to a PR.

---

# Current Architecture

The system is a linear, single-threaded pipeline with one deliberate concurrency point (the live-source reader thread). Layering is strict: no module imports a heavier one, and `ultralytics`, `torch`, `paddleocr` and `cv2` are all imported *inside functions*, never at module scope.

```
                 alpr.cli  (argparse entrypoint)
                     │
    ┌────────────────┴──────────────────────────────────┐
    │                                                   │
alpr.build ──▶ alpr.data ──▶ alpr.train           alpr.pipeline
 (fetch)      (schema/ingest/    (Ultralytics)          │
              manifest/split/                           │
              export/stats)                             │
                    │                                   │
              alpr.evaluate                             │
              alpr.dupes        ┌────────────────────┬──┴────────────┬──────────────┐
              alpr.endtoend     │                    │               │              │
              alpr.cer      alpr.sources        alpr.detect     alpr.ocr      alpr.excel
              alpr.label    (file/cam/rtsp/img)   (YOLO)      (PaddleOCR)   (JSONL WAL
                                                     │             │         + openpyxl)
                                                alpr.track ──▶ alpr.vote ──▶ alpr.plates
                                                (greedy IoU)  (per-char)   (IN/DE/PL grammars)
                                                                                │
                                                                           alpr.dedup
                                                                           (cooldown)
```

Key architectural decisions, all of them stated and defended in module docstrings:

| Decision | Where | Rationale as given |
|---|---|---|
| Manifest (JSONL) is the source of truth; the YOLO tree is a generated artifact | `data/schema.py`, `data/export.py` | YOLO label files cannot carry `PlateBox.text`, which OCR scoring needs |
| Detection region-agnostic, reading region-specific | `plates/__init__.py`, `data/schema.py` | A plate detector transfers across countries; grammars need no training data |
| Emit one row per **track**, not per frame | `track.py`, `pipeline.py` | A vehicle visible 40 frames must produce 1 row, voted across all 40 |
| Per-**character** voting, confidence-weighted | `vote.py` | String-majority throws away signal when every read differs |
| Correction constrained by grammar, bounded edit search | `plates/correct.py` | Blind confusable rewriting breaks every plate containing a real `0` |
| Excel via a JSONL write-ahead log | `excel.py` | `.xlsx` is a zip of XML; per-row saves are O(n²), end-only saves lose the run |
| Threaded, frame-dropping capture for live sources only | `sources.py` | On a file, dropping loses data; on a camera, keeping loses time |
| Ultralytics / Paddle / cv2 imported lazily | everywhere | `alpr env` must stay instant; CI must run without a GPU |

---

# Current Data Flow

## 1. How data enters the system

Two entry points, and they are completely separate.

**Training data** enters through `alpr.build.SOURCES` — a hard-coded tuple of two `RoboflowSource` records (European License Plates `e-hh49k/european-license-plates-tjviy`, Indian License Plate `nivu/indian-license-plate-knte7`), each carrying workspace, project, version, target directory, `Region`, source tag, id prefix, licence string and URL. `download_sources()` uses the `roboflow` client with a key obtained from `alpr.env.get_credential` (Colab secret store, then environment; **no hardcoded fallback anywhere**). A failed download `rmtree`s the partial target so it can never ingest as a silently smaller dataset.

`ingest_sources()` → `from_roboflow_export()` → `from_yolo_dir()` per split directory. Ingest:
- pools `train/`, `valid/`, `val/`, `test/` — **Roboflow's own split is deliberately discarded** and preserved only as `meta["roboflow_split"]`;
- namespaces `image_id` with `{id_prefix}{split_dir}-{stem}` so stem collisions across upstream splits cannot silently collide;
- reads image dimensions from the header via PIL (`image_size`) rather than decoding pixels;
- collapses all source class ids to a single `license_plate` class;
- records rather than hides problems: `IngestReport` carries `skipped_no_label`, `skipped_bad_label`, `dropped_boxes`.

**Inference data** enters through `alpr.sources.open_source(spec)`, which dispatches on the spec: `int` or digit string → `CameraSource`; `rtsp://`/`rtsps://`/`http(s)://` → `RtspSource`; directory → `ImageSource` over sorted images; image extension → `ImageSource`; anything else → `VideoFileSource`. All yield a frozen `Frame(index, image, timestamp, source_name)`.

## 2. How datasets are split

`alpr.data.split.split_records()` — three properties, each defended:

- **Deterministic.** `_group_rank` is `blake2b(f"{seed}:{group_key}")`, explicitly not `hash()` (Python salts string hashing per process).
- **Grouped.** Assignment is per `ImageRecord.group_key`, which is `group` if set, else `image_id` with `_FRAME_SUFFIX` stripped. That regex matches two shapes: `..._frame_0137` and `..._mp4-t-1062` (the Roboflow video-frame convention). A bare trailing-number rule is deliberately *not* used, with the reason given: it would collapse `license_plate_205` and `license_plate_242` into one group.
- **Stratified by region.** Records are bucketed by `primary_region` (the modal region of an image's boxes), and each region is split to the target ratios independently. A group whose images span regions is claimed by whichever region reaches it first, so a group never straddles a split.

Placement is largest-group-first into whichever split has the largest *deficit* (`target − placed`), with the hash as tiebreak. Default ratios 0.70/0.15/0.15, normalized if they do not sum to 1. `verify_split()` then raises on unassigned groups, stale groups, or an empty split — and is called from `build_dataset()`, so it runs on every build rather than only in a notebook.

## 3. How duplicate/leakage detection works

`alpr.dupes` implements a **dHash** (9×8 greyscale thumbnail, adjacent-pixel comparison, 64-bit) perceptual audit:

- `find_duplicates()` hashes every record, buckets identical hashes (linear), then compares distinct hashes pairwise for Hamming distance ≤ threshold (default 5 bits).
- `DuplicatePair.contaminates_evaluation` is the meaningful predicate: crosses splits **and** one side is `TRAIN`.
- `duplicate_clusters()` runs union-find so a chain A~B~C becomes one cluster.
- `regroup_by_duplicates()` rewrites `record.group` to `dup:<root>` — the general fix.
- `clean_subset()` returns the uncontaminated subset of a split, enabling re-scoring without retraining.

**This is where the largest gap in the repository sits.** Verified by grep: `find_duplicates`, `duplicate_clusters`, `regroup_by_duplicates` and `clean_subset` are referenced **only by `tests/test_dupes.py`**. They are not called from `build_dataset()`, not from `ensure_dataset()`, not from any notebook, and not exposed by the CLI. The 5.8%-contamination audit the README reports was therefore run ad hoc in a session that no longer exists, and it is not reproducible from anything committed.

The fix that *is* in the codebase is narrower than the README implies: the `-mp4-t-\d+` alternative was added to `_FRAME_SUFFIX` in `data/schema.py`, which catches the specific Roboflow naming pattern that leaked. Perceptual regrouping — the general fix, already written and tested — remains disconnected.

## 4. How YOLO is trained and evaluated

**Training** (`alpr.train`): `TrainConfig` is a frozen dataclass loaded from `configs/detector.yaml`. `from_yaml` **rejects unknown keys** (a typo would otherwise silently fall back to a default), routing deliberate passthroughs to an `extra:` dict. `__post_init__` validates epochs, batch, `imgsz % 32 == 0` (Ultralytics silently rounds otherwise), the [0,1] range of flip/mosaic probabilities, and `close_mosaic <= epochs`.

`validate_dataset()` runs before any GPU time is spent: checks the yaml exists, has `train`/`val`/`names`, and that both directories exist and are non-empty. `require_gpu()` (via `nvidia-smi`, not torch) fails fast on a CPU runtime. After the run, `provenance(...)` writes `provenance.json` — alpr version, **ultralytics version**, python version, timestamp, resolved config — into the directory Ultralytics actually used (`results.save_dir`), not a reconstructed `project/name`, because that reconstruction had previously written provenance into an empty directory.

Augmentation is plate-specific and justified per parameter: `flipud=0.0` (a plate is never upside down), `degrees=10.0` / `perspective=0.0005` / `shear=2.0` (Ultralytics defaults these to zero, under-training the actual variation), `mosaic=1.0` with `close_mosaic=10`, HSV jitter for headlights/dusk/sun. `fliplr=0.5` is explicitly left at the default and flagged as something Phase 3 should ablate — **that ablation was never run.**

**Evaluation** (`alpr.evaluate` + `notebooks/03`): Ultralytics `model.val(split="test")` gives mAP; then `alpr.evaluate` adds what Ultralytics does not — greedy confidence-ordered matching at IoU 0.5, **precision/recall bucketed by ground-truth plate width** (<32 / 32–64 / 64–128 / ≥128 px), and a colour-coded failure gallery (green=hit, red=miss, orange=FP) written to disk. False positives are attributed to the image's widest ground-truth plate, or skipped on an image with no plates — a documented approximation.

## 5. How detection results flow into tracking

`PlateDetector.detect()` returns `Detection` objects in **normalized** xyxy, sorted by descending confidence, so nothing downstream carries image dimensions. `Tracker.update(detections, frame_index)`:

1. `_associate()` computes IoU for every (track, detection) pair above `iou_threshold` (0.3), sorts candidates by overlap descending, and greedily claims each track and detection once.
2. Matched tracks update (`detection`, `last_frame`, `hits += 1`, `age = 0`, append to `history`).
3. Unmatched tracks age; any track with `hits >= min_hits` (3) is flagged `confirmed`.
4. Unmatched detections start new tracks.
5. `_retire()` drops tracks with `age > max_age` (15), pushing **only confirmed ones** onto `_finished`.
6. Returns the currently confirmed tracks.

`finish()` retires everything still live at end of stream, so a vehicle in frame on the last frame is not lost. `completed()` is a generator over `_finished` that sets `emitted = True` — this is the mechanism that guarantees one row per vehicle.

The choice of greedy IoU over ByteTrack/Kalman is argued in the docstring: one class, smooth motion, plates rarely occlude, and Ultralytics' tracking API would couple the whole pipeline to that library. Note that ROADMAP Phase 6 specifies ByteTrack; the implementation deliberately diverges and says so.

## 6. How OCR is performed

`alpr.ocr.PlateReader` wraps PaddleOCR's **`TextRecognition`** head only (not full detection+recognition — the YOLO detector already localizes). The model loads lazily on first `.model` access, and a missing `paddleocr` raises a directed `OcrError` naming the extra to install.

`read(image, detection)` → `crop_plate()` (normalized box → pixel crop with `padding=0.08` of box size per side, clamped to the frame, raising on a degenerate crop) → `prepare()` → `read_image()` (converts to RGB numpy, `model.predict`).

`_first_result()` is defensively written to handle both dict and attribute result shapes, because PaddleOCR's result type has changed across major versions and a silent shape mismatch would read as "OCR found nothing".

**`Preprocess()` defaults to doing nothing but padding**, and this is the strongest empirical result in the repo. The ablation on 124 hand-labelled real crops:

| variant | CER |
|---|---|
| raw (control) | **0.2291** |
| upscale only | 0.2301 |
| gray + contrast | 0.2380 |
| upscale + gray + contrast | 0.2410 ← the previous default |
| + sharpen | 0.2500 |

Every step hurt, and the shipped default had been the second-worst available setting. The explanation given — PaddleOCR resizes and normalizes internally, so preprocessing resamples twice — is coherent. The steps are retained as opt-in flags for a reader that does not normalize internally. The docstring also correctly states what preprocessing *cannot* do: perspective de-skew needs four corners, and the detector emits axis-aligned boxes, so that is a Phase-2 model change, not a preprocessing one.

## 7. How OCR results are aggregated

`alpr.vote` — per-character, confidence-weighted, in three steps:

1. **Length is decided first**, by summed confidence per length rather than by count, so a clear 10-character read outweighs several blurred 9-character ones. Only reads of the winning length vote on characters (otherwise characters smear into wrong slots).
2. **Per position**, characters accumulate their read's confidence; the winner is `max(sorted(votes), key=...)` — ties break alphabetically, deliberately, so the same input always votes the same way.
3. **Confidence = agreement × mean read confidence**, rounded to 3dp — because unanimity among four uncertain reads is not the same evidence as unanimity among four confident ones.

`VoteResult` also carries `per_character` agreements, `runner_up` (most-supported non-winning string), and `weakest_position()` for review. `TrackVoter` buffers `{track_id: [Read]}`; `pop()` votes and discards, enforcing one emission per track. Below `min_reads` (2), `vote()` returns `None` rather than a fragile answer, and the pipeline counts that as `too_few_reads` rather than dropping it silently.

## 8. How grammar correction works

`alpr.plates.correct.parse_plate(raw, formats, region, max_edits=2, min_confidence=0.7)`:

1. `district_hint(raw)` is extracted **before** normalization, because normalization drops the separator that distinguishes `M-AB` (München) from `MA-B` (Mannheim).
2. `normalize(raw)` uppercases, strips everything outside `[A-Z0-9ÄÖÜ]` (umlauts kept — LÖ, GÖ, WÜ are real districts), and drops a leading decorative `IND`.
3. For each candidate format, `correct_to_format()` runs a bounded edit search: try the exact string; if it matches at confidence ≥ 1.0, stop. Otherwise enumerate positions with a confusable alternative (`DIGIT_TO_LETTERS` / `LETTER_TO_DIGITS`), and for `n_edits` = 1 then 2, try every combination × product of replacements, scoring `match.confidence − n_edits × 0.18`. Break as soon as an edit count has produced an improvement, since a 2-edit reading can never beat a 1-edit reading of the same string.
4. The best match across formats wins on `(confidence, −edits)`.
5. **A confidence floor of 0.7 discards the rest.** This is the anti-hallucination guard: `hello` → `HELLO` → (O→0) → `HEL-L 0`, a structurally valid German plate, scores 0.80 − 0.18 = 0.62 and is rejected. Every genuine reading observed sat at 0.80 or above.

The nuance worth preserving: an *exact* match does not short-circuit the search unless it is fully confident, because `DAXYI23` parses as-is into unknown district DAX (0.80) while repairing I→1 yields Darmstadt's DA-XY 123 (1.00 − 0.18 = 0.82).

The three grammars:

- **India** (`india.py`) — `_STANDARD` = 2-letter state + 1–2 district digits + 0–3 series letters + **1–4** number digits, plus a separate `_BH` rule for Bharat series. The 1–4 relaxation is documented as a bug found by end-to-end measurement: the original "exactly four" silently rejected `KL54H369`, `TN58AM1`, `KL7BZ99`, and the unit tests missed it because they encoded the same wrong assumption as the code. A 39-entry closed `STATE_CODES` table scores confidence (1.0 known / 0.55 unknown) but never rejects.
- **Germany** (`germany.py`) — digits split the string first; the letter block's district/letters split is then resolved by `(matches district_hint, in DISTRICT_PREFIXES, longer)`. Hard cap of 8 body characters, floor of 4 (a legal 3-character plate is traded away against OCR noise). ~70 prefixes, used only to *raise* confidence (1.0 / 0.8) because Germany has ~700 and the set changes. `E`/`H` suffixes handled.
- **Poland** (`poland.py`) — the strongest grammar in the repo, and the reasoning is genuinely good: `Q` appears on no Polish plate, and **B, D, I, O, Z are banned from the series precisely because they resemble digits**, so a `B` there is a certain `8` rather than a probable one. Standard = 2–3 letter code + 4–5 series chars with at most 2 letters; a separate `PL-individual` rule (1 voivodeship letter + 3–5 chars, requiring at least one digit) at confidence 0.85. That second rule exists because measurement found every grammar-corrupted read was a 1-letter plate that a single edit pushed into the 2-letter standard shape (`P74103`→`PT4103`, `W2515T`→`WZ515T`); modelling individuals below standard confidence lets the exact individual match win. Corruptions fell 3 → 1.

## 9. How country/region is determined

Region is **not** inferred from the image, GPS, or any classifier. It is decided in exactly two ways:

1. **Optionally pinned** by the operator — `PipelineConfig.region` / `--region`, which restricts `parse()` to one grammar. The stated purpose is to stop an Indian reading winning on a German road.
2. **Otherwise inferred from which grammar wins**, by `(confidence, −edits)` across India, Germany and Poland. The winning `PlateMatch.region` becomes the logged `Region` and drives `format_display()`.

For the *dataset*, region is a static tag applied at ingest per source (`Region.EUROPE` for the European set, `Region.INDIA` for the Indian one), used only for split stratification and stats. `Region.EUROPE` is deliberately distinct from `Region.GERMANY` — the European set is pan-European, and tagging it `DE` would be a false claim about the data.

## 10. How deduplication works

Two distinct mechanisms for two distinct duplicates, and the separation is correct:

- **Within one pass** of a vehicle — handled entirely by tracking + `Track.emitted`. No cooldown involved.
- **Across passes** (car circles a car park, waits at a barrier, or the tracker loses and re-acquires it) — `alpr.dedup.Deduplicator`, an `OrderedDict[plate → Suppression]` with a 5-minute default cooldown. The window **restarts on each suppressed sighting**, so a vehicle sitting in view for twenty minutes yields one row, not one row per window. Bounded at `max_tracked=10_000` with oldest-first eviction so a long run cannot grow memory without bound.

`ExcelLog` owns a `Deduplicator` and consults it in `add()` before journalling. On `open()`, events recovered from an existing journal are **replayed through the deduplicator**, so a resumed run does not re-log what the interrupted one already recorded.

## 11. How results are persisted

`alpr.excel.ExcelLog` — a write-ahead log, which is the right answer to the fact that `.xlsx` is a zip of XML:

- Every accepted event is appended to `<out>.xlsx.jsonl` as one JSON line and `flush()`ed to the OS immediately. Cheap, and it survives a crash.
- The workbook is materialized from the buffered events every `flush_every` (50) events and on `close()`.
- `write_workbook()` writes to `<path>.tmp` and `replace()`s, so an interrupted write cannot leave a truncated workbook where a valid one was.
- `PermissionError` (the workbook is open in Excel — locks the file on Windows) is caught and re-raised as an `ExcelLogError` that explicitly says no data is lost and names the journal.
- `read_journal()` tolerates exactly one torn line — the last — and raises on a corrupt line anywhere else.
- `recover()` rebuilds a workbook from its journal.
- Timezone-aware datetimes are converted to local and stripped, because openpyxl refuses them; dropping the offset silently was rejected as worse.

Ten columns: Timestamp, Plate, Formatted, Region, Confidence, OCR fixes, Track, Frame, Source, Crop. Header styled, frozen panes, auto-filter, number formats on timestamp and confidence.

## 12. How live video differs from offline video

Four differences, three automatic:

| | Offline (`VideoFileSource`) | Live (`CameraSource` / `RtspSource`) |
|---|---|---|
| Threading | none — a plain read loop | background thread reads continuously |
| Frame handling | every frame processed, in order | **only the newest frame is kept**; the rest are counted in `dropped` |
| `Frame.index` | sequential 0,1,2,… | the capture sequence number, so gaps are visible |
| Failure | ends at EOF | `RtspSource` reconnects with exponential backoff (1s → 30s cap), counting `reconnects` |

The rationale is stated crisply: dropping frames on a file discards *data*; keeping them on a camera discards *time* — if detection takes 60 ms while the camera produces a frame every 33 ms, the buffer fills and the pipeline drifts seconds behind reality and never recovers. `CAP_PROP_BUFFERSIZE=1` is set as a secondary measure. `RtspSource` subclasses `CameraSource` and re-implements `frames()` only to pass `reconnect=True`.

Both live modes are explicitly local-only: Colab cannot reach a Mac's webcam or a LAN RTSP camera. Practical consequence: `--ocr-every` is the live throughput lever (measured 14.3 fps at 1, 23.5 fps at 3 on an M4).

**Stills are a third mode.** `ImageSource.is_still` is True, and `pipeline.run` reacts by using `config.for_stills()` (`min_hits=1, min_reads=1, ocr_every=1`) and by **rebuilding the tracker and voter between images** — otherwise two unrelated photographs with a plate in a similar position would be linked into one track and their plates voted together. Each frame carries `source_name` so a row traces back to its photograph. The README states plainly that stills give up multi-frame voting and therefore score at raw-OCR accuracy.

## 13. How evaluation is performed

Four independent evaluation layers, each answering a different question:

| Module | Question | Status |
|---|---|---|
| `alpr.evaluate` | Does detection find plates, and does it fail on small ones? | Implemented + notebook driver (`03`) |
| `alpr.cer` | Does preprocessing help? Do grammars beat raw OCR? | Implemented, **no notebook or CLI driver** |
| `alpr.endtoend` | Where does the whole system lose plates? | Implemented, **no notebook or CLI driver** |
| `alpr.dupes` | Is the test set actually unseen? | Implemented, **not wired anywhere** |

`alpr.endtoend` is the most valuable of the four. It runs detect → crop → read → validate on labelled ground-truth boxes and attributes each plate to the stage that lost it: `CORRECT`, `DETECTION_MISS`, `OCR_ERROR`, `GRAMMAR_REJECTED`, `GRAMMAR_CORRUPTED`. That last category — OCR was right and the grammar made it wrong — is the one most projects never measure, and the report prints an explicit WARNING when it is non-zero. `precision` correctly excludes rejected and missed plates from the denominator; `recall` is aliased to accuracy.

Ground truth for OCR comes from `alpr.label`: crops are cut from **ground-truth boxes, not detector output** (so a bad number cannot be ambiguous between finding and reading), sampled **round-robin across size buckets** (so the score is not flattered by easy large plates), and written to a self-contained `label.html` with crops as base64 data URIs, localStorage autosave and a download button — works from `file://`, which matters because crops are generated on Colab and labelled on a laptop. 124 labels exist locally (`labels.json`), correctly gitignored as personal data.

## 14. How the existing tests are organized

25 test modules, one per source module, mirroring `src/alpr/` exactly. 518 tests, all passing locally. Organized into `TestX` classes by behaviour, with docstrings that state *what is faked and why*.

The faking discipline is consistent and correct: heavy dependencies are stubbed, pure logic is tested for real.

- `test_pipeline.py` — `FakeSource` (scripted numpy frames), `FakeDetector` (one drifting plate for N frames), `FakeReader` (cycles scripted strings). Tests composition, not YOLO/Paddle. Includes the genuinely good `AlwaysDetects` "contamination trap" for stills: the same box position in three consecutive images must produce three tracks, not one.
- `test_ocr.py` — preprocessing tested for real (pure PIL work); the model stubbed via `_StubModel`; a marked test exercises the real reader for anyone who has it.
- `test_sources.py` — `VideoFileSource` tested against a real video written with OpenCV; camera and RTSP stubbed at the capture level, so the *frame-dropping and reconnection logic* is what gets tested.
- `test_notebooks.py` — guards the committed notebooks: valid JSON, **no saved outputs**, no removed `[gpu]` extra, no credential-shaped tokens (matched by *shape*, never against a real key), no literal API-key assignments. The docstring explains exactly why this is a test and not a hook: saving from Colab writes straight to the repo, bypassing pre-commit entirely, and that has already reverted a dependency fix twice.
- `test_plates.py` (51 tests) + `test_plates_poland.py` (19) — valid/invalid/confusable-corrupted cases per grammar, including regression tests for both halves of the Indian 1–4-digit trade-off.

Three opt-in markers (`gpu`, `weights`, `camera`) are excluded by default in `pyproject.toml`, with the stated reason that "a suite nobody can run green is a suite nobody runs".

## 15. What CI actually executes

`.github/workflows/ci.yml` — one job, `ubuntu-latest`, on push to any branch and PRs to main:

1. `actions/checkout@v5`
2. `actions/setup-python@v6` with Python **3.12** and pip cache
3. `pip install -e ".[dev]"` — base + dev only; **not** `[ocr]`, **not** `[live]`
4. `ruff check .` and `ruff format --check .`
5. `pytest` — which resolves to `-q -m 'not gpu and not weights and not camera'`

What CI does **not** do, all of it worth knowing:

- Does not run pre-commit, so **gitleaks never runs in CI**. The README's "a `gitleaks` pre-commit hook and a CI test both refuse credential-shaped strings in the repo" is only half true: the CI-side check is `test_notebooks.py`, which scans **notebooks only**, for five vendor prefixes. A key in a `.py` file, a config, or a markdown file passes CI.
- Does not test against Python 3.11 or 3.13, despite `requires-python = ">=3.11"`. Local development is on 3.13.
- Does not exercise `paddleocr`, `ultralytics` model loading, `cv2` windowing, or any GPU path. Every heavyweight integration is faked.
- No coverage measurement, no build/publish step, no dependency audit, no notebook execution.

---

# Existing ML Components

| Component | Module | State |
|---|---|---|
| Dataset schema + validation | `data/schema.py` | **Complete.** Frozen dataclasses, epsilon-tolerant box validation, `clipped()`, JSON round-trip |
| Roboflow ingest | `data/ingest.py` | **Complete.** Multi-split pooling, id namespacing, non-silent skip reporting |
| Manifest I/O | `data/manifest.py` | **Complete.** Atomic write via tmp+rename, streaming read, duplicate-id rejection, line-numbered errors |
| Grouped/stratified/deterministic split | `data/split.py` | **Complete and well-argued** |
| YOLO export | `data/export.py` | **Complete.** Symlinks by default (relative, so the tree survives being moved), label clipping, generated `data.yaml` |
| Dataset stats + exit criteria | `data/stats.py` | **Complete** |
| Detector training | `train.py` | **Complete.** Config-driven, unknown-key rejection, pre-flight validation, provenance with library version |
| Detector inference | `detect.py` | **Complete.** Normalized coords, lazy load, CUDA→MPS→CPU selection, batch API |
| Detection evaluation | `evaluate.py` | **Complete.** Greedy matching, size slices, failure gallery |
| Leakage audit | `dupes.py` | **Implemented, tested, and disconnected** — see Weaknesses |
| Tracking | `track.py` | **Complete for its chosen design** (greedy IoU); diverges from ROADMAP's ByteTrack, by argument |
| Multi-frame voting | `vote.py` | **Complete.** The strongest single module in the repo |
| OCR | `ocr.py` | **Complete** as a recognition-head wrapper; preprocessing measured and correctly disabled |
| Plate grammars | `plates/` | **India, Germany, Poland complete.** Extension point clean (`PlateFormat` ABC) |
| Grammar correction | `plates/correct.py` | **Complete.** Bounded search, edit penalty, confidence floor |
| CER / ablation harness | `cer.py` | **Complete** as a library; no committed driver |
| End-to-end evaluation | `endtoend.py` | **Complete** as a library; no committed driver |
| Labelling tool | `label.py` | **Complete** |
| Deduplication | `dedup.py` | **Complete.** Bounded, cooldown-refreshing |
| Excel logging | `excel.py` | **Complete.** WAL, atomic, crash-safe, file-lock-aware |
| Frame sources | `sources.py` | **Complete.** File/camera/RTSP/stills, threaded drop, reconnect |
| Viewer | `viewer.py` | **Complete.** Window + mp4, pure drawing functions unit-tested headlessly |
| Pipeline orchestration | `pipeline.py` | **Complete**, with two implementation gaps noted below |
| CLI | `cli.py` | **Partial** — 5 subcommands; several implemented capabilities unreachable |

---

# Existing Strengths

1. **Every claim in the README is backed by a number, and the negative results are reported.** Preprocessing was removed *because it was measured and it hurt*. The grammar's European failure (0.2658 → 0.2605 CER, exact-match unchanged) is stated as prominently as the Indian success. A `GRAMMAR_CORRUPTED` category exists specifically to catch the correction step damaging correct reads, and the report prints a WARNING when it fires. This is the rarest quality in an ML portfolio project and it is the repo's single biggest asset.

2. **The reasoning is committed alongside the code.** Module docstrings explain *why* — why blake2b not `hash()`, why the bare-trailing-number grouping rule was rejected, why `fliplr` was left on, why exact matches do not short-circuit the correction search, why per-character voting beats string voting. A reviewer can reconstruct the decisions without the author.

3. **Test discipline.** 518 passing tests, one module per source module, faking exactly the heavy dependencies and testing all pure logic for real. `test_notebooks.py` guarding against a Colab save bypassing pre-commit is a genuinely clever institutional-memory test.

4. **Dependency hygiene.** `ultralytics`, `torch`, `paddleocr` and `cv2` are imported inside functions everywhere. `alpr env` stays instant, and CI installs neither OCR nor GUI OpenCV.

5. **Failure modes are handled rather than ignored.** Workbook open in Excel; torn journal tail; interrupted run; RTSP drop; camera permission on macOS; Colab session death; partial Roboflow download; Ultralytics writing to a different directory than `project/name`. Each has code and a comment naming the real-world scenario.

6. **The credential story is sound.** No hardcoded fallback in `get_credential`; Colab secrets → environment → informative raise. gitleaks in pre-commit. Notebook scanning by token *shape*, never against a real key. `labels/`, `*.xlsx`, `*.mp4`, `data/**` and `.env*` all gitignored with GDPR reasoning written out.

7. **The Poland result is real engineering insight.** Recognising that Poland bans B/D/I/O/Z from the series *because they resemble digits*, and that this converts a probabilistic correction into a deterministic one, is the best idea in the project. It produced +15 points end-to-end with precision rising too — with no model change at all.

8. **Excel logging is architected, not bolted on.** The WAL design is the correct answer to `.xlsx` semantics, and `recover()` makes it operationally real.

---

# Existing Weaknesses

Ordered by impact.

### W1 — The leakage fix is written but not connected *(highest impact)*

`regroup_by_duplicates()` is the general fix for split contamination. `build_dataset()` does not call it. Verified: the only non-self references to `alpr.dupes` in the entire repo are in `tests/test_dupes.py`. Consequences:

- Every future `alpr fetch-data` / `ensure_dataset()` rebuild reproduces the *old* split logic, protected only by the filename regex.
- The README's headline leakage audit ("5.8% of test images had a near-duplicate in train") is not reproducible from anything committed.
- `clean_subset()` — the tool that produced the uncontaminated 438-image re-score — is likewise unreachable outside tests.

### W2 — Phases 4–9 have no committed driver

Notebooks exist for phases 0–3 only. The OCR ablation, the grammar-gain measurement, the end-to-end failure attribution and the leakage audit — i.e. **most of the README's numbers** — were produced in sessions that no longer exist. `alpr.cer`, `alpr.endtoend` and `alpr.dupes` are libraries with tests and no entrypoint. Anyone (including the author in three months) who wants to re-run them must reconstruct the glue.

### W3 — The pipeline reads the wrong crop, and its own docstring says so

`pipeline.py`'s module docstring states: *"the crop worth reading is the biggest and sharpest one, not whichever frame happens to come next."* The implementation reads `track.detection` — the current frame's box. `Track.best` (highest-confidence sighting) exists, is documented as "the best crop to read", and is referenced **only by `tests/test_track.py:156`**. The stated design is not the shipped behaviour.

### W4 — `--region` cannot select the Poland grammar

`cli.py` restricts `--region` to `choices=["IN", "DE"]`. The Polish grammar — which the README credits with the largest single accuracy gain in the project — is only reachable by leaving the region unpinned. Related and verified by execution: `parse(text, region=Region.EUROPE)` returns `None` for every input, because no `PlateFormat` has `region = EUROPE`. That is currently unreachable through the CLI but is a live trap for any programmatic caller, since `EUROPE` is exactly the tag the training data carries.

### W5 — Most of `PipelineConfig` is unreachable from the CLI

`alpr run` exposes `ocr_every`, `region` and `confidence`. It does not expose `min_reads`, `min_hits`, `max_age`, or `cooldown` — including the 5-minute cooldown, which `dedup.py` itself describes as "a judgement call, not a constant of nature" that "a motorway camera would want far less" of. Tuning the pipeline currently requires editing Python.

### W6 — Detector-evaluation and end-to-end matching logic are near-duplicates

`evaluate.match()` and `endtoend.best_detection_for()` both greedily pick the best detection above an IoU threshold, with the same default of 0.5 and the same `>=`-against-a-rising-floor idiom. They are not shared. This is the clearest duplication in the codebase; it is small, but it is the kind that drifts.

### W7 — `Tracker._finished` and `Track.history` grow without bound

`_retire()` and `finish()` append to `_finished`; nothing ever removes from it, and `completed()` only sets a flag. Every confirmed track — with its full `history` list of `(frame, Detection)` — is retained for the lifetime of the `Tracker`. For an offline clip this is irrelevant. For the RTSP mode the project explicitly supports (unattended, reconnecting, long-running), it is an unbounded leak, and `completed()` also re-scans the whole list on every frame. `Deduplicator` got a `max_tracked` cap for exactly this reason; `Tracker` did not.

### W8 — Workbook rewriting is O(n²) over a long run

`flush()` calls `write_workbook(path, self._events)` with **all** events, every 50. A run producing 5,000 rows rewrites the workbook 100 times, the last of which serializes 5,000 rows. The docstring identifies "saving after every plate would rewrite the entire workbook thousands of times" as the problem being solved; batching by 50 reduces the constant but not the complexity class.

### W9 — Journal resume is surprising on a fresh run

`ExcelLog.open()` reads any existing `<out>.xlsx.jsonl` and adopts its events. Running `alpr run` twice over *different* sources with the same `--out` merges both runs into one workbook. `--fresh` exists and its help text acknowledges the surprise ("A run resumes an existing log by design… That is surprising when demoing the same clip twice"), but the default is the surprising direction, and nothing records which source a journal belongs to.

### W10 — The `crop_path` column is permanently empty

`PlateEvent.crop_path` is a declared column (width 30) and `pipeline.py` never sets it. Verified in the local `demo_plates.xlsx.jsonl`: `"crop_path": null` on all 10 rows. ROADMAP Phase 7 lists it as a deliverable. A reviewer looking at a logged plate has no way to see the crop it came from.

### W11 — CI is narrower than the README claims, and single-version

- `requires-python = ">=3.11"`, local dev is 3.13, CI tests only 3.12.
- gitleaks runs only in pre-commit, never in CI. The README's "a `gitleaks` pre-commit hook and a CI test both refuse credential-shaped strings **in the repo**" overstates the CI half, which scans notebooks only.
- No coverage gate, no dependency audit.

### W12 — Grammar acceptance is looser than the confidence score implies

Verified against real output: `FBR16M0C` was logged as a Polish plate at confidence 0.687 with 1 edit, and `PolandFormat` scored the code at full confidence purely because `F` is a voivodeship letter — `FBR` itself is not validated as a real territorial code. The same pattern applies to Germany (any unlisted 1–3 letter prefix scores 0.8) and India (any unknown state code scores 0.55 but is still accepted). This is a deliberate, documented trade — closed lists would make new codes unreadable — but it means grammar confidence measures *shape*, not *registry membership*, and the pipeline's precision (76.6% measured) reflects that.

### W13 — Documentation drift

- README says "485 tests" twice; the suite is now **518**.
- README's pipeline diagram and `src/alpr/__init__.py`'s phase map both say the grammars are "India + Germany"; Poland has been implemented since PR #22 and is the best-performing of the three.
- ROADMAP Phase 6 specifies ByteTrack; `track.py` implements greedy IoU and argues the case, but ROADMAP was never updated.
- ROADMAP Phase 1's GDPR "blur-and-strip step before any upload" is not implemented anywhere.
- `.gitignore` has negations for `tests/fixtures/*.xlsx` and `tests/fixtures/*.mp4`; `tests/fixtures/` does not exist.

### W14 — Three empty placeholder packages

`ui/`, `agent/`, `mcp_servers/` each contain one comment line, are not in `pyproject.toml`'s wheel packages, and are referenced by nothing. They advertise capability the repo does not have.

### W15 — Minor implementation details

- `pipeline.py:202` — `if frame.index % config.ocr_every: continue` is loop-invariant but evaluated per track.
- `pipeline.py:214` — `timings["ocr"]` is accumulated from a `mark` that is only correct when at least one crop was read; harmless today, fragile to edit.
- `cli.py:70,94` — `ViewerError` is bound by an import *inside* the `try` block but named in the `except` tuple. If that import ever fails, the handler raises `NameError` instead of reporting the real error.
- `train.py:279` — `best_weights` falls back to "most recent `best.pt` anywhere under `project/`", which can silently return a different run's weights.
- `evaluate.py:234` — false positives are attributed to the image's widest ground-truth plate, which biases the per-bucket precision figures toward the large buckets. Documented, but it does affect the README's size table.

---

# Technical Debt

| # | Debt | Type | Cost of leaving it |
|---|---|---|---|
| D1 | `alpr.dupes` disconnected from the build path | Missing wiring | Leakage recurs silently on every rebuild |
| D2 | No drivers for phases 4–9 | Missing reproducibility | Headline results cannot be re-derived |
| D3 | `Track.best` unused; docstring describes unimplemented behaviour | Spec/impl divergence | Accuracy left on the table; docstring misleads |
| D4 | Duplicated greedy IoU matching (`evaluate` / `endtoend`) | Duplication | Two places to fix one bug |
| D5 | `Tracker._finished` / `Track.history` unbounded | Resource leak | Long RTSP runs degrade then OOM |
| D6 | Full-workbook rewrite per flush | Algorithmic | Long runs slow superlinearly |
| D7 | CLI exposes ~⅓ of `PipelineConfig`; `--region` omits PL | Incomplete interface | Implemented capability unusable |
| D8 | `parse(region=EUROPE)` silently rejects everything | Latent trap | A programmatic caller gets zero rows with no error |
| D9 | `crop_path` column declared, never populated | Dead field | Reviewers cannot audit a logged plate |
| D10 | README/ROADMAP/`__init__` drift (test count, Poland, ByteTrack, fixtures) | Doc debt | Erodes the project's main credibility asset |
| D11 | Single-Python CI; gitleaks not in CI | CI gap | 3.11/3.13 breakage and non-notebook secrets pass |
| D12 | `ui/`, `agent/`, `mcp_servers/` empty | Dead scaffolding | Advertises absent capability |
| D13 | Journal auto-resume as the default | Surprising default | Cross-run contamination of a workbook |
| D14 | `fliplr` ablation promised in two places, never run | Unfinished experiment | An open question presented as settled |
| D15 | GDPR blur-and-strip (ROADMAP Phase 1) never implemented | Missing compliance step | Blocks the dataset publication the roadmap plans |

---

# Current Bottlenecks

**Accuracy.** End-to-end is **58.1%**, and detection contributes **zero** losses across the 124 labelled plates (consistent with 0.998 recall). The entire remaining loss is in reading:

| Loss | Share | Where the fix is |
|---|---|---|
| Grammar rejected | 24.2% | Grammar coverage — more European formats |
| OCR error | 16.9% | The recognition model, or reading a better crop (`Track.best`) |
| Grammar corrupted | 0.8% | Already reduced 3→1 by the individual-plate rule |

By region: Indian 67.7% correct, European 47.5%, of which **Polish 72.2%** — Polish now outperforms Indian. The gap is entirely explained by unmodelled formats (Norwegian, Czech, Dutch and others in the pan-European set). Adding one grammar was worth +15 points end-to-end; adding a model was worth nothing, because detection loses nothing. **More detector training is provably the wrong investment.**

**Throughput.** Measured on an M4, detection on MPS, recognition on CPU: detection 34.6 fps (29 ms/frame), OCR 41.1 ms per crop. Recognition costs more than detection, which is exactly why `ocr_every` exists — full pipeline 14.3 fps at `ocr_every=1`, 23.5 fps at 3. The bottleneck is per-crop recognition, and it scales with the number of *confirmed tracks in view*, not with frame rate.

**Long-run stability.** `Tracker._finished` growth (W7) and O(n²) workbook rewriting (W8) are the two mechanisms that would degrade an unattended multi-hour RTSP deployment. Neither is visible in the current test suite, because no test runs more than 50 frames.

**Development velocity.** The absence of drivers for phases 4–9 (W2) is the practical bottleneck on *this* work: any change to OCR, grammars, or correction cannot be re-measured without first rebuilding the harness that produced the original numbers.

---

# Recommended Improvement Order

Sequenced so each step makes the next one measurable. Nothing here is implemented — this is the proposal.

**Stage 1 — Restore measurability** *(nothing else can be evaluated without this)*
1. Add committed drivers for the unmeasured phases: `notebooks/04_ocr_ablation.ipynb`, `notebooks/05_end_to_end.ipynb`, `notebooks/06_leakage_audit.ipynb`, or equivalently `alpr evaluate` / `alpr endtoend` / `alpr dupes` CLI subcommands. Prefer CLI subcommands — they are testable and the existing notebook style is already thin. Files: new drivers + `cli.py`.
2. Wire `regroup_by_duplicates()` into `build_dataset()` behind a flag, defaulting on. Files: `build.py`, `cli.py`. This closes the highest-impact gap and makes the leakage audit a permanent property of the build rather than a one-off.

**Stage 2 — Close the spec/implementation gaps** *(cheap, and each is a measurable accuracy or usability win)*
3. Read `Track.best` instead of `track.detection` in `pipeline.py`, then re-run Stage 1's end-to-end harness to confirm the docstring's claim. If it does not help, delete the claim. Files: `pipeline.py`.
4. Add `PL` to `--region` choices, and make `parse(region=...)` fail loudly (or fall back sensibly) for a region with no grammar. Files: `cli.py`, `plates/__init__.py` or `plates/correct.py`.
5. Expose `min_reads`, `min_hits`, `max_age`, `cooldown` as `alpr run` flags. Files: `cli.py`, `pipeline.py`.
6. Populate `crop_path` — write the voted track's best crop next to the workbook and record the path. Files: `pipeline.py`, `excel.py`.

**Stage 3 — Accuracy, in the order the measurements point** *(all of it downstream of reading, none of it in the detector)*
7. Add European grammars in descending dataset frequency — the 24.2% grammar-rejected bucket is the largest single loss, and Poland showed the pattern is worth +15 points. Files: new `plates/<country>.py`, `plates/__init__.py`.
8. Only then consider recognition: a plate-specific recognition model, or a second reader voted against PaddleOCR. This addresses the 16.9% OCR-error bucket and is the expensive option — do it after grammars.
9. Run the `fliplr` ablation that `configs/detector.yaml` and `train.py` both promise, and record the answer either way.

**Stage 4 — Production robustness**
10. Bound `Tracker._finished` (drain on `completed()`, or cap like `Deduplicator.max_tracked`) and cap `Track.history`. Files: `track.py`.
11. Make workbook flushing incremental, or accept the O(n²) explicitly with a documented row ceiling. Files: `excel.py`.
12. Share the greedy IoU matching between `evaluate.py` and `endtoend.py`. Files: `detect.py` (natural home), `evaluate.py`, `endtoend.py`.
13. Consider making journal resume opt-in (`--resume`) rather than default, or key the journal to its source. Files: `excel.py`, `cli.py`.

**Stage 5 — CI and documentation**
14. Matrix CI over Python 3.11/3.12/3.13; add gitleaks as a CI step. Files: `.github/workflows/ci.yml`.
15. Correct the drift: test count, Poland in the README diagram and `__init__.py` phase map, ByteTrack→greedy-IoU in ROADMAP, remove stale `.gitignore` fixture negations. Files: `README.md`, `ROADMAP.md`, `src/alpr/__init__.py`, `.gitignore`.
16. Delete `ui/`, `agent/`, `mcp_servers/` — or build one of them. Empty packages advertising absent capability are a net negative on a portfolio repo.

**Explicitly not recommended:** further detector training, a rewrite of the tracker, restructuring `src/alpr/`, or any parallel/V2 implementation. The measurements say detection loses nothing and the layering is sound.
