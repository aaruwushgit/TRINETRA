# Real Dataset Rebuild Audit

> ## ✅ CLOSED — the canonical dataset is frozen (2026-08-14)
>
> Everything below documents how we got here and is retained as the audit trail. The accepted
> result is summarised in **[§0 Canonical dataset](#0-canonical-dataset--frozen)** immediately
> below; the numbers in §3–§6 describe **intermediate builds**, not the canonical one.
>
> Two defects were found and fixed after the first corrected build:
> a degenerate-hash bridge (see [`degenerate_hash_fix.md`](degenerate_hash_fix.md)) and one
> cross-region approximate false positive.

---

# 0. Canonical dataset — FROZEN

| | Original (historical baseline) | **Canonical** |
|---|---|---|
| Location | `/Volumes/MySSD/alpr-data/` | `/Volumes/MySSD/alpr-data/canonical/` |
| Split | **2174 / 466 / 465** | **2175 / 465 / 465** |
| Images / annotations | 3,105 / 3,273 | 3,105 / 3,273 |
| EU split | 70.0 / 15.0 / 15.0 | **70.0 / 15.0 / 15.0** |
| IN split | — (ungrouped) | **70.0 / 15.0 / 15.0** |
| Groups | 3,105 (all size 1) | 1,844 |
| Duplicate edges | — | 222 |
| Cross-region duplicate edges | — | **0** |
| Degenerate duplicate edges | — | **0** |
| Manifest SHA-256 | `8fb18bbb828ebc13c735f6cdb941aeded7d9372d5f8894f6324c94f5f9f7d57e` | `1058c65db750deb42f3333aba56a3a60d606c26bd0ee793d268cdedb818aeafc` |
| Export SHA-256 | `0ba221dd0cc62c18132b541a921260c18d2ca40f0b3365a1f1230b30a233fa74` | `8e30e57d4644f6ca4a67db9088d130a79b7f0a0b0f59ad30a9a2460f00cfb983` |

**Do not confuse the two hashes.** The original is retained unmodified as the historical baseline;
it is *not* the canonical dataset.

Full machine-readable record: `/Volumes/MySSD/alpr-data/canonical/provenance.json`.

### What the correction did

- **28 leakage groups corrected** — groups whose members straddled the original train/val/test
  boundary, 25 of them touching train.
- **Clip grouping was the dominant correction**: 1,173 images (37.8 % of the dataset) against 166
  (5.3 %) for perceptual duplication. The fix that mattered was reading the Roboflow export suffix
  and the two clip conventions, not hashing.
- **The degenerate perceptual bridge was removed.** One pure black frame (thumbnail range 0, hash
  0/64) had welded a European clip, an Indian clip and 49 stills into a single 514-image group.
- **One cross-region approximate false positive was removed** — a German plate crop 5 bits from an
  Indian street scene, which had pushed Indian validation to 19.1 % against a 15 % target.
- **The final duplicate graph has 0 cross-region edges** and 0 edges touching a degenerate image.
- **Both EU and IN land at exactly 70/15/15.**

### Grouping rules in force

| | |
|---|---|
| Clip conventions | `…-mp4-t-<frame>`, `video<N>_<frame>`, `…_frame_<N>`; Roboflow `_<ext>.rf.<32-hex>` stripped first |
| Clip identity | independent of the upstream Roboflow directory |
| Same-region dHash threshold | 5 |
| Cross-region dHash threshold | 2 |
| Exact hash matches | cross regions freely |
| Degenerate images | thumbnail range ≤ 4 → exact matches only, and only with other degenerate images |
| Unknown region (`XX`) | falls back to the same-region bound; never tightens on absent evidence |

### Limitations carried forward — not hidden

- **The historical detector metrics are not comparable to this dataset.** See §9.
- **The provenance of the historical training run remains NOT PROVABLE.** See §2 — no
  `provenance.json` accompanied it, the baseline manifest's mtime postdates the weights, and
  README's val mAP@50 (0.9911) does not appear in `results.csv` (0.9871 at the last epoch).
- **`git_commit` alone does not identify the code that built the canonical dataset.** The working
  tree was dirty when it was built; `provenance.json` records `src_alpr_sha256` as the authoritative
  code fingerprint, plus the list of uncommitted paths.
- Two large same-region duplicate components (48 and 20 images) are German plate photographs of
  *different* vehicles that share an identical composition. dHash cannot read the number at 9×8, so
  they merge. Conservative for leakage, imprecise as identity. Not split — see
  [`degenerate_hash_fix.md`](degenerate_hash_fix.md) §4.

---

Audit date: 2026-08-13 (superseded by §0 above)
Source data: `/Volumes/MySSD/alpr-data/` (preserved original — **not modified**)
Rebuild output: scratchpad only. No canonical artifact was regenerated.
Detector: **not retrained.** Published metrics: **unchanged**, still the historical baseline.

Machine-readable companion: [`leakage_groups.csv`](leakage_groups.csv) — 26 rows, one per leakage group.

> **Supersedes the earlier blocked version of this document.** The original dataset was never lost;
> it was on an external SSD that my earlier disk search did not cover. No Roboflow credential was
> needed or used for this rebuild.

Every claim below is tagged **CONFIRMED**, **NOT PROVABLE**, or **NOT MEASURED**.

---

## 1. Baseline verification

**CONFIRMED — the original manifest is internally consistent with the original dataset.**

| Check | Result |
|---|---|
| Manifest records | **3,105** |
| Annotations | **3,273** |
| Images by source | EU 1,455 / IN 1,650 |
| Boxes by region | EU 1,632 / IN 1,641 |
| Images with no boxes | 9 (legitimate background images) |
| Unique `image_id` / `file_name` | 3,105 / 3,105 — no duplicates |
| Manifest images missing on disk | **0** |
| Disk images absent from manifest | **0** |
| Label files with no matching image | **0** |
| Non-label `.txt` on disk | 4 — Roboflow's own `README.roboflow.txt` / `README.dataset.txt` ×2 sources (explains 3,109 vs 3,105) |
| Metadata keys | `roboflow_split` only |
| Upstream directory mix | train 2,174 / valid 621 / test 310 |
| Original grouping | **3,105 groups, all size 1** — `group == image_id` for every record, confirming pre-fix ingest |

**CONFIRMED — the original YOLO export is internally consistent with the original manifest.**

| Check | Result |
|---|---|
| Export label counts | train **2,174** / val **466** / test **465** = 3,105 |
| Export image counts | identical, and images are **relative symlinks into `raw/`** |
| `split_records(manifest, seed=0)` vs export | **3,105 / 3,105 images match — 100.0%** |
| Same at seeds 1–5 | 52.8 %–53.7 % — so seed 0 is not a coincidence |
| Label file box counts vs manifest (300 sampled) | 0 mismatches |

The export is exactly `split_records(manifest, seed=0)`. That is a strong, non-circular result: five other seeds land near chance.

### Baseline fingerprint

```
manifest sha256      8fb18bbb828ebc13c735f6cdb941aeded7d9372d5f8894f6324c94f5f9f7d57e
manifest records     3105
source images        3105
annotations          3273
split (export)       2174 / 466 / 465
groups               3105  (all size 1)
```

Re-verified byte-for-byte after the rebuild: **unchanged**.

---

## 2. Provenance — is this the dataset behind the published metrics?

# NOT PROVABLE FROM AVAILABLE EVIDENCE

Circumstantial evidence is strong, and it is not enough.

**Supporting:**
- All **22** hyperparameters in `configs/detector.yaml` match `results/args.yaml` exactly — zero mismatches.
- `args.yaml` records `seed: 0`, the seed that reproduces the export 100 %.
- `args.yaml` `save_dir: /content/ALPR/runs/detect/runs/detect/plates` — the Colab nested-directory artifact documented in `train.py`.
- README's dataset claims (3,105 images / 3,273 plates / 2174-466-465) match the preserved artifacts exactly.

**Against, or simply absent:**
- **No `provenance.json` anywhere.** `train()` writes one recording the alpr and Ultralytics versions; it did not survive. Nothing pins the library version, and nothing records a dataset hash.
- **`manifest.jsonl` mtime is 2026-08-12 01:38 — two days *after* `best.pt` (2026-08-10 00:56).** The build is deterministic, so a later rebuild would be content-identical; but mtime cannot establish that, and no hash was recorded at training time.
- **README's val figure is not in `results.csv`.** README/`results/README.md` quote val mAP@50 **0.9911**; `results.csv` shows **0.9871** at the last epoch and **0.9881** at the best. mAP@50-95 0.8296 *does* match the last epoch exactly. The gap is consistent with a separate post-training `model.val()` call using different conf/IoU defaults — plausible, **not proven**.
- **The published test metrics have no committed artifact at all.** 0.9921 / 0.8377 / 0.9816 / 0.9917 exist only as prose in the README and model card; no CSV, JSON or log records them.

The dataset is almost certainly the right one. "Almost certainly" is not CONFIRMED, and this report will not upgrade it.

---

## 3. Corrected build

Run against `/Volumes/MySSD/alpr-data/raw/{roboflow-eu,roboflow-in}`, output to scratchpad, current implementation unmodified. Includes export-suffix handling, clip grouping, `video<N>_<M>` grouping, perceptual duplicate grouping, union of both relations, deterministic split, region stratification, `verify_split()`.

```
rebuilt manifest sha256   9aab3862e5ddc5b4114e42495d4b909d032645a46ab2cd6780e421c7ff5a6835
build time                4.8 s (duplicate audit included)
```

**Determinism: CONFIRMED.** Two independent builds produced **byte-identical** manifests and identical split assignments.

---

## 4. ORIGINAL vs REBUILT

### Dataset — nothing gained, nothing lost

| | Original | Rebuilt |
|---|---|---|
| Total images | 3,105 | **3,105** |
| Total annotations | 3,273 | **3,273** |
| EU images | 1,455 | 1,455 |
| IN images | 1,650 | 1,650 |
| Identical `image_id` sets | — | **yes** |
| Identical boxes per image | — | **yes** |

**Images removed: 0. Images created: 0. Annotation changes: 0.**

### Splits

| | train | val | test |
|---|---|---|---|
| Original | 2,174 | 466 | 465 |
| Rebuilt | **2,199** | **453** | **453** |

### Groups

| | Original | Rebuilt |
|---|---|---|
| Group count | 3,105 | **1,841** |
| Size distribution | `{1: 3105}` | `{1: 1799, 2: 24, 3: 3, 7: 1, 8: 1, 20: 2, 28: 1, 33: 1, 38: 1, 43: 1, 50: 1, 70: 1, 75: 1, 76: 1, 82: 1, 185: 1, 514: 1}` |
| Multi-image groups | 0 | 42, covering **1,306 images** |
| Clip groups | 0 | 14, covering 1,173 images |
| Duplicate-merged groups | 0 | 37 |

### Movement

| From → To | Images |
|---|---|
| train → val | 272 |
| train → test | 277 |
| val → train | 283 |
| val → test | 51 |
| test → train | 291 |
| test → val | 49 |
| **Moved** | **1,223 (39.4 %)** |
| **Unchanged** | 1,882 |

Original groups were all size 1, so group movement equals image movement at the original granularity. The meaningful group-level figure is below.

**Groups split originally, unified after rebuilding: 26, covering 1,273 images (41.0 % of the dataset).**

---

## 5. Leakage groups — CONFIRMED, 26 of them

Full machine-readable detail in [`leakage_groups.csv`](leakage_groups.csv): `group_id`, `image_count`,
`original_splits`, `original_distribution`, `rebuilt_split`, `reason`, `clip_grouping_contributed`,
`perceptual_duplicate_contributed`, `clips_involved`, `naming_conventions`.

**25 of 26 straddled train and a held-out split.** All 26 now resolve to a single split (train).

| images | reason | original distribution | conventions |
|---|---|---|---|
| 514 | BOTH | test 83 / train 347 / val 84 | mp4-t + videoN_M + still |
| 185 | BOTH | test 23 / train 132 / val 30 | videoN_M |
| 82 | BOTH | test 14 / train 56 / val 12 | mp4-t |
| 76 | CLIP_GROUPING | test 8 / train 58 / val 10 | videoN_M |
| 75 | BOTH | test 11 / train 51 / val 13 | videoN_M |
| 70 | CLIP_GROUPING | test 15 / train 46 / val 9 | videoN_M |
| 50 | CLIP_GROUPING | test 3 / train 42 / val 5 | videoN_M |
| 43 | BOTH | test 9 / train 30 / val 4 | videoN_M |
| 38 | BOTH | test 4 / train 28 / val 6 | mp4-t |
| 33 | BOTH | test 4 / train 24 / val 5 | videoN_M |
| 28 | BOTH | test 2 / train 23 / val 3 | mp4-t |
| 20 | CLIP_GROUPING | test 2 / train 13 / val 5 | videoN_M |
| 20 | PERCEPTUAL_DUPLICATE | test 1 / train 18 / val 1 | still |
| 8 | CLIP_GROUPING | test 2 / train 4 / val 2 | videoN_M |
| 7 | PERCEPTUAL_DUPLICATE | train 6 / val 1 | still |
| 3 ×2, 2 ×10 | PERCEPTUAL_DUPLICATE | train + held-out | still |

### Correction type

| Reason | Groups | Images |
|---|---|---|
| BOTH | 8 | 998 |
| CLIP_GROUPING | 5 | 224 |
| PERCEPTUAL_DUPLICATE | 13 | 51 |
| OTHER / UNKNOWN | **0** | 0 |

### Which mechanism actually does the work — ablation

| Mechanism | Leakage groups detected | Images |
|---|---|---|
| Clip grouping only | 14 | **1,173 (37.8 %)** |
| Perceptual only | 36 | 166 (5.3 %) |
| Both (shipped union) | 26 | 1,273 (41.0 %) |

**Clip/filename grouping is dominant by an order of magnitude.** Perceptual duplication finds more
*groups* but they are tiny (mostly pairs of stills); clip grouping finds the mass.

---

## 6. Defect found in the rebuilt split — one black frame merges 514 images

**This is the one result that argues against adopting the rebuild as-is.**

The largest rebuilt group holds **514 images (16.6 % of the dataset)** and spans **two source datasets and two regions**:

```
eu-dayride_type1_001   385 frames   (European)
in-video3               80 frames   (Indian)
<no clip>               49 stills
```

It is created by exactly **two duplicate edges, both at distance 5 — the threshold boundary** — and both terminating on the same image:

```
d=5   eu-train-dayride_type1_001-mp4-t-558  <->  in-train-video3_1090
d=5   eu-valid-d_license_plate_206          <->  in-train-video3_1090
```

**Root cause: `dayride_type1_001-mp4-t-558` is a pure black frame** — greyscale mean 0.0, stddev 0.00, dHash 0 (popcount 0/64). A blank image's hash is all zeros, so it sits within 5 bits of any image whose thumbnail is smooth enough to set ≤5 bits. `video3_1090` has popcount 5; `d_license_plate_206` has 6.

Scope, measured:
- Images with greyscale stddev < 5: **1** (that frame).
- Images whose hash has ≤5 bits set: **3**. None have ≥59 bits set.

Counterfactual (analysis only — no code changed): removing those two cross-source edges splits the group into **433 + 80** and changes the group count 1,841 → 1,843. Nothing else moves.

**Consequences:**
- 80 Indian images are pinned into a group claimed by the EU stratum, so they are placed by EU's ratios.
- Region stratification: EU lands at exactly **70.0 / 15.0 / 15.0**; IN at **71.5 / 14.3 / 14.3** — 1.5 pp off target. (A third bucket, `XX`, holds the 9 box-less background images at 7/1/1.)
- 16.6 % of the dataset is forced into one split, which is why train grew to 2,199 and val/test shrank to 453.

**It does not cause leakage** — over-merging is the conservative direction. It costs split balance and stratification accuracy, and it welds two unrelated clips together for no real reason.

Per the closing condition on the grouping design — *closed unless the full real dataset provides contradictory evidence* — **that condition is now met**, specifically for the perceptual stage. The clip-grouping work is unaffected and behaved exactly as designed.

---

## 7. Split verification — all 11 checks PASS

| # | Check | Result |
|---|---|---|
| 1 | No corrected group crosses train/val/test | PASS |
| 2 | `verify_split()` passes | PASS |
| 3 | Manifest re-split == exported tree | PASS |
| 4 | Region stratification within 2 pp of target | PASS |
| 5 | Rebuild produces the same manifest (byte-identical) | PASS |
| 6 | Rebuild produces the same split assignments | PASS |
| 7 | Duplicate groups intact (one split each) | PASS |
| 8 | Clip groups intact (one split each) | PASS |
| 9 | No images disappear | PASS |
| 10 | No unexpected images created | PASS |
| 11 | Annotation counts preserved | PASS |

---

## 8. Against the original YOLO export

The original export **does contain split leakage under the corrected grouping definition**: 26 groups
covering 1,273 images (41.0 %) straddled its train/val/test boundaries, 25 of them touching train.

The original export was **not modified**, **not overwritten**, and **not regenerated**.

---

## 9. Detector baseline status — unchanged

> **HISTORICAL BASELINE — ORIGINAL DATASET SPLIT**
> mAP@50 **0.9921** · mAP@50-95 **0.8377** · Precision **0.9816** · Recall **0.9917**

Not retrained, not relabelled, nothing in `results/` or `README.md` touched.

These numbers were measured on the **original** 2174/466/465 split, which this audit shows
contained images whose group-mates were in train — 83 of the 465 original test images belong to the
514-group alone.

**They were not evaluated on the canonical dataset and must not be presented as if they were.**
Any future number measured on the canonical split is a different measurement on a different
evaluation set, and the two should be published side by side, each labelled with its split.

**The provenance of the historical run remains NOT PROVABLE FROM AVAILABLE EVIDENCE** (§2): no
`provenance.json` accompanied it, the baseline manifest's mtime postdates `best.pt` by two days, the
published test metrics have no committed artifact, and README's val mAP@50 of 0.9911 does not appear
in `results.csv` (0.9871 last epoch, 0.9881 best). That limitation is preserved deliberately — it is
the reason `provenance.json` now exists for the canonical dataset.

---

## 10. Final answers

**A. Is the original manifest internally consistent with the original dataset?**
**CONFIRMED — yes.** 3,105 records ↔ 3,105 files, zero missing, zero orphans, annotations and regions all reconcile.

**B. Is the original YOLO export internally consistent with the original manifest?**
**CONFIRMED — yes.** The export is exactly `split_records(manifest, seed=0)`: 3,105/3,105 images match, against 52.8–53.7 % for five other seeds.

**C. Does the corrected pipeline eliminate every observed split-leakage group?**
**CONFIRMED — yes**, for every group observable under the corrected grouping definition. All 26 leakage groups are unified; all 11 validation checks pass; no group crosses a split. This is a guarantee for recognised clip conventions (structural) and a measured result for perceptual duplicates.

**D. How many images/groups move?**
**1,223 images move split** (272 + 277 + 283 + 51 + 291 + 49). **26 groups** that were split in the original assignment are now unified, covering **1,273 images**. 1,882 images stay put.

**E. What percentage of the dataset is affected?**
**39.4 % move split; 41.0 % sit in a group that was previously split.** 42.1 % (1,306 images) are now in a multi-image group.

**F. What is the dominant reason for movement?**
**Clip/filename grouping**, decisively: 1,173 images (37.8 %) versus 166 (5.3 %) for perceptual duplication. The fix that mattered was reading the Roboflow export suffix and the two clip conventions — not hashing.

**G. Is the corrected dataset safe to become the canonical training/evaluation dataset?**
**Not yet — one defect should be resolved first.** On leakage it is sound: every observed leakage group is eliminated, the guarantee holds, determinism and data preservation are verified. But the 514-image cross-region group is an artifact of a single blank frame, not a real relationship. It costs 1.5 pp of Indian stratification and pins 16.6 % of the dataset into one split. That is a small, well-localised fix (the degenerate-hash case), and it is much cheaper to make before a canonical export than after a retrain. My recommendation is to fix it, rebuild, then freeze.

**H. Is the original 0.9921 still directly comparable to the corrected split?**
**No.** The test split changed from 465 to 453 images and 39.4 % of the dataset moved; 83 of the original test images were group-mates of training images in the 514-group alone. The old number describes a different, leakier evaluation set. Keep it labelled as the historical baseline; do not compare it to any future number.

**I. What must happen before retraining?**
1. Decide on the degenerate-frame bridge (the reopening condition is met). Options range from excluding near-uniform images from hashing to requiring a minimum hash popcount for an edge — **not implemented, not designed here**.
2. Rebuild and re-verify (determinism, the 11 checks, region stratification).
3. Regenerate the canonical export to the SSD as a *new* artifact, leaving the original intact.
4. Record a `provenance.json` **and the manifest SHA-256** with the run, so this provenance question is never NOT PROVABLE again.
5. Retrain, then re-evaluate on the corrected test split.
6. Publish both numbers side by side, labelled by split.

---

## Not measured

- **NOT MEASURED:** whether the published detector weights were produced from this exact manifest content (see §2).
- **NOT MEASURED:** the corrected split's effect on detector accuracy — that requires retraining, which was out of scope.
- **NOT MEASURED:** whether any near-duplicate pair below the 5-bit threshold still links train and held-out. The audit reports what the threshold finds; it cannot report what it misses.
- **NOT MEASURED:** end-to-end/OCR metrics under the corrected split.

## Notes on handling

No credential was used or required; the local source data is authoritative. `.env` and `.env.example` untouched. The preserved manifest hash was re-verified after all work: **unchanged**. Three identifiers in the leakage report embed a plate number in the filename; those are redacted as `car-wbs-<PLATE-REDACTED>` in the committed CSV, consistent with the repository's treatment of plate text as personal data.
