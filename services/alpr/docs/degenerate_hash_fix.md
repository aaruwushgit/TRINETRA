# Degenerate Perceptual Hash Fix

Date: 2026-08-13
Scope: `alpr.dupes` edge generation only. Clip grouping, `split_records`, region stratification,
union-find, detector, OCR, tracker, voting and grammar are untouched. Nothing retrained.

Follows [`dataset_rebuild_audit.md`](dataset_rebuild_audit.md) §6, which found a single blank frame
welding 514 images into one group.

---

## 1. The rule

**Invariant:** a low-information image must not create approximate perceptual edges to unrelated
images.

Two changes inside `find_duplicates`, nothing else:

1. **Degenerate images are excluded from the approximate pass.** A hash that describes nothing
   cannot be evidence that two pictures match.
2. **Exact-hash pairs are formed within kind, not across it.** Two blank frames that hash alike are
   the same empty picture and still pair. A blank frame and an ordinary one do not — a left-to-right
   ramp also drives every comparison False, so it lands in the same bucket as black without being
   remotely the same picture.

`dhash()` is behaviourally unchanged; it now delegates to `_thumbnail_hash()`, which returns the hash
and the thumbnail's brightness range from **one** decode, so degeneracy costs no extra image I/O.

### Choosing the threshold — `MIN_CONTRAST = 4`

Degeneracy is measured as the **peak-to-peak brightness range of the 9×8 thumbnail**, not as hash
popcount. Measured across all 3,105 real images:

| thumbnail range | images |
|---|---|
| 0 | **1** |
| 1 – 54 | **0** |
| 55 – 255 | 3,104 |

A **54-level empty band**. Any cut inside it behaves identically on this dataset; 4 of 256 levels
(1.6 %) is the conservative end — low enough to touch nothing observed, high enough to catch a frame
that is near-flat rather than exactly flat.

**Hash popcount was rejected as the test, on evidence.** The three real images with the fewest hash
bits set are ordinary high-contrast photographs that happen to ramp left to right:

| image | popcount | thumbnail range | verdict |
|---|---|---|---|
| `dayride_type1_001-mp4-t-558` | 0 | **0** | genuinely blank |
| `pl_license_plate_205` | 4 | 176 | normal photo |
| `video3_1090` | 5 | 185 | normal photo |
| `d_license_plate_206` | 6 | 139 | normal photo |

A popcount rule would have excluded three good images from duplicate detection while catching the
one blank frame by accident. `test_a_normal_monotonic_image_keeps_its_duplicates` pins this.

---

## 2. Tests

**+14 tests (673 → 687). All pass; ruff clean.**

The regression fixture was written **first** and demonstrated failing against the old code: 7
failures, including `test_unrelated_images_do_not_merge_through_a_blank_frame` — the old
implementation did produce one connected component.

The fixture reproduces the real failure structurally: two high-contrast images (thumbnail range 238
and 239) that are 10 bits apart, each within 5 bits of a blank frame. No filename is special-cased.

Preservation coverage, all passing:

| # | Case | Behaviour |
|---|---|---|
| 1 | exact identical normal images | duplicates (distance 0) |
| 2 | near-identical normal images | duplicates |
| 3 | exact identical degenerate images | **still grouped** |
| 4 | four identical degenerate images | one cluster |
| 5 | blank vs smooth image | **not paired** |
| 6 | low-information vs unrelated | not paired |
| 7 | unrelated normal images | not paired |
| 8 | degenerate between two unrelated normals | **no component** |

Case 5 is the one that forced the rule to be refined: blank and ramp share hash 0, so excluding
degenerate images from *approximate* matching alone was not enough.

---

## 3. Duplicate graph — before vs after (real 3,105 images)

| | Before | After |
|---|---|---|
| Duplicate edges | 225 | **223** |
| Edges touching a degenerate image | 2 | **0** |
| Components (≥2 images) | 64 | 65 |
| Images in components | 226 | 225 |
| **Largest component** | **51** (50 EU + 1 IN) | **48** (all EU) |
| Degenerate images | — | 1 |

### The 514 group — measured, not assumed

The previous audit predicted it would become **433 + 80**. That prediction was **wrong**; it modelled
only the two cross-source edges, while the rule also severs the blank frame's third edge (d = 4,
EU-internal). Measured:

| | images | contents |
|---|---|---|
| group 1 | **385** | `eu-dayride_type1_001`, whole and pure |
| group 2 | **81** | `in-video3` (80) + 1 EU still |
| group 3 | **48** | EU stills |

Largest group across the whole build: **514 → 385**.

---

## 4. Graph audit — remaining components

65 components, 225 images, **zero degenerate nodes**. Everything ≥5 images or otherwise flagged:

| size | regions | sources | edges | d min–max | flag |
|---|---|---|---|---|---|
| 48 | EU | eu | 62 | 3–5 | LARGE |
| 20 | EU | eu | 26 | 2–5 | LARGE |
| 8 | EU | eu | 27 | 0–5 | — |
| 7 | EU | eu | 8 | 2–5 | — |
| 5 | EU | eu | 10 | 0–5 | — |
| 5 | IN | in | 7 | 1–5 | — |
| **2** | **EU\|IN** | **eu\|in** | **1** | **5–5** | **CROSS-REGION, CROSS-SOURCE** |

Inspected visually rather than assumed:

- **The 48 and 20 components** are isolated German plate photographs on a black background — same
  composition every time, different plates and different vehicles. At 9×8 resolution dHash sees
  "white bar on black" and cannot read the number. These are false merges in the *identity* sense but
  conservative for leakage, and separating them would require a different hash, not a different
  threshold. **Not split — the evidence does not support a change here.**
- **The cross-region pair** is `d_license_plate_206` (a German plate crop, `HRO EX 360`) against
  `video3_1090` (an Indian street scene with lens flare). They are plainly not the same picture. It
  is a single false positive sitting exactly on the 5-bit threshold.

---

## 5. Rebuild verification — all 11 checks PASS

3,105 images remain · annotations unchanged (3,273) · no images removed or created · boxes identical
per image · clip groups intact · perceptual groups intact · no corrected group crosses a split ·
`verify_split()` passes · manifest and tree agree · deterministic rebuild · identical split
assignments across two runs.

Built twice; manifests **byte-identical**. Original baseline re-hashed afterwards: **unchanged**.

---

## 6. ORIGINAL vs PREVIOUS CORRECTED vs NEW CORRECTED

| build | train | val | test | groups | multi-image | largest |
|---|---|---|---|---|---|---|
| ORIGINAL | 2174 | 466 | 465 | 3105 | 0 | 1 |
| PREVIOUS CORRECTED | 2199 | 453 | 453 | 1841 | 42 | **514** |
| **NEW CORRECTED** | 2119 | 533 | 453 | 1843 | 44 | **385** |

| movement | images |
|---|---|
| ORIGINAL → PREVIOUS | 1,223 |
| ORIGINAL → NEW | 1,263 |
| PREVIOUS → NEW | **402** |

Leakage groups measured against the original split: previous 26, new 28 — higher only because the
514 group split into three, two of which independently straddled the original assignment.

**Effect of removing the degenerate bridge:** largest group 514 → 385; cross-region contamination in
the duplicate graph eliminated; 402 images reassigned; two spurious edges removed.

---

## 7. Residual issue — NOT fixed, needs a decision

The single cross-region edge above now dominates split balance, because `split_records` claims a
cross-region group for whichever region sorts first and weights it by *that region's* image count
only. The 81-image group is 80 IN + 1 EU, so EU claims it with a weight of **1** while all 81 images
follow the placement.

| | train | val | test | EU | IN |
|---|---|---|---|---|---|
| **With the edge (shipped)** | 68.2 % | 17.2 % | 14.6 % | 70.0 / 15.0 / 15.0 | **66.6 / 19.1 / 14.3** |
| **Without it (counterfactual)** | **70.0 %** | **15.0 %** | **15.0 %** | 70.0 / 15.0 / 15.0 | **70.0 / 15.0 / 15.0** |

One false-positive edge accounts for the entire deviation. Removing it yields a textbook-perfect
split in both regions.

It is **not fixed here** because every available lever is out of scope for this task: the global
dHash threshold must not change, `split_records` and region stratification must not change, and the
image is not degenerate so the new rule correctly leaves it alone. It causes **no leakage** — the
merge is conservative — but it does cost 4.1 pp of Indian validation balance.

---

## 8. Answers

1. **Is the degenerate-hash bridge fixed?** **Yes.** Zero edges touch a degenerate image; the 514
   group is gone; the regression test fails against the old code and passes against the new.
2. **Are genuine duplicates still detected?** **Yes.** 223 of the original 225 edges survive — only
   the two degenerate ones were removed. All eight preservation cases pass, including exact
   grouping of identical blank frames.
3. **Did any suspicious large components remain?** **Yes, three, all characterised.** Two large EU
   still-components (48, 20) are same-composition plate photographs — conservative, not split. One
   cross-region pair remains and is a demonstrated false positive (§4, §7).
4. **Does clip grouping remain intact?** **Yes** — untouched, and verified: every clip group occupies
   exactly one split.
5. **Are all split invariants satisfied?** **Yes** — all 11 checks pass, including determinism and
   manifest/tree agreement.
6. **Is the resulting dataset safe to freeze as canonical?** **Not yet.** It is sound on leakage and
   integrity, but §7's single false-positive edge visibly distorts Indian stratification (19.1 % val
   against a 15 % target). That is a small, fully-diagnosed issue with a measured fix, and it is
   cheaper to resolve before freezing than after.
7. **What artifact should be frozen?** Once §7 is resolved: the rebuilt `manifest.jsonl` as the
   source of truth, plus the regenerated YOLO export, written to a **new** path on the SSD with the
   original left in place. Nothing should be frozen from this run.
8. **What SHA-256 hashes should be recorded?** For this run, for traceability — not as a freeze:

```
baseline manifest (unchanged)  8fb18bbb828ebc13c735f6cdb941aeded7d9372d5f8894f6324c94f5f9f7d57e
rebuilt manifest               b9c2fa7616c661b42c8578fe4653d8b2c8a2c3e40b8b01856402eb0c4409e126
rebuilt export (labels+split)  dd897729cc1bb4f4aacbb8683f7e252b7f11e7551d6591102d528cea33febe17
```

Whatever is eventually frozen should be recorded alongside a `provenance.json` carrying the alpr and
Ultralytics versions and the manifest hash, so the provenance gap in
[`dataset_rebuild_audit.md`](dataset_rebuild_audit.md) §2 is never repeated.

---

## Detector status — unchanged

> **HISTORICAL BASELINE — ORIGINAL DATASET SPLIT**
> mAP@50 **0.9921** · mAP@50-95 **0.8377** · Precision **0.9816** · Recall **0.9917**

Not retrained, not evaluated on any corrected split, not relabelled. Nothing in `results/` or
`README.md` was touched.
