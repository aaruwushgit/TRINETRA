# Clip-Level Split Integrity — Implementation Report

Date: 2026-08-12
Scope: dataset grouping and leakage only. Detector, OCR, tracker, voting, grammar and the
inference pipeline are untouched. Nothing was retrained.

Follows [`dataset_rebuild_audit.md`](dataset_rebuild_audit.md), which established that the previous
grouping did **not** guarantee clip-level split integrity on the real dataset.

---

## 1. Files changed

| File | Change |
|---|---|
| `src/alpr/data/schema.py` | Added `strip_export_suffix()` and `clip_id()`; replaced `_FRAME_SUFFIX` with `_EXPORT_SUFFIX` + `_CLIP_PATTERNS`; `group_key` now strips the export suffix before reading the id as a filename |
| `src/alpr/data/ingest.py` | `from_roboflow_export` gained `group_by_clip` (default True, replaces `group_by_stem`); new `_group_for()` assigns each record its group; frames get `<source prefix><clip>` with the upstream directory dropped, stills keep a per-image group |
| `src/alpr/data/__init__.py` | Re-export `clip_id`, `strip_export_suffix` |
| `.gitignore` | Added `!.env.example` so the template is tracked and therefore scanned by gitleaks |
| `.env.example` | Rewritten with placeholders only |

Unchanged and deliberately so: `alpr/dupes.py` (perceptual grouping — already sound),
`alpr/data/split.py`, `alpr/build.py`, `alpr/data/manifest.py`.

## 2. Files added

| File | Purpose |
|---|---|
| `tests/test_grouping.py` | The invariant chain and the 13 adversarial scenarios |
| `docs/grouping_fix.md` | This report |

## 3. Files removed

None.

## 4. Tests added

**66 new tests** (607 → 673).

| Module | Added | Covers |
|---|---|---|
| `tests/test_data_schema.py` | 32 | `strip_export_suffix` (3 real extension flavours + 6 near-misses), `clip_id` (10 positive, 14 negative) |
| `tests/test_data_ingest.py` | 6 | Ingest grouping across upstream directories, using verbatim real filenames |
| `tests/test_grouping.py` | 25 | Invariant chain (steps 1–4), transitivity, 13 adversarial scenarios |
| `tests/test_build.py` | 3 | Adjusted for the new explicit still grouping |

Fixtures are **verbatim real filenames** from `labels/index.json`, not synthetic stand-ins —
synthetic names would not have caught this bug, because the bug *is* the real naming.

## 5. Tests passed

```
673 passed          ruff check: clean          ruff format: clean
```

The suite was mutation-tested to confirm it bites. Four mutants, all caught:

| Mutant | Caught by |
|---|---|
| Stop stripping the Roboflow export suffix | `test_two_clips_do_not_merge` (+5 others) |
| Put the upstream directory back in the clip group | `test_video_n_frames_across_directories_share_a_group` |
| Drop the `group_key` relation from the union-find | `test_filename_and_perceptual_grouping_interact` |
| Drop the `video<N>_<M>` rule | `test_video_n_frames_across_directories_share_a_group` |

## 6. Grouping invariants

### What matches

**Export suffix** — `_EXPORT_SUFFIX` removes `_<ext>.rf.<32-hex>`, requiring a specific image
extension, the literal `.rf.`, exactly 32 hex characters, at end of string. Rejects
`photo.rf.deadbeef` (short), `_txt.rf.…` (not an image), a 33-character tail, and any occurrence
not at the end.

**Clip identity** — three patterns, each anchored on an explicit marker, each capturing the clip:

| Convention | Example | Clip |
|---|---|---|
| Roboflow video export | `dayride_type1_001-mp4-t-451` | `dayride_type1_001` |
| Indian source | `video3_2190` | `video3` |
| Classic | `clip_042_frame_0137` | `clip_042` |

`video<N>_<M>` is scoped tightly and documented at the definition. It matches `video3_2190`,
`video11_1870`, `in-train-video8_870`. It does **not** match `video_2190` (no clip number),
`myvideo3_2190` (`\b` — `video` must start a word), `video3` (no frame number), `video3_2190_640`
(frame number must end the string), or `videos_12`. There is deliberately no bare trailing-number
rule: it would collapse the real stills `pl_license_plate_205` and `pl_license_plate_242`.

### The guarantee, and why it holds

> **Two frames of one recognised clip cannot be assigned to different splits.**

Four steps. `tests/test_grouping.py::TestInvariant` asserts each individually, so a broken link
names itself.

**Step 1 — clip identity does not depend on the upstream directory.**
`clip_id(stem)` is a pure function of the stem. The Roboflow directory is not an input, so it
cannot be an output.

**Step 2 — ingest gives every frame of a clip the same `group`.**
`_group_for()` builds `f"{id_prefix}{clip}"` — source prefix plus clip, directory dropped. Two
frames of one clip therefore receive byte-identical `group` strings regardless of which directory
they were exported into. `image_id` keeps the directory and stays unique, so nothing else changes.

**Step 3 — regrouping only ever merges.**
`regroup_by_duplicates` rewrites `group`, so it could in principle pull a clip apart. It cannot:
its union-find closes over the existing `group_key` relation *as well as* the hash pairs, so a
connected component is either rewritten whole or left alone. Formally: if `group_key(a) ==
group_key(b)` before, then after regrouping `group_key(a) == group_key(b)`. Merge-only.

**Step 4 — one group key is one split.**
`split_records` builds `assignment: dict[group_key, Split]`, and `claimed` ensures each key is
processed by exactly one region. `SplitAssignment.of()` is a lookup on `group_key`. The mapping is
single-valued by construction, so equal keys ⇒ equal split — for any ratios, any seed, any dataset.

Chaining: steps 1+2 put a clip in one group, step 3 keeps it there, step 4 sends one group to one
split. **Same clip → same `group_key` → same split.** This is structural, not statistical; no seed
sweep is load-bearing.

### What the guarantee does *not* cover

It is a guarantee over **recognised** clip identities. A clip whose frames Roboflow renamed into a
convention none of the three patterns describes is invisible to it, and falls back to perceptual
hashing — a probabilistic backstop, not a guarantee. Adding a convention is a one-line addition to
`_CLIP_PATTERNS` plus tests.

## 7. Adversarial test results

All 13 required scenarios, each run across 5 seeds unless noted, asserting **no same-clip
cross-split leakage** and — where the metadata permits the distinction — that unrelated images were
not merged.

| # | Scenario | Result |
|---|---|---|
| 1 | Clip entirely in train | pass |
| 2 | Clip entirely in test | pass |
| 3 | Clip split across train/test | pass |
| 4 | Clip split across validation/test | pass |
| 5 | Multiple clips with similar names | pass — 5 lookalike names stayed 5 groups (`dayride_type1_001` / `_002` / `_0010`, `video1` / `video11`) |
| 6 | Identical stems, genuinely unrelated files | pass — **not merged** |
| 7 | Roboflow suffixes (`jpg`/`jpeg`/`png`) | pass — extension flavour does not change clip identity |
| 8 | `video<N>_<M>` | pass — `video3` and `video11` stayed distinct |
| 9 | Perceptual duplicates without shared filenames | pass — merged, as intended |
| 10 | Filename + perceptual grouping interacting | pass — clip and twin closed into one group |
| 11 | Multiple upstream directories (`train`/`valid`/`val`/`test`) | pass |
| 12 | Different split ratios (5 ratio sets, incl. 90/5/5 and 20/40/40) | pass |
| 13 | Different seeds (50 seeds) | pass |
| — | The Stage-2 counterexample (clip half > 55% of its region) | **pass — 0/25 seeds leak, was 25/25** |

## 8. Collision analysis

Measured over all **188 surviving real identifiers** (`labels/index.json` ∪ `labels.json`) — every
identifier that still exists from the real dataset.

| Measure | Value |
|---|---|
| Identifiers analysed | 188 |
| Frames recognised as video | 94 (50%) |
| Candidate clip groups | 12 |
| **Clip groups spanning >1 upstream directory** | **9 of 12** |
| Non-video stems appearing in >1 upstream directory | **0** |
| Non-video stems appearing more than once | **0** |
| Clip groups containing two frames with the same stem | **0** |
| Potential false merges | **0** |
| Known real clip splits | 9 clips, 89 frames affected |
| Groups affected by dropping the directory from clip identity | 12 |

Clips spanning multiple upstream directories:

```
eu-dayride_type1_001    43 frames   test, train, valid
in-video3               10 frames   test, train, valid
eu-nightride_type3_001   9 frames   train, valid
in-video11               8 frames   train, valid
in-video8                6 frames   train, valid
in-video4                5 frames   test, train, valid
in-video2                4 frames   train, valid
eu-dayride_type1_003     2 frames   train, valid
in-video9                2 frames   train, valid
```

**Zero measured collision risk**, and the design bounds it further: only filenames that explicitly
announce themselves as video frames get the directory dropped. Stills keep a per-image group, so
the false-merge surface excludes them entirely — which is why `test_identical_stems_but_unrelated_files_do_not_merge`
passes.

**Sampling caveat.** These 188 are the crops sampled for hand-labelling, all drawn from the old
test split (~465 images) — roughly 6% of the 3,105-image dataset. Collision risk on the other
~2,900 images is **NOT MEASURABLE FROM AVAILABLE DATA**.

## 9. What can be proven from the surviving data

- Every surviving real identifier carries the `_<ext>.rf.<32-hex>` suffix (188/188) — so the old
  anchored frame rule matched **0/188**, and filename grouping had never once fired on real data.
- Both real clip conventions are now recognised: `…-mp4-t-N` and `video<N>_<M>`.
- The 188 identifiers produce 106 groups, 11 of them multi-frame clips, 9 spanning upstream
  directories.
- Across **400 configurations** (4 ratio sets × 100 seeds) on the real identifiers: **0 clip
  leaks**.
- The guarantee itself is proven structurally (§6), not by sampling.
- The build remains deterministic, `verify_split` still runs on every build, the manifest remains
  the source of truth, and the exported tree still agrees with a re-split of the manifest.

## 10. What cannot be proven — original dataset unavailable

- **Any before/after comparison.** The original manifest does not exist: `data/manifest.jsonl` is
  0 bytes, `data/**` is gitignored, the HF dataset repo returns 401. It is not recoverable, only
  reconstructible — which needs the source images.
- **Real image counts, annotation counts, split sizes, region distribution** after a rebuild.
- **How many images actually move split**, and the real duplicate-group count and size
  distribution.
- **Collision risk on the ~94% of the dataset with no surviving identifier.**
- **Whether contamination is actually eliminated** on the real data — the 5.8% figure in the README
  cannot be recomputed.
- **Whether the published detector metrics would change**, since the split they used cannot be
  reconstructed.

## 11. Is the implementation now safe by construction?

**For recognised clip conventions: yes.** The guarantee in §6 is structural — equal clip ⇒ equal
`group_key` ⇒ equal split, for any ratios and any seed. The Stage-2 counterexample that leaked at
every seed now leaks at none, and it does so because there is no longer anything to place apart,
not because placement got luckier.

**For the dataset as a whole: not yet demonstrated**, and that gap is data availability, not
design. The implementation is sound; it has not been run against the real 3,105 images.

Two residual risks are inherent rather than fixable here:

- A clip in a **fourth naming convention** would be invisible to the guarantee and fall back to
  hashing. Nothing in the surviving sample suggests one exists, but 94% of the dataset is unseen.
- **Perceptual grouping remains probabilistic**, by nature. It is the backstop, not the guarantee.

## 12. External data still required

1. **A valid `ROBOFLOW_API_KEY`** with read access to `e-hh49k/european-license-plates-tjviy` v1 and
   `nivu/indian-license-plate-knte7` v1. This is the only blocker.
2. Optionally, confirmation that those pinned versions have not been re-exported upstream since the
   original build — otherwise even a reconstructed baseline is approximate.

With (1), the rebuild is one command (`alpr fetch-data --force`) and the audit numbers in
`dataset_rebuild_audit.md` §2 can be filled in.

## 13. Security changes made

**Root cause.** `.gitignore` line 47 was `.env.*`, which matched `.env.example`. That file's entire
purpose is to be committed, so ignoring it had two effects: developers cloning the repo got no
template, and — because pre-commit hooks run on staged files — **gitleaks had never scanned it**.
Unscanned and unshared, it accumulated values byte-identical to `.env`.

**Fixed.**

- `.env.example` rewritten with obvious placeholders. Verified: no credential-shaped string
  remains, and no value hashes equal to its `.env` counterpart.
- `.gitignore` gained `!.env.example` after `.env.*`. Verified by `git add --dry-run`:
  `.env.example` is now addable, while `.env` and `.env.local` are still refused.
- The file is now tracked-eligible, so gitleaks will scan it on every future commit.

**Not done, and outside this repository's reach:**

> **The credentials require rotation.** They have **not** been rotated. The Roboflow key returns
> `401 — This API key does not exist (or has been revoked)`, which indicates it is already dead, but
> that is Roboflow's state, not an action taken here. The Kaggle token's status is **unknown and
> untested**. Both should be rotated in their respective provider dashboards.

No credential value was printed, copied into any file, committed, or included in any report, test,
diff or log. Comparisons were done on truncated SHA-256 hashes.

## 14. Remaining risks

| Risk | Severity | Note |
|---|---|---|
| Real rebuild never run | **high** | Everything in §10 stays unproven until credentials exist |
| Collision risk unmeasured on ~94% of the dataset | medium | 0 collisions in the 6% sample; design bounds the surface to video-named files |
| A fourth, unrecognised clip convention | medium | Would fall back to hashing; not visible in the sample |
| Perceptual grouping is probabilistic | low, inherent | Backstop by design, not the guarantee |
| Published detector metrics measured on an unreconstructable split | medium | See below |
| Kaggle token rotation status unknown | medium | Requires action outside this repo |

**Detector metrics status (unchanged, per instruction):**

> **HISTORICAL BASELINE — ORIGINAL DATASET SPLIT.**
> mAP@50 0.9921 / recall 0.9917 were measured on a split that (a) no longer exists as an artifact
> and (b) is now known to contain clips straddling train and test — `dayride_type1_001` and
> `video3` are both visible doing so in the surviving labelled sample. Nothing in `results/` or
> `README.md` was changed. These numbers must not be relabelled as results from the corrected split
> until a model is retrained on it.
