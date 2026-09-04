"""Near-duplicate detection across splits.

A test score is only meaningful if the test images are genuinely unseen. Two
things can break that without any bug in the splitting code:

1. Roboflow and similar platforms often export **augmented copies** of the
   same photograph — rotated, brightness-shifted, re-cropped — as separate
   images with unrelated filenames.
2. Public datasets get assembled from overlapping sources, so the same
   photograph can appear twice under different names.

`alpr.data.split` groups by `group_key` to stop video frames leaking, but it
cannot see either of these: with unrelated stills, every image is its own
group, so grouping is a no-op and near-duplicates split freely.

Detection is by **dHash** — a perceptual hash comparing adjacent pixel
brightness on a 9x8 greyscale thumbnail. It survives rescaling, compression
and mild brightness changes, which is exactly the family of transformations an
augmentation pipeline applies, while cryptographic hashing would see every
augmented copy as a completely different file.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

from alpr.data.schema import ImageRecord, Region, Split
from alpr.data.split import SplitAssignment

# 8x8 comparisons -> a 64-bit hash. Distances are in bits.
HASH_SIZE = 8

# Below this Hamming distance two images are treated as the same picture.
# Zero means byte-identical after thumbnailing; a handful of bits absorbs
# JPEG artefacts and small brightness shifts without matching genuinely
# different photographs of similar scenes.
DEFAULT_THRESHOLD = 5

# The same bound, tightened for two images annotated with *different* plate
# regions. Exact hash equality still crosses regions untouched — this governs
# approximate matching only.
#
# Two datasets built independently on different continents can of course share
# a photograph, and that has to keep merging when it happens. What they do not
# plausibly share is a picture that merely resembles another at 5 bits of a
# 64-bit hash: with 1,455 European and 1,650 Indian images the number of
# cross-source hash pairs rises smoothly with distance — none closer than 5,
# then 3 at 7, 38 at 10, 943 at 15 — which is the shape of coincidence, not of
# duplication. The one pair the old bound admitted was a German plate crop
# against an Indian street scene.
#
# 2 rather than 0: dHash compares neighbouring pixels, so it is invariant to
# monotonic brightness change, and a genuine re-encoded or augmented copy lands
# at 0-1 rather than reliably at 0 (measured: brightness 1, contrast 1, blur 1,
# rescale 0, JPEG 0). An exact-only rule would drop those. Every cut from 0 to 4
# produces byte-identical results on the real dataset, so this costs nothing
# observed and keeps a margin for augmented copies.
CROSS_REGION_THRESHOLD = 2

# Prefix on the split-group key written by `regroup_by_duplicates`. Exported so
# callers can recognise a merged group without matching a string literal.
DUPLICATE_GROUP_PREFIX = "dup:"


# Thumbnail brightness range at or below which an image carries no structure.
#
# dHash records only the *sign* of each adjacent-pixel comparison, never the
# magnitude, so it cannot tell "flat" from "faint". A frame whose entire 9x8
# thumbnail spans fewer levels than resampling and JPEG noise produce is not
# being described by its hash — the bits are an artefact of how ties fall. A
# pure black frame is the extreme: every comparison is False, so its hash is 0,
# and 0 sits within `DEFAULT_THRESHOLD` of any hash with few bits set.
#
# Measured on the real 3,105-image dataset: exactly one image has a range of 0,
# and the next lowest is 55. Any cut in that 54-level gap behaves identically
# here; 4 of 256 levels (1.6%) is chosen as the conservative end of it — low
# enough to touch nothing observed, high enough to still catch a frame that is
# near-flat rather than exactly flat.
#
# **Hash popcount is deliberately not the test.** The three real images whose
# hashes have the fewest bits set (4, 5 and 6) are ordinary high-contrast
# photographs that happen to ramp left-to-right; their thumbnail ranges are 176,
# 185 and 139. Treating a low popcount as degenerate would have excluded two
# perfectly good images from duplicate detection.
MIN_CONTRAST = 4


def _thumbnail_hash(path: str | Path, hash_size: int = HASH_SIZE) -> tuple[int, int]:
    """Return `(hash, thumbnail brightness range)` from a single decode.

    Both come off the same thumbnail so degeneracy costs no extra image I/O.
    """
    from PIL import Image

    with Image.open(path) as img:
        thumb = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        # tobytes() over getdata(): one byte per pixel in mode "L", and a
        # stable API — getdata() is deprecated for removal in Pillow 14.
        pixels = thumb.tobytes()

    bits = 0
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for col in range(hash_size):
            left = pixels[offset + col]
            right = pixels[offset + col + 1]
            bits = (bits << 1) | int(left > right)
    return bits, max(pixels) - min(pixels)


def dhash(path: str | Path, hash_size: int = HASH_SIZE) -> int:
    """Perceptual hash of an image, as an integer.

    Compares each pixel with its right-hand neighbour on a greyscale
    thumbnail, so the result depends on structure rather than on exact pixel
    values or file bytes.
    """
    return _thumbnail_hash(path, hash_size)[0]


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


@dataclass
class DuplicatePair:
    """Two images that appear to be the same picture."""

    left: str
    right: str
    distance: int
    left_split: Split
    right_split: Split

    @property
    def crosses_splits(self) -> bool:
        return self.left_split is not self.right_split

    @property
    def contaminates_evaluation(self) -> bool:
        """True when a held-out image has a twin the model trained on.

        This is the case that inflates a score: val and test are supposed to
        be unseen, and a twin in train means they are not.
        """
        return self.crosses_splits and Split.TRAIN in (self.left_split, self.right_split)


@dataclass
class DuplicateReport:
    images_hashed: int
    unreadable: list[str] = field(default_factory=list)
    pairs: list[DuplicatePair] = field(default_factory=list)
    threshold: int = DEFAULT_THRESHOLD
    # Images with no thumbnail contrast. Hashed and reported, but held out of
    # approximate matching; see `MIN_CONTRAST`.
    degenerate: list[str] = field(default_factory=list)

    @property
    def cross_split(self) -> list[DuplicatePair]:
        return [p for p in self.pairs if p.crosses_splits]

    @property
    def contaminating(self) -> list[DuplicatePair]:
        return [p for p in self.pairs if p.contaminates_evaluation]

    def contaminated_images(self, split: Split) -> set[str]:
        """Held-out images in `split` that have a twin in train."""
        out: set[str] = set()
        for pair in self.contaminating:
            if pair.left_split is split:
                out.add(pair.left)
            if pair.right_split is split:
                out.add(pair.right)
        return out

    def report(self, split_totals: dict[Split, int] | None = None) -> str:
        lines = [
            f"images hashed        {self.images_hashed}",
            f"threshold            {self.threshold} bits (of 64)",
            f"duplicate pairs      {len(self.pairs)}",
            f"  within one split   {len(self.pairs) - len(self.cross_split)}",
            f"  across splits      {len(self.cross_split)}",
            f"  train<->held-out   {len(self.contaminating)}",
        ]
        if self.unreadable:
            lines.append(f"unreadable           {len(self.unreadable)}")
        if self.degenerate:
            lines.append(
                f"degenerate           {len(self.degenerate)} (no contrast; exact matches only)"
            )

        for split in (Split.VAL, Split.TEST):
            affected = self.contaminated_images(split)
            line = f"{split.value} images with a twin in train: {len(affected)}"
            if split_totals and split_totals.get(split):
                line += f"  ({len(affected) / split_totals[split]:.1%} of {split_totals[split]})"
            lines.append(line)

        lines.append("")
        if not self.contaminating:
            lines.append("VERDICT: no train/held-out contamination — the scores stand.")
        else:
            lines.append(
                "VERDICT: held-out images have near-duplicates in train. Scores on "
                "those images measure memorization, not generalization."
            )
        return "\n".join(lines)


def _bound_for(left: Region, right: Region, threshold: int, cross_region: int) -> int:
    """The distance two images may sit apart and still count as duplicates.

    A pair is held to the tighter cross-region bound only when both regions are
    **known** and they differ. `Region.UNKNOWN` means the image carried no plate
    to tag — the 9 box-less background images in the real dataset — so it is
    evidence of nothing, and an absence of evidence must not tighten the rule
    and drop a duplicate that may well be genuine. Unknown therefore falls back
    to the permissive same-region bound, which is the conservative direction for
    the thing that actually matters: one picture must not reach two splits.

    No duplicate edge in the real dataset involves an UNKNOWN record, so this
    choice is currently unobservable — it is stated here rather than left to
    fall out of whatever `!=` happens to do.
    """
    if Region.UNKNOWN in (left, right) or left == right:
        return threshold
    return cross_region


def find_duplicates(
    records: Sequence[ImageRecord],
    assignment: SplitAssignment,
    image_root: str | Path,
    *,
    threshold: int = DEFAULT_THRESHOLD,
    cross_region_threshold: int = CROSS_REGION_THRESHOLD,
) -> DuplicateReport:
    """Hash every image and report pairs that look like the same picture.

    Exact matches are found by grouping identical hashes, which is linear.
    Near-matches need pairwise comparison, but only between *distinct* hashes,
    so the quadratic part shrinks by however many exact duplicates exist.

    Args:
        threshold: bits two images may differ by and still be the same picture.
        cross_region_threshold: the same, tightened for images annotated with
            different regions. Applies to approximate matching only — exact
            hash equality crosses regions regardless. Clamped to `threshold`, so
            `threshold=0` still means exact matches only.
    """
    image_root = Path(image_root)

    by_hash: dict[int, list[str]] = defaultdict(list)
    # The same buckets, minus degenerate images. Approximate matching runs over
    # these; exact matching runs over all of them.
    approximable: dict[int, list[str]] = defaultdict(list)
    split_of: dict[str, Split] = {}
    region_of: dict[str, Region] = {}
    unreadable: list[str] = []
    degenerate: list[str] = []

    for record in records:
        if not record.file_name:
            unreadable.append(record.image_id)
            continue
        path = image_root / record.file_name
        try:
            value, contrast = _thumbnail_hash(path)
        except Exception:
            unreadable.append(record.image_id)
            continue
        by_hash[value].append(record.image_id)
        if contrast <= MIN_CONTRAST:
            degenerate.append(record.image_id)
        else:
            approximable[value].append(record.image_id)
        split_of[record.image_id] = assignment.of(record)
        region_of[record.image_id] = record.primary_region

    pairs: list[DuplicatePair] = []

    def add(left: str, right: str, distance: int) -> None:
        pairs.append(
            DuplicatePair(
                left=left,
                right=right,
                distance=distance,
                left_split=split_of[left],
                right_split=split_of[right],
            )
        )

    # Identical hashes: every pair within the bucket is a duplicate — but
    # degenerate and ordinary images are paired only among their own kind.
    #
    # Two frames that are both blank *and* hash alike are the same empty
    # picture, and equality needs no similarity judgement to decide, so they
    # pair here. A blank frame and an ordinary one are a different matter: a
    # left-to-right ramp also drives every comparison False, so it lands in the
    # same bucket as black without being remotely the same picture. Hash
    # equality is only evidence when both sides had something to describe.
    degenerate_set = set(degenerate)
    for ids in by_hash.values():
        blank = sorted(i for i in ids if i in degenerate_set)
        ordinary = sorted(i for i in ids if i not in degenerate_set)
        for kind in (blank, ordinary):
            for left, right in combinations(kind, 2):
                add(left, right, 0)

    # Near matches between distinct hashes, degenerate images excluded: a hash
    # that describes nothing must not be allowed to sit near another one and
    # weld two unrelated pictures into one component.
    #
    # The bound is per *pair*, not per hash bucket, because two images sharing a
    # hash can carry different regions.
    cross_region = min(threshold, cross_region_threshold)
    if threshold > 0:
        values = sorted(approximable)
        for i, left_hash in enumerate(values):
            for right_hash in values[i + 1 :]:
                distance = hamming(left_hash, right_hash)
                if distance > threshold:
                    continue
                for left in approximable[left_hash]:
                    for right in approximable[right_hash]:
                        if distance <= _bound_for(
                            region_of[left], region_of[right], threshold, cross_region
                        ):
                            add(left, right, distance)

    return DuplicateReport(
        images_hashed=len(split_of),
        unreadable=unreadable,
        pairs=pairs,
        threshold=threshold,
        degenerate=sorted(degenerate),
    )


class _UnionFind:
    """Minimal union-find keyed by image id, with a stable representative.

    The lowest id always becomes the root, so a cluster's representative — and
    therefore the group key derived from it — is identical across runs and
    independent of the order records arrive in. That is what keeps a rebuilt
    dataset byte-for-byte reproducible.
    """

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            low, high = sorted((root_a, root_b))
            self.parent[high] = low


def duplicate_clusters(report: DuplicateReport) -> dict[str, str]:
    """Map each duplicated image to a shared cluster id.

    Union-find over the duplicate pairs, so a chain of near-matches (A~B,
    B~C) becomes one cluster even when A and C are not directly similar
    enough to pair. Splitting a chain would leak exactly as badly as
    splitting a pair.

    Returns:
        `{image_id: cluster_id}` for images in a cluster of two or more.
        Images with no duplicates are absent.
    """
    sets = _UnionFind()
    for pair in report.pairs:
        sets.union(pair.left, pair.right)

    return {image: f"{DUPLICATE_GROUP_PREFIX}{sets.find(image)}" for image in sets.parent}


def regroup_by_duplicates(
    records: Sequence[ImageRecord], report: DuplicateReport
) -> list[ImageRecord]:
    """Return records with duplicates forced into the same split group.

    This is the general fix for split leakage. Filename-based grouping only
    catches what a filename admits to; two copies of one photograph under
    unrelated names, or frames a naming convention does not mark, are
    invisible to it. Hashing sees all of them.

    **The two protections are merged, not swapped.** Writing `group` replaces
    whatever `group_key` derived from the filename, so hashing alone would
    *undo* the frame-suffix rule wherever it only partially fires: if frame 1
    of a clip happens to hash-match an unrelated photograph while frame 2 does
    not, frame 1 leaves for a `dup:` group and frame 2 keeps the clip group —
    and the clip is split, which is the exact leak both rules exist to
    prevent. So the returned grouping is the transitive closure over *both*
    relations: images hashing alike, and images already sharing a `group_key`.

    Records whose cluster contains no duplicate pair are returned unchanged,
    so an ordinary still keeps `group=None` and continues to split per image.
    """
    from dataclasses import replace

    sets = _UnionFind()

    # Relation 1: what hashing found. Seeded from the pair-level clustering so
    # chains (A~B, B~C) are already collapsed.
    members: dict[str, list[str]] = defaultdict(list)
    for image_id, cluster in duplicate_clusters(report).items():
        members[cluster].append(image_id)
    for ids in members.values():
        for other in ids[1:]:
            sets.union(ids[0], other)

    # Every image hashing tied to another. Captured before the second relation
    # is applied, so sharing a filename group with a duplicate is what pulls a
    # record in — sharing one with an ordinary record is not.
    touched_by_hashing = set(sets.parent)
    if not touched_by_hashing:
        return list(records)

    # Relation 2: what the filename already admitted to.
    by_key: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_key[record.group_key].append(record.image_id)
    for ids in by_key.values():
        for other in ids[1:]:
            sets.union(ids[0], other)

    roots = {sets.find(image_id) for image_id in touched_by_hashing}

    out: list[ImageRecord] = []
    for record in records:
        root = sets.find(record.image_id)
        if root in roots:
            out.append(replace(record, group=f"{DUPLICATE_GROUP_PREFIX}{root}"))
        else:
            out.append(record)
    return out


def clean_subset(
    records: Sequence[ImageRecord], report: DuplicateReport, split: Split
) -> list[ImageRecord]:
    """Records from `split` that have no near-duplicate in train.

    A valid uncontaminated evaluation set for a model that has *already* been
    trained: these images were genuinely unseen, so scoring them needs no
    retraining. The rest were effectively memorized.
    """
    contaminated = report.contaminated_images(split)
    return [r for r in records if r.image_id not in contaminated]
