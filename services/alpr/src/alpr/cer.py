"""Character error rate, and the ablation that justifies each pipeline step.

Two questions this answers, both of which the project otherwise only asserts:

**Does preprocessing help?** Each step in `alpr.ocr.Preprocess` is switchable,
so each can be measured against the control of no preprocessing at all. A step
that cannot be shown to help should not be in the pipeline.

**Do the grammars help?** This is the central claim of Phase 5 — that
correcting OCR against a plate grammar beats reading it raw. Comparing CER
before and after correction is the only way to know, and it is entirely
possible for the answer to be no: a grammar that "corrects" a correct read has
made things worse.

CER is edit distance over reference length, so a 10-character plate read with
one wrong character scores 0.1. Exact-match accuracy is reported alongside it
because for a plate log that is what actually matters — a plate with one wrong
character is a wrong plate, not a 90%-right one.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field


def levenshtein(a: str, b: str) -> int:
    """Edit distance between two strings.

    Two rows rather than a full matrix: plates are short, but this is called
    once per sample per variant per ablation, and the allocation dominates.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (char_a != char_b),  # substitution
                )
            )
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate for one pair.

    An empty reference with a non-empty hypothesis is a full error rather than
    a division by zero: the reader invented a plate where there was none.
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein(reference, hypothesis) / len(reference)


@dataclass
class CerReport:
    """Aggregate accuracy over a labelled set."""

    name: str
    samples: int = 0
    total_distance: int = 0
    total_reference_chars: int = 0
    exact_matches: int = 0
    empty_reads: int = 0
    worst: list[tuple[str, str, float]] = field(default_factory=list)

    @property
    def cer(self) -> float:
        """Corpus CER: total edits over total reference characters.

        Aggregated over the corpus rather than averaged per sample, so one
        badly-read short plate cannot dominate the score.
        """
        if not self.total_reference_chars:
            return 0.0
        return self.total_distance / self.total_reference_chars

    @property
    def accuracy(self) -> float:
        """Share of plates read exactly right — what a log actually needs."""
        return self.exact_matches / self.samples if self.samples else 0.0

    def row(self) -> str:
        return (
            f"{self.name:<28} {self.samples:>6} {self.cer:>8.4f} "
            f"{self.accuracy:>9.4f} {self.empty_reads:>7}"
        )


def score(
    pairs: Sequence[tuple[str, str]],
    name: str = "",
    *,
    keep_worst: int = 10,
) -> CerReport:
    """Score (reference, hypothesis) pairs."""
    report = CerReport(name=name)
    scored: list[tuple[str, str, float]] = []

    for reference, hypothesis in pairs:
        report.samples += 1
        report.total_distance += levenshtein(reference, hypothesis)
        report.total_reference_chars += len(reference)
        if reference == hypothesis:
            report.exact_matches += 1
        if not hypothesis:
            report.empty_reads += 1
        scored.append((reference, hypothesis, cer(reference, hypothesis)))

    scored.sort(key=lambda s: s[2], reverse=True)
    report.worst = [s for s in scored[:keep_worst] if s[2] > 0]
    return report


def header() -> str:
    return f"{'variant':<28} {'n':>6} {'CER':>8} {'accuracy':>9} {'empty':>7}\n" + "-" * 62


def compare(reports: Sequence[CerReport]) -> str:
    """Render an ablation table, best CER first."""
    lines = [header()]
    for report in sorted(reports, key=lambda r: r.cer):
        lines.append(report.row())
    return "\n".join(lines)


def ablate(
    samples: Sequence[tuple[str, str]],
    read: Callable[[str, object], str],
    variants: dict[str, object],
    *,
    keep_worst: int = 10,
) -> list[CerReport]:
    """Score each preprocessing variant over the same labelled samples.

    Args:
        samples: `(image_path, true_text)` pairs.
        read: called as `read(image_path, variant)`, returning the text.
        variants: named `Preprocess` settings, including a control.

    Returns:
        One report per variant.
    """
    reports = []
    for name, variant in variants.items():
        pairs = [(truth, read(path, variant)) for path, truth in samples]
        reports.append(score(pairs, name=name, keep_worst=keep_worst))
    return reports


def grammar_gain(
    raw_pairs: Sequence[tuple[str, str]],
    corrected_pairs: Sequence[tuple[str, str]],
) -> str:
    """Compare raw OCR against grammar-corrected OCR.

    This is Phase 5's claim under test. A grammar that lowers accuracy is a
    grammar to remove, and reporting the comparison is the only way to find
    that out rather than assume otherwise.
    """
    raw = score(raw_pairs, name="raw OCR")
    corrected = score(corrected_pairs, name="grammar-corrected")

    lines = [compare([raw, corrected]), ""]
    delta_cer = raw.cer - corrected.cer
    delta_accuracy = corrected.accuracy - raw.accuracy

    lines.append(
        f"CER {'improved' if delta_cer > 0 else 'worsened'} by {abs(delta_cer):.4f}; "
        f"exact-match accuracy {'improved' if delta_accuracy > 0 else 'worsened'} "
        f"by {abs(delta_accuracy):.4f}"
    )
    if delta_accuracy <= 0:
        lines.append(
            "The grammar is not earning its place on this data — check that "
            "corrections are not overwriting reads that were already right."
        )
    return "\n".join(lines)
