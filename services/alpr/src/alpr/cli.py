"""Command-line entrypoint.

Subcommands land as their phases do. `env` works today because diagnosing a
wrong Colab runtime is the first thing that goes wrong in this project.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from alpr import __version__

# Cheap: alpr.dupes imports nothing heavier than stdlib at module scope (PIL is
# imported inside dhash), so `alpr env` still pays nothing for this.
from alpr.dupes import DEFAULT_THRESHOLD
from alpr.env import detect_gpus, in_colab


def _cmd_env(_: argparse.Namespace) -> int:
    print(f"alpr {__version__}")
    print(f"python  {sys.version.split()[0]}")
    print(f"colab   {in_colab()}")

    gpus = detect_gpus()
    if not gpus:
        print("gpu     none visible (CPU-only runtime)")
    else:
        for i, gpu in enumerate(gpus):
            print(f"gpu[{i}]  {gpu}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    # Imported here so `alpr env` does not pay for loading torch.
    from alpr.train import TrainConfig, TrainingError, train

    try:
        config = TrainConfig.from_yaml(args.config)
        train(config, args.data, require_gpu=not args.allow_cpu)
    except TrainingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from pathlib import Path

    from alpr.data.schema import Region
    from alpr.detect import PlateDetector
    from alpr.ocr import PlateReader
    from alpr.pipeline import Pipeline, PipelineConfig
    from alpr.sources import SourceError, open_source
    from alpr.viewer import Viewer, ViewerError, annotate

    if args.fresh:
        # A run resumes an existing log by design — the journal is what makes
        # an interrupted run recoverable. That is surprising when demoing the
        # same clip twice, so this starts clean on request.
        out = Path(args.out)
        out.unlink(missing_ok=True)
        out.with_suffix(out.suffix + ".jsonl").unlink(missing_ok=True)

    try:
        detector = PlateDetector(args.weights, device=args.device)
        config = PipelineConfig(
            ocr_every=args.ocr_every,
            region=Region(args.region) if args.region else None,
            confidence=args.confidence,
        )
        pipeline = Pipeline(detector, PlateReader(), config)

        with open_source(args.source) as source:
            fps = getattr(source, "fps", 0) or 20.0
            viewer = Viewer(show=args.show, save_path=args.save_video, fps=fps)

            def draw(frame, detections, tracks, texts):
                if not viewer.active:
                    return True
                hud = [f"frame {frame.index}", f"tracks {len(tracks)}"]
                annotate(frame.image, detections, tracks, texts, hud)
                return viewer.push(frame.image)

            try:
                stats = pipeline.run(
                    source,
                    args.out,
                    max_frames=args.max_frames,
                    on_frame=draw if viewer.active else None,
                )
            finally:
                viewer.close()
            if viewer.save_path:
                print(f"annotated video: {viewer.save_path}\n")
    except (SourceError, ViewerError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(stats.report())
    return 0


def _cmd_label(args: argparse.Namespace) -> int:
    from alpr.data import Split, read_manifest, split_records
    from alpr.label import build_page, extract_crops

    records = read_manifest(args.manifest)
    if args.split:
        assignment = split_records(records, seed=args.seed)
        records = assignment.partition(records)[Split(args.split)]

    refs = extract_crops(records, args.images, args.out, limit=args.limit, seed=args.seed)
    if not refs:
        print("error: no crops extracted — check --images points at the frames", file=sys.stderr)
        return 1

    page = build_page(args.out)
    buckets: dict[str, int] = {}
    for ref in refs:
        buckets[ref.bucket] = buckets.get(ref.bucket, 0) + 1

    print(f"{len(refs)} crops -> {args.out}")
    for bucket, count in sorted(buckets.items()):
        print(f"  {bucket:<20} {count}")
    print(f"\nopen {page} in a browser, type the plates, then Download labels.json")
    return 0


def _cmd_fetch_data(args: argparse.Namespace) -> int:
    from alpr.build import BuildError, ensure_dataset
    from alpr.env import MissingCredential

    def show_duplicates(report) -> None:
        # The audit describes the split filename grouping alone would have
        # produced; the build has already merged what it found, so this is the
        # leakage that was prevented rather than leakage that remains.
        print("--- duplicate audit (before regrouping) ---")
        print(report.report())
        print("-------------------------------------------\n")

    try:
        data_yaml = ensure_dataset(
            raw_dir=args.raw_dir,
            out_dir=args.out_dir,
            manifest_path=args.manifest,
            seed=args.seed,
            force=args.force,
            check_duplicates=not args.no_duplicate_check,
            duplicate_threshold=args.duplicate_threshold,
            on_duplicates=show_duplicates,
        )
    except MissingCredential as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    from alpr.data import read_manifest, split_records
    from alpr.data.stats import compute_stats

    records = read_manifest(args.manifest)
    stats = compute_stats(records, split_records(records, seed=args.seed))
    print(stats.report())
    print(f"\ndata.yaml: {data_yaml}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alpr", description=__doc__)
    parser.add_argument("--version", action="version", version=f"alpr {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    env = sub.add_parser("env", help="report interpreter, Colab, and GPU status")
    env.set_defaults(func=_cmd_env)

    train_cmd = sub.add_parser("train", help="train the plate detector")
    train_cmd.add_argument("--config", default="configs/detector.yaml", help="training config")
    train_cmd.add_argument("--data", default="data/yolo/data.yaml", help="dataset config")
    train_cmd.add_argument(
        "--allow-cpu",
        action="store_true",
        help="do not require a CUDA GPU (a CPU run will not finish; for smoke tests only)",
    )
    train_cmd.set_defaults(func=_cmd_train)

    run_cmd = sub.add_parser("run", help="detect, read and log plates from a source")
    run_cmd.add_argument(
        "--source",
        required=True,
        help=(
            "video path, image, folder of images, camera index (0), or rtsp:// url. "
            "Images are treated as unrelated stills: each is scored on its own, "
            "without the multi-frame voting that video gets"
        ),
    )
    run_cmd.add_argument("--out", default="plates.xlsx", help="Excel log to write")
    run_cmd.add_argument("--weights", default="best.pt", help="trained detector weights")
    run_cmd.add_argument(
        "--device", default=None, help="cuda index, 'mps' or 'cpu' (auto-detected)"
    )
    run_cmd.add_argument(
        "--region",
        choices=["IN", "DE"],
        default=None,
        help="restrict plate parsing to one country's grammar",
    )
    run_cmd.add_argument(
        "--ocr-every",
        type=int,
        default=3,
        dest="ocr_every",
        help="read each track once every N frames (1 is most accurate, slowest)",
    )
    run_cmd.add_argument("--confidence", type=float, default=0.25)
    run_cmd.add_argument("--max-frames", type=int, default=None, dest="max_frames")
    run_cmd.add_argument(
        "--show",
        action="store_true",
        help="open a live window with boxes and plate text (needs a GUI OpenCV build)",
    )
    run_cmd.add_argument(
        "--fresh",
        action="store_true",
        help="delete an existing log first instead of appending to it",
    )
    run_cmd.add_argument(
        "--save-video",
        default=None,
        dest="save_video",
        help="write an annotated mp4; works without a display and is shareable",
    )
    run_cmd.set_defaults(func=_cmd_run)

    label_cmd = sub.add_parser(
        "label", help="extract plate crops and build a page for hand-labelling"
    )
    label_cmd.add_argument("--manifest", default="data/manifest.jsonl")
    label_cmd.add_argument("--images", default="data/raw", help="frame root")
    label_cmd.add_argument("--out", default="labels", help="where crops and the page go")
    label_cmd.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="label crops from this split (test is what CER should be measured on)",
    )
    label_cmd.add_argument("--limit", type=int, default=200)
    label_cmd.add_argument("--seed", type=int, default=0)
    label_cmd.set_defaults(func=_cmd_label)

    fetch_cmd = sub.add_parser(
        "fetch-data", help="download the source datasets and build the split"
    )
    fetch_cmd.add_argument(
        "--raw-dir",
        default="data/raw",
        dest="raw_dir",
        help="where the downloaded frames go (an external disk is fine)",
    )
    fetch_cmd.add_argument("--out-dir", default="data/yolo", dest="out_dir")
    fetch_cmd.add_argument("--manifest", default="data/manifest.jsonl")
    fetch_cmd.add_argument("--seed", type=int, default=0)
    fetch_cmd.add_argument("--force", action="store_true", help="rebuild even if the export exists")
    fetch_cmd.add_argument(
        "--no-duplicate-check",
        action="store_true",
        dest="no_duplicate_check",
        help=(
            "skip the perceptual near-duplicate audit. Faster on a very large "
            "dataset, and leak-prone: augmented copies of one photograph then "
            "split freely between train and test"
        ),
    )
    fetch_cmd.add_argument(
        "--duplicate-threshold",
        type=int,
        default=DEFAULT_THRESHOLD,
        dest="duplicate_threshold",
        help=(
            f"Hamming distance (of 64) below which two images are the same "
            f"picture (default {DEFAULT_THRESHOLD}; 0 = exact matches only)"
        ),
    )
    fetch_cmd.set_defaults(func=_cmd_fetch_data)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
