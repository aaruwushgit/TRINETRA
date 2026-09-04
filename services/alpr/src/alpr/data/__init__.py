"""Dataset construction: ingest → manifest → split → YOLO export.

The manifest (JSONL) is the source of truth. Everything else is derived from
it and can be regenerated, which is what makes the pipeline reproducible from
a committed config rather than from whatever was in a Colab session.
"""

from alpr.data.export import CLASS_NAMES, ExportResult, export_yolo, format_label_file
from alpr.data.ingest import IngestReport, from_roboflow_export, from_yolo_dir, image_size
from alpr.data.manifest import iter_manifest, read_manifest, write_manifest
from alpr.data.schema import (
    DatasetError,
    ImageRecord,
    PlateBox,
    Region,
    Split,
    clip_id,
    strip_export_suffix,
)
from alpr.data.split import (
    DEFAULT_RATIOS,
    SplitAssignment,
    split_records,
    verify_split,
)
from alpr.data.stats import DatasetStats, check_exit_criteria, compute_stats

__all__ = [
    "CLASS_NAMES",
    "DEFAULT_RATIOS",
    "DatasetError",
    "DatasetStats",
    "ExportResult",
    "ImageRecord",
    "IngestReport",
    "PlateBox",
    "Region",
    "Split",
    "SplitAssignment",
    "check_exit_criteria",
    "clip_id",
    "compute_stats",
    "export_yolo",
    "format_label_file",
    "from_roboflow_export",
    "from_yolo_dir",
    "image_size",
    "iter_manifest",
    "read_manifest",
    "split_records",
    "strip_export_suffix",
    "verify_split",
    "write_manifest",
]
