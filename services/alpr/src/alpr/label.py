"""Hand-labelling plate text, so OCR accuracy can be measured at all.

Neither source dataset ships plate strings — only boxes. Without them, CER,
end-to-end precision/recall and the project's central claim (that
grammar-constrained correction beats raw OCR) are all unmeasurable. This
module turns roughly half an hour of typing into the labelled set that unlocks
them.

**Crops come from ground-truth boxes, not from the detector.** Reading a crop
the detector produced measures detection and recognition together, and a bad
number would not say which failed. Ground-truth crops isolate OCR: a poor
score then means poor reading, full stop. End-to-end scoring is Phase 8's job
and uses detector output deliberately.

**Sampling is stratified by plate size.** Two hundred plates drawn at random
would be mostly large, easy ones, and the resulting accuracy would flatter the
system. Sampling across size buckets means the number covers the range the
pipeline actually meets.

The output is a self-contained HTML page: crops embedded as data URIs, one
text box each, and a download button. No server, no dependencies, and it works
from `file://` — which matters because the crops are usually generated on a
Colab runtime and labelled on a laptop.
"""

from __future__ import annotations

import base64
import json
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path

from alpr.data.schema import ImageRecord
from alpr.evaluate import bucket_for

# Crops are saved with a margin: a plate cut exactly to its box loses the
# quiet space around the characters that both humans and OCR read better with.
CROP_PADDING = 0.10

# Displayed at this height in the browser. Big enough to read a 40px plate
# without squinting, which is the whole point.
DISPLAY_HEIGHT = 96


@dataclass(frozen=True)
class CropRef:
    """One plate crop awaiting a label."""

    crop_id: str
    image_id: str
    file_name: str
    width_px: int
    bucket: str

    def to_dict(self) -> dict:
        return {
            "crop_id": self.crop_id,
            "image_id": self.image_id,
            "file_name": self.file_name,
            "width_px": self.width_px,
            "bucket": self.bucket,
        }


def sample_records(
    records: Sequence[ImageRecord],
    limit: int,
    *,
    seed: int = 0,
) -> list[tuple[ImageRecord, int]]:
    """Pick `(record, box_index)` pairs, spread across plate sizes.

    Round-robins over size buckets so a run of large plates cannot crowd out
    the small ones. Deterministic for a given seed, so a labelling session can
    be resumed or extended without relabelling what is already done.
    """
    by_bucket: dict[str, list[tuple[ImageRecord, int]]] = defaultdict(list)
    for record in records:
        for index, box in enumerate(record.boxes):
            width_px = box.pixel_size(record.width, record.height)[0]
            by_bucket[bucket_for(width_px)].append((record, index))

    rng = random.Random(seed)
    for items in by_bucket.values():
        items.sort(key=lambda item: item[0].image_id)
        rng.shuffle(items)

    picked: list[tuple[ImageRecord, int]] = []
    buckets = sorted(by_bucket)
    while len(picked) < limit and any(by_bucket[b] for b in buckets):
        for bucket in buckets:
            if by_bucket[bucket] and len(picked) < limit:
                picked.append(by_bucket[bucket].pop())
    return picked


def extract_crops(
    records: Sequence[ImageRecord],
    image_root: str | Path,
    out_dir: str | Path,
    *,
    limit: int = 200,
    seed: int = 0,
    padding: float = CROP_PADDING,
) -> list[CropRef]:
    """Cut sampled plates out of their frames and write them to `out_dir`.

    Returns the crop index, also written as `index.json`.
    """
    from PIL import Image

    image_root = Path(image_root)
    out_dir = Path(out_dir)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    refs: list[CropRef] = []
    for record, box_index in sample_records(records, limit, seed=seed):
        if not record.file_name:
            continue
        source = image_root / record.file_name
        if not source.exists():
            continue

        box = record.boxes[box_index]
        with Image.open(source) as image:
            image = image.convert("RGB")
            width, height = image.size
            x1, y1, x2, y2 = box.clipped().xyxy
            pad_x = (x2 - x1) * width * padding
            pad_y = (y2 - y1) * height * padding
            crop = image.crop(
                (
                    max(0, int(x1 * width - pad_x)),
                    max(0, int(y1 * height - pad_y)),
                    min(width, int(x2 * width + pad_x)),
                    min(height, int(y2 * height + pad_y)),
                )
            )

        crop_id = f"{record.image_id}#{box_index}"
        file_name = f"{len(refs):04d}.png"
        crop.save(crops_dir / file_name)

        refs.append(
            CropRef(
                crop_id=crop_id,
                image_id=record.image_id,
                file_name=file_name,
                width_px=int(box.pixel_size(record.width, record.height)[0]),
                bucket=bucket_for(box.pixel_size(record.width, record.height)[0]),
            )
        )

    (out_dir / "index.json").write_text(
        json.dumps([r.to_dict() for r in refs], indent=2), encoding="utf-8"
    )
    return refs


def build_page(out_dir: str | Path, title: str = "ALPR — label plates") -> Path:
    """Write a self-contained labelling page next to the crops.

    Crops are embedded as data URIs so the file works from `file://` after
    being downloaded off a Colab runtime, with no server and nothing to
    install.
    """
    out_dir = Path(out_dir)
    refs = json.loads((out_dir / "index.json").read_text(encoding="utf-8"))

    rows = []
    for position, ref in enumerate(refs):
        data = base64.b64encode((out_dir / "crops" / ref["file_name"]).read_bytes()).decode()
        rows.append(
            f"""<div class="row" data-i="{position}">
  <div class="n">{position + 1}</div>
  <img src="data:image/png;base64,{data}" alt="plate crop">
  <div class="meta">{ref["width_px"]}px<br><span>{escape(ref["bucket"])}</span></div>
  <input id="i{position}" data-crop="{escape(ref["crop_id"])}"
         autocomplete="off" autocapitalize="characters" spellcheck="false"
         placeholder="type what you see, or leave blank if unreadable">
</div>"""
        )

    page = _PAGE_TEMPLATE.format(
        title=escape(title),
        count=len(refs),
        display_height=DISPLAY_HEIGHT,
        rows="\n".join(rows),
    )
    path = out_dir / "label.html"
    path.write_text(page, encoding="utf-8")
    return path


def load_labels(path: str | Path) -> dict[str, str]:
    """Read the JSON the labelling page produces: `{crop_id: text}`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "labels" in payload:
        payload = payload["labels"]
    return {str(k): str(v).strip().upper() for k, v in payload.items() if str(v).strip()}


def labelled_crops(
    out_dir: str | Path,
    labels_path: str | Path,
) -> list[tuple[Path, str]]:
    """Pair each labelled crop file with its plate text.

    The pairs feed straight into `alpr.cer.ablate` and `alpr.cer.grammar_gain`.
    """
    out_dir = Path(out_dir)
    refs = json.loads((out_dir / "index.json").read_text(encoding="utf-8"))
    labels = load_labels(labels_path)

    pairs = []
    for ref in refs:
        text = labels.get(ref["crop_id"])
        if text:
            pairs.append((out_dir / "crops" / ref["file_name"], text))
    return pairs


_PAGE_TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>{title}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.5 system-ui, sans-serif; margin: 0; padding: 1rem 1rem 6rem; }}
  header {{ position: sticky; top: 0; background: Canvas; padding: .75rem 0;
            border-bottom: 1px solid color-mix(in srgb, CanvasText 20%, transparent);
            display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; }}
  h1 {{ font-size: 1rem; margin: 0; }}
  .row {{ display: grid; grid-template-columns: 3rem auto 5rem 1fr; gap: 1rem;
          align-items: center; padding: .5rem 0;
          border-bottom: 1px solid color-mix(in srgb, CanvasText 10%, transparent); }}
  .n {{ opacity: .5; font-variant-numeric: tabular-nums; }}
  img {{ height: {display_height}px; image-rendering: pixelated;
         background: #fff; border-radius: 4px; }}
  .meta {{ font-size: .8rem; opacity: .6; }}
  input {{ font: 600 1rem ui-monospace, monospace; padding: .5rem .6rem;
           text-transform: uppercase; width: 100%; box-sizing: border-box;
           border: 1px solid color-mix(in srgb, CanvasText 30%, transparent);
           border-radius: 6px; background: Canvas; color: CanvasText; }}
  input:focus {{ outline: 2px solid Highlight; }}
  .done input {{ border-color: seagreen; }}
  button {{ font: inherit; padding: .5rem .9rem; border-radius: 6px; cursor: pointer;
            border: 1px solid color-mix(in srgb, CanvasText 30%, transparent);
            background: Canvas; color: CanvasText; }}
  button.primary {{ background: seagreen; color: #fff; border-color: seagreen; }}
  #progress {{ font-variant-numeric: tabular-nums; }}
</style>

<header>
  <h1>Label plates</h1>
  <span id="progress">0 / {count}</span>
  <button class="primary" onclick="save()">Download labels.json</button>
  <span style="opacity:.6">Enter = next · blank = unreadable · autosaves in this browser</span>
</header>

{rows}

<script>
  const KEY = "alpr-labels";
  const inputs = [...document.querySelectorAll("input")];

  // Restore anything typed before a refresh: losing a labelling session to a
  // stray reload would be the fastest way to make nobody finish one.
  const saved = JSON.parse(localStorage.getItem(KEY) || "{{}}");
  inputs.forEach(i => {{ if (saved[i.dataset.crop]) i.value = saved[i.dataset.crop]; }});

  function store() {{
    const out = {{}};
    inputs.forEach(i => {{
      const v = i.value.trim();
      if (v) out[i.dataset.crop] = v.toUpperCase();
    }});
    localStorage.setItem(KEY, JSON.stringify(out));
    document.getElementById("progress").textContent =
      Object.keys(out).length + " / " + inputs.length;
    inputs.forEach(i => i.parentElement.classList.toggle("done", !!i.value.trim()));
    return out;
  }}

  inputs.forEach((input, index) => {{
    input.addEventListener("input", store);
    input.addEventListener("keydown", event => {{
      if (event.key === "Enter") {{
        event.preventDefault();
        const next = inputs[index + 1];
        if (next) {{ next.focus(); next.scrollIntoView({{block: "center"}}); }}
      }}
    }});
  }});

  function save() {{
    const blob = new Blob([JSON.stringify(store(), null, 2)], {{type: "application/json"}});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "labels.json";
    a.click();
  }}

  store();
  if (inputs.length) inputs[0].focus();
</script>
"""
