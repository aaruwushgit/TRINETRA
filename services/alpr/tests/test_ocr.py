"""Crop extraction, preprocessing, and the PaddleOCR wrapper.

Preprocessing is tested for real — it is pure image work. The model itself is
stubbed, so CI needs no 20 MB download; a marked test exercises the real
reader for anyone who has it installed.
"""

from __future__ import annotations

import pytest
from PIL import Image

from alpr.detect import Detection
from alpr.ocr import (
    TARGET_HEIGHT,
    OcrError,
    OcrResult,
    PlateReader,
    Preprocess,
    _first_result,
    crop_plate,
    prepare,
)


def frame(width=1280, height=720, colour=(40, 45, 55)):
    return Image.new("RGB", (width, height), colour)


class TestCropPlate:
    def test_crops_the_box(self):
        detection = Detection(0.25, 0.5, 0.45, 0.6, 0.9)
        crop = crop_plate(frame(), detection, padding=0.0)
        assert crop.size == (256, 72)  # 0.20*1280, 0.10*720

    def test_padding_widens_the_crop(self):
        detection = Detection(0.25, 0.5, 0.45, 0.6, 0.9)
        tight = crop_plate(frame(), detection, padding=0.0)
        padded = crop_plate(frame(), detection, padding=0.1)
        assert padded.width > tight.width
        assert padded.height > tight.height

    def test_padding_is_clamped_to_the_frame(self):
        # A plate against the frame edge must not produce a crop outside it.
        detection = Detection(0.0, 0.0, 0.15, 0.08, 0.9)
        crop = crop_plate(frame(), detection, padding=0.5)
        assert crop.width <= 1280
        assert crop.height <= 720

    def test_degenerate_box_raises(self):
        with pytest.raises(OcrError, match="degenerate"):
            crop_plate(frame(), Detection(0.5, 0.5, 0.5001, 0.5001, 0.9), padding=0.0)


class TestPrepare:
    def test_defaults_apply_nothing(self):
        # Measured on 124 labelled crops: every step made CER worse, because
        # PaddleOCR normalizes crops itself and doing it first resamples twice.
        crop = Image.new("RGB", (100, 20), (90, 100, 110))
        out = prepare(crop, Preprocess())
        assert out.size == (100, 20)
        assert out.mode == "RGB"

    def test_upscales_when_asked(self):
        small = Image.new("RGB", (100, 20))
        settings = Preprocess(upscale=True)
        assert prepare(small, settings).height == TARGET_HEIGHT

    def test_does_not_downscale_large_crops(self):
        large = Image.new("RGB", (400, 120))
        assert prepare(large, Preprocess(upscale=True)).height == 120

    def test_upscale_preserves_aspect_ratio(self):
        crop = Image.new("RGB", (100, 20))
        out = prepare(crop, Preprocess(upscale=True))
        assert out.width == pytest.approx(100 * TARGET_HEIGHT / 20, rel=0.02)

    def test_grayscale_when_asked(self):
        out = prepare(Image.new("RGB", (200, 60)), Preprocess(grayscale=True))
        assert out.mode == "L"

    def test_autocontrast_stretches_the_range(self):
        # A washed-out plate should come back with usable contrast.
        flat = Image.new("L", (100, 60), 120)
        flat.putpixel((0, 0), 130)
        out = prepare(flat, Preprocess(autocontrast=True))
        assert out.getextrema() != (120, 130)

    def test_control_variant_changes_nothing(self):
        crop = Image.new("RGB", (200, 60), (100, 110, 120))
        out = prepare(crop, Preprocess.none())
        assert out.mode == "RGB"
        assert out.size == (200, 60)

    def test_describe_names_the_active_steps(self):
        assert Preprocess.none().describe() == "raw"
        assert Preprocess().describe() == "pad"
        described = Preprocess.all_on().describe()
        assert "upscale" in described and "gray" in described and "sharpen" in described


class TestFirstResult:
    def test_reads_dict_shape(self):
        assert _first_result([{"rec_text": "MH12AB1234", "rec_score": 0.97}]) == OcrResult(
            "MH12AB1234", 0.97
        )

    def test_reads_attribute_shape(self):
        class R:
            rec_text = "DAXY123"
            rec_score = 0.8

        assert _first_result([R()]).text == "DAXY123"

    def test_empty_results(self):
        assert _first_result([]) == OcrResult("", 0.0)
        assert not _first_result([])

    def test_unexpected_shape_degrades_quietly(self):
        # PaddleOCR's result shape has changed across major versions; an
        # unreadable score must not crash a long run.
        assert _first_result([{"rec_text": "ABC", "rec_score": "n/a"}]).confidence == 0.0

    def test_missing_keys(self):
        assert _first_result([{}]) == OcrResult("", 0.0)


class _StubModel:
    def __init__(self, text="MH12AB1234", score=0.95):
        self.text, self.score = text, score
        self.calls = 0

    def predict(self, _):
        self.calls += 1
        return [{"rec_text": self.text, "rec_score": self.score}]


class TestPlateReader:
    def test_reads_a_detection(self):
        reader = PlateReader()
        reader._model = _StubModel()
        result = reader.read(frame(), Detection(0.25, 0.5, 0.45, 0.6, 0.9))
        assert result.text == "MH12AB1234"
        assert result.confidence == 0.95

    def test_reads_several_detections(self):
        reader = PlateReader()
        reader._model = stub = _StubModel()
        results = reader.read_all(
            frame(), [Detection(0.1, 0.5, 0.2, 0.55, 0.9), Detection(0.6, 0.5, 0.7, 0.55, 0.9)]
        )
        assert len(results) == 2
        assert stub.calls == 2

    def test_model_is_not_loaded_until_used(self):
        # Building a reader must stay cheap — the ablation constructs several
        # and only some get exercised.
        assert PlateReader()._model is None

    def test_missing_paddleocr_explains_the_fix(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _fail(name, *args, **kwargs):
            if name == "paddleocr":
                raise ImportError("no paddleocr")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fail)
        with pytest.raises(OcrError, match=r"\[ocr\]"):
            _ = PlateReader().model


@pytest.mark.weights
def test_real_reader_on_a_rendered_plate(tmp_path):
    """Opt-in: needs paddleocr installed and the model downloaded."""
    from PIL import ImageDraw, ImageFont

    image = Image.new("RGB", (520, 130), (245, 245, 240))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=72)
    except TypeError:
        font = ImageFont.load_default()
    draw.text((20, 25), "MH12AB1234", fill=(15, 15, 15), font=font)
    path = tmp_path / "plate.png"
    image.save(path)

    result = PlateReader().read_path(path)
    assert result.text.startswith("MH12AB")
