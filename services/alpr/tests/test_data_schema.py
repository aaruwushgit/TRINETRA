"""Record validation and serialization."""

from __future__ import annotations

import pytest

from alpr.data import DatasetError, ImageRecord, PlateBox, Region, clip_id, strip_export_suffix


class TestStripExportSuffix:
    """Roboflow renames every export to `<original>_<ext>.rf.<32-hex>`.

    All 188 surviving real identifiers carry it, in three extension flavours.
    The frame patterns anchor at the end of the string, so leaving it attached
    is what made the frame rule match nothing on the real dataset.
    """

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            # The three extensions the real ids actually use: jpg (173),
            # jpeg (13), png (2).
            (
                "dayride_type1_001-mp4-t-751_jpg.rf.5c6687a87b9a7a5a3cff5183d1ab1730",
                "dayride_type1_001-mp4-t-751",
            ),
            (
                "car-wbs-KL54H369_00000_jpeg.rf.ec38c1bf7ae1c0a7d5a550a0c2e59a47",
                "car-wbs-KL54H369_00000",
            ),
            (
                "video3_2190_png.rf.f9a4bf94d32d92ec52483139a3029050",
                "video3_2190",
            ),
        ],
    )
    def test_strips_the_real_suffix(self, name, expected):
        assert strip_export_suffix(name) == expected

    @pytest.mark.parametrize(
        "name",
        [
            "plain_photo",
            "photo.rf.deadbeef",  # too short to be an md5
            "photo_jpg.rf.notallhexdigitsxxxxxxxxxxxxxxxxx",  # not hex
            "photo_txt.rf.5c6687a87b9a7a5a3cff5183d1ab1730",  # not an image ext
            "photo_jpg.rf.5c6687a87b9a7a5a3cff5183d1ab1730x",  # 33 chars
            "prefix_jpg.rf.5c6687a87b9a7a5a3cff5183d1ab1730_suffix",  # not at end
        ],
    )
    def test_leaves_everything_else_alone(self, name):
        assert strip_export_suffix(name) == name


class TestClipId:
    @pytest.mark.parametrize(
        ("name", "clip"),
        [
            # --- Roboflow video export, real ids ---
            (
                "dayride_type1_001-mp4-t-451_jpg.rf.570aecd2ac67cc8efe361d014d8b4a0f",
                "dayride_type1_001",
            ),
            (
                "dayride_type1_001-mp4-t-1115_jpg.rf.0e19cdbaf4ebcd9807f9821253e88a63",
                "dayride_type1_001",
            ),
            (
                "nightride_type3_001-mp4-t-256_jpg.rf.5590acf9db420e97fd1d4fa81ccb14f8",
                "nightride_type3_001",
            ),
            # ... and unsuffixed, which is what the rule used to assume
            ("dayride_type1_001-mp4-t-1062", "dayride_type1_001"),
            ("clip_avi_t_9", "clip"),
            # --- video<N>_<M>, real ids ---
            ("video3_2190_jpg.rf.f9a4bf94d32d92ec52483139a3029050", "video3"),
            ("video11_1870_jpg.rf.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "video11"),
            ("video8_870", "video8"),
            # --- classic frame naming ---
            ("clip_042_frame_0137", "clip_042"),
            ("clip-7-frame-9", "clip-7"),
        ],
    )
    def test_recognised_frames(self, name, clip):
        assert clip_id(name) == clip

    @pytest.mark.parametrize(
        "name",
        [
            # Ordinary stills — the population a loose rule would wreck.
            "d_license_plate_188_jpg.rf.487f425dd3a99b3d42ac72be15690f65",
            "pl_license_plate_327_jpg.rf.dcd2b7a1c3c0296310c252682e78e48f",
            "car-wbs-KL54H369_00000_jpeg.rf.ec38c1bf7ae1c0a7d5a550a0c2e59a47",
            "license_plate_205",
            "license_plate_242",
            # video<N>_<M> near-misses, each documented in _CLIP_PATTERNS
            "video_2190",  # no clip number after `video`
            "myvideo3_2190",  # `video` must start a word
            "video3",  # no frame number
            "video3_2190_640",  # frame number must end the string
            "videos_12",  # `video` not followed by a digit
            # malformed / degenerate
            "",
            "frame_12",  # nothing before the marker to name a clip
            "-mp4-t-5",  # ditto
            "___",
        ],
    )
    def test_not_a_frame(self, name):
        assert clip_id(name) is None

    def test_the_two_real_205_242_stills_stay_apart(self):
        # schema.py refuses a bare trailing-number rule precisely because it
        # would collapse these into one group and hand most of the dataset to a
        # single split.
        assert clip_id("pl_license_plate_205") is None
        assert clip_id("pl_license_plate_242") is None


class TestPlateBox:
    def test_rejects_zero_size(self):
        with pytest.raises(DatasetError, match="degenerate"):
            PlateBox(cx=0.5, cy=0.5, w=0.0, h=0.1)

    def test_rejects_negative_size(self):
        with pytest.raises(DatasetError, match="degenerate"):
            PlateBox(cx=0.5, cy=0.5, w=-0.1, h=0.1)

    def test_rejects_center_outside_image(self):
        with pytest.raises(DatasetError, match="outside image"):
            PlateBox(cx=1.4, cy=0.5, w=0.1, h=0.1)

    def test_tolerates_float_noise_at_the_boundary(self):
        # A box round-tripped through JSON can come back as 1.0000000000000002;
        # rejecting that would fail on honest data.
        PlateBox(cx=1.0000000000000002, cy=0.5, w=0.1, h=0.1)

    def test_xyxy_corners(self):
        box = PlateBox(cx=0.5, cy=0.5, w=0.2, h=0.1)
        x1, y1, x2, y2 = box.xyxy
        assert (x1, y1, x2, y2) == pytest.approx((0.4, 0.45, 0.6, 0.55))

    def test_pixel_size_unnormalizes(self):
        box = PlateBox(cx=0.5, cy=0.5, w=0.1, h=0.05)
        assert box.pixel_size(1920, 1080) == pytest.approx((192.0, 54.0))

    def test_clipped_pulls_box_inside_frame(self):
        # A plate half out of frame: legitimate annotation, invalid YOLO label.
        box = PlateBox(cx=0.05, cy=0.5, w=0.2, h=0.1)
        assert box.xyxy[0] < 0
        clipped = box.clipped()
        assert clipped.xyxy[0] == pytest.approx(0.0)
        assert clipped.w == pytest.approx(0.15)

    def test_clipped_preserves_text_and_region(self):
        box = PlateBox(0.05, 0.5, 0.2, 0.1, text="MH12AB1234", region=Region.INDIA)
        clipped = box.clipped()
        assert clipped.text == "MH12AB1234"
        assert clipped.region is Region.INDIA

    def test_round_trip(self):
        box = PlateBox(0.5, 0.5, 0.2, 0.1, text="DA XY 123", region=Region.GERMANY)
        assert PlateBox.from_dict(box.to_dict()) == box

    def test_round_trip_omits_defaults(self):
        d = PlateBox(0.5, 0.5, 0.2, 0.1).to_dict()
        assert "text" not in d and "region" not in d
        assert PlateBox.from_dict(d).region is Region.UNKNOWN


class TestImageRecord:
    def test_rejects_bad_dimensions(self):
        with pytest.raises(DatasetError, match="bad image size"):
            ImageRecord(image_id="a", width=0, height=100)

    def test_rejects_empty_id(self):
        with pytest.raises(DatasetError, match="non-empty"):
            ImageRecord(image_id="", width=10, height=10)

    def test_boxes_coerced_to_tuple(self):
        record = ImageRecord("a", 10, 10, boxes=[PlateBox(0.5, 0.5, 0.1, 0.1)])
        assert isinstance(record.boxes, tuple)

    def test_background_image_is_valid(self):
        # Images with no plates suppress false positives; they are not an error.
        assert ImageRecord("a", 10, 10).boxes == ()

    def test_group_key_strips_frame_suffix(self):
        assert ImageRecord("clip_042_frame_0137", 10, 10).group_key == "clip_042"
        assert ImageRecord("clip-7-frame-9", 10, 10).group_key == "clip-7"

    def test_explicit_group_wins(self):
        record = ImageRecord("clip_042_frame_0137", 10, 10, group="session-A")
        assert record.group_key == "session-A"

    def test_group_key_falls_back_to_image_id(self):
        # An unframed still is its own group, not lumped with every other image.
        assert ImageRecord("random_photo", 10, 10).group_key == "random_photo"

    def test_primary_region_is_the_most_common(self):
        record = ImageRecord(
            "a",
            10,
            10,
            boxes=(
                PlateBox(0.2, 0.5, 0.1, 0.1, region=Region.INDIA),
                PlateBox(0.5, 0.5, 0.1, 0.1, region=Region.GERMANY),
                PlateBox(0.8, 0.5, 0.1, 0.1, region=Region.INDIA),
            ),
        )
        assert record.primary_region is Region.INDIA

    def test_primary_region_ignores_unknown_when_a_known_one_exists(self):
        record = ImageRecord(
            "a",
            10,
            10,
            boxes=(
                PlateBox(0.2, 0.5, 0.1, 0.1),
                PlateBox(0.5, 0.5, 0.1, 0.1),
                PlateBox(0.8, 0.5, 0.1, 0.1, region=Region.GERMANY),
            ),
        )
        assert record.primary_region is Region.GERMANY

    def test_round_trip(self):
        record = ImageRecord(
            image_id="img1",
            width=1920,
            height=1080,
            boxes=(PlateBox(0.5, 0.5, 0.1, 0.05, text="ABC", region=Region.INDIA),),
            file_name="sub/img1.jpg",
            group="clip1",
            source="uc3m-lp",
            meta={"licence": "ODbL-1.0"},
        )
        assert ImageRecord.from_dict(record.to_dict()) == record
