from types import SimpleNamespace

import numpy as np

from core.detector import Detector
from core.types import DetectionCandidate


class FakeModel:
    def __init__(self):
        self.images = []

    def __call__(self, image, **kwargs):
        self.images.append(image.copy())
        boxes = SimpleNamespace(
            xyxy=np.array([[1.0, 2.0, 5.0, 10.0]]),
            cls=np.array([0]),
            conf=np.array([0.75]),
        )
        return [SimpleNamespace(boxes=boxes)]


def make_detector() -> tuple[Detector, FakeModel]:
    detector = Detector(model_path="")
    model = FakeModel()
    detector._model = model
    detector._names = {0: "wrench"}
    return detector, model


def test_l3_regions_are_cropped_and_boxes_restored_to_full_image():
    detector, model = make_detector()
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    candidates = detector.detect(
        image=image,
        level="L3",
        frame_id=7,
        regions=[(10, 20, 60, 80), (90, 70, 150, 120)],
    )

    assert [crop.shape[:2] for crop in model.images] == [(60, 50), (30, 10)]
    assert [candidate.bbox for candidate in candidates] == [
        (11.0, 22.0, 15.0, 30.0),
        (91.0, 72.0, 95.0, 80.0),
    ]
    assert all(candidate.image_width == 100 for candidate in candidates)
    assert all(candidate.image_height == 100 for candidate in candidates)
    assert all(candidate.source == "L3" for candidate in candidates)


def test_regions_are_clipped_and_empty_regions_are_skipped():
    detector, model = make_detector()
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    candidates = detector.detect(
        image=image,
        level="L3",
        frame_id=8,
        regions=[(-10, -20, 20, 30), (50, 50, 50, 70), (200, 200, 220, 220)],
    )

    assert [crop.shape[:2] for crop in model.images] == [(30, 20)]
    assert [candidate.bbox for candidate in candidates] == [
        (1.0, 2.0, 5.0, 10.0),
    ]


def test_none_regions_uses_full_image_and_empty_regions_run_nothing():
    detector, model = make_detector()
    image = np.zeros((40, 60, 3), dtype=np.uint8)

    full_image_candidates = detector.detect(
        image=image,
        level="L1",
        frame_id=1,
        regions=None,
    )
    empty_region_candidates = detector.detect(
        image=image,
        level="L3",
        frame_id=2,
        regions=[],
    )

    assert [call.shape[:2] for call in model.images] == [(40, 60)]
    assert full_image_candidates[0].bbox == (1.0, 2.0, 5.0, 10.0)
    assert empty_region_candidates == []


def _candidate(
    bbox=(20, 20, 40, 40),
    confidence=0.2,
    image_width=100,
    image_height=100,
):
    return DetectionCandidate(
        frame_id=1,
        bbox=bbox,
        class_id=0,
        class_name="wrench",
        confidence=confidence,
        source="L2",
        image_width=image_width,
        image_height=image_height,
    )


def test_l3_regions_are_empty_when_l3_is_disabled():
    regions = Detector.select_l3_regions(
        [_candidate()],
        {
            "enabled": False,
            "low_confidence_upper": 0.35,
            "min_box_area_ratio": 0.001,
            "roi_margin_ratio": 0.20,
        },
    )

    assert regions == []


def test_l3_regions_filter_and_expand_low_confidence_boxes():
    regions = Detector.select_l3_regions(
        [
            _candidate(bbox=(20, 20, 40, 40), confidence=0.20),
            _candidate(bbox=(50, 50, 52, 52), confidence=0.20),
            _candidate(bbox=(60, 60, 80, 80), confidence=0.50),
        ],
        {
            "enabled": True,
            "low_confidence_upper": 0.35,
            "min_box_area_ratio": 0.01,
            "roi_margin_ratio": 0.20,
        },
    )

    assert regions == [(16, 16, 44, 44)]
