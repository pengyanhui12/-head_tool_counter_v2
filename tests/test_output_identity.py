import json
from pathlib import Path

import cv2
import numpy as np

from apps import offline_scan
from core.evidence_extractor import EvidenceExtractor
from core.global_mosaic import generate_global_mosaic
from core.session_store import SessionStore
from core.types import (
    ConfirmationStatus,
    GlobalDetection,
    GlobalObject,
    VisibilityStatus,
)


def make_detection() -> GlobalDetection:
    return GlobalDetection(
        frame_id=1,
        keyframe_id=1,
        track_id=1,
        projected_corners=np.array([[0, 0], [20, 0], [20, 10], [0, 10]]),
        projected_center=(10, 5),
        polygon_centroid=(10, 5),
        polygon_area=200,
        class_id=0,
        class_name="wrench",
        detection_confidence=0.9,
        sharpness=100.0,
        mapping_quality=0.8,
        edge_quality=1.0,
        size_quality=1.0,
        transform_version=1,
        source="L1",
        bbox_pixels=(1, 1, 10, 8),
    )


def make_object(provisional_id, status, persistent_id=None):
    return GlobalObject(
        provisional_id=provisional_id,
        persistent_id=persistent_id,
        class_name="wrench",
        confirmation_status=status,
        visibility_status=VisibilityStatus.ACTIVE,
        observations=[make_detection()],
        centroid_xy=(10, 5),
        confidence=0.9,
        observation_count=1,
    )


def test_evidence_uses_persistent_and_provisional_filenames(tmp_path, monkeypatch):
    counted = make_object("P-0001", ConfirmationStatus.CONFIRMED, "GO-0001")
    tentative = make_object("P-0002", ConfirmationStatus.TENTATIVE)

    class FakeCapture:
        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: FakeCapture())
    monkeypatch.setattr(
        EvidenceExtractor,
        "_read_frame",
        staticmethod(lambda _cap, _frame_id, _cache: np.zeros((20, 20, 3), dtype=np.uint8)),
    )
    written = []
    monkeypatch.setattr(cv2, "imwrite", lambda path, _image: written.append(Path(path)) or True)

    EvidenceExtractor().extract("unused.mp4", [counted, tentative], str(tmp_path))

    assert [path.name for path in written] == [
        "GO-0001_wrench.jpg",
        "P-0002_wrench.jpg",
    ]


def test_mosaic_uses_provisional_label_for_tentative(tmp_path, monkeypatch):
    tentative = make_object("P-0002", ConfirmationStatus.TENTATIVE)

    class Graph:
        num_keyframes = 1
        nodes = [(0, 0, np.eye(3))]

    labels = []
    original_put_text = cv2.putText

    def record_label(image, text, *args, **kwargs):
        labels.append(text)
        return original_put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", record_label)
    monkeypatch.setattr(cv2, "imwrite", lambda _path, _image: True)

    result = generate_global_mosaic(
        "unused.mp4", Graph(), [tentative], str(tmp_path), skip_warp=True
    )

    assert result is not None
    assert "P-0002 wrench" in labels
    assert all("None" not in label for label in labels)


def test_session_report_payload_retains_counted_and_review_objects(tmp_path):
    store = SessionStore(str(tmp_path))
    session_dir = Path(store.create_session("input.mp4"))
    report = {
        "total_objects": 1,
        "objects": [{"provisional_id": "P-0001", "counted": True}],
        "review_candidates": [{"provisional_id": "P-0002", "counted": False}],
    }

    store.save_report(report)

    saved = json.loads((session_dir / "objects.json").read_text(encoding="utf-8"))
    assert saved["objects"] == report["objects"]
    assert saved["review_candidates"] == report["review_candidates"]


def test_report_snapshot_generates_json_once_and_keeps_csv_together():
    class Generator:
        def __init__(self):
            self.json_calls = 0

        def generate_json_report(self):
            self.json_calls += 1
            return {"total_objects": 2, "objects": [], "review_candidates": []}

        def generate_csv_report(self, objects):
            assert objects == ["object"]
            return "csv\n"

    generator = Generator()

    report, csv_text = offline_scan.build_report_snapshot(generator, ["object"])

    assert generator.json_calls == 1
    assert report["total_objects"] == 2
    assert csv_text == "csv\n"
