"""报告生成器 — JSON + CSV + 证据帧选择"""
import csv
import io
import json

from core.types import GlobalObject, GlobalDetection


def _to_serializable(obj):
    if hasattr(obj, "value"):  # Enum
        return obj.value
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        return list(obj)
    return str(obj)


class ReportGenerator:
    def __init__(self, object_map=None):
        self.object_map = object_map

    def generate_json_report(self) -> dict:
        objects = self.object_map.get_all() if self.object_map else []
        return {
            "total_objects": len(objects),
            "objects": [
                {
                    "persistent_id": obj.persistent_id,
                    "provisional_id": obj.provisional_id,
                    "class_name": obj.class_name,
                    "confirmation_status": obj.confirmation_status.value,
                    "visibility_status": obj.visibility_status.value,
                    "confidence": obj.confidence,
                    "vote_distribution": obj.vote_distribution,
                    "observation_count": obj.observation_count,
                    "keyframe_count": len(obj.keyframe_ids),
                    "track_count": len(obj.track_ids),
                    "centroid_x": obj.centroid_xy[0],
                    "centroid_y": obj.centroid_xy[1],
                    "review_flags": [f.value for f in obj.review_flags],
                    "best_frame_id": obj.best_frame_id,
                    "uncertainty_reasons": obj.uncertainty_reasons,
                }
                for obj in objects
            ],
        }

    def generate_csv_report(self, objects: list[GlobalObject]) -> str:
        buf = io.StringIO(newline="")
        writer = csv.writer(buf)
        writer.writerow([
            "persistent_id", "class_name", "confirmation_status",
            "visibility_status", "confidence", "observation_count",
            "keyframe_count", "track_count", "centroid_x", "centroid_y",
            "review_flags", "best_frame_id",
        ])
        for obj in objects:
            writer.writerow([
                obj.persistent_id or "",
                obj.class_name,
                obj.confirmation_status.value,
                obj.visibility_status.value,
                f"{obj.confidence:.3f}",
                obj.observation_count,
                len(obj.keyframe_ids),
                len(obj.track_ids),
                f"{obj.centroid_xy[0]:.1f}",
                f"{obj.centroid_xy[1]:.1f}",
                "|".join(f.value for f in obj.review_flags),
                obj.best_frame_id or "",
            ])
        return buf.getvalue()

    def find_evidence_frames(self, objects: list[GlobalObject]) -> dict[str, int]:
        result = {}
        for obj in objects:
            if not obj.observations:
                continue
            best = max(
                obj.observations,
                key=lambda o: o.sharpness * o.detection_confidence
            )
            result[obj.provisional_id] = best.frame_id
            obj.best_frame_id = best.frame_id
        return result
