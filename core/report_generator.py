"""报告生成器 — JSON + CSV + 证据帧选择

不变量:
- R1: total_objects = confirmed + tentative + uncertain
- R2: rejected 不进入 total_objects
- R3: JSON、CSV、控制台、API 使用同一个 reportable_objects
- R4: persistent_id 只分配给最终 reportable objects
- R5: rejected 保留 provisional_id 和完整审计信息
"""
import csv
import io
import json

from core.types import GlobalObject, GlobalDetection, ConfirmationStatus


def _to_serializable(obj):
    if hasattr(obj, "value"):  # Enum
        return obj.value
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        return list(obj)
    return str(obj)


def get_reportable_objects(objects: list[GlobalObject]) -> list[GlobalObject]:
    """返回所有非 REJECTED 的对象（统一的 reportable 判定）。"""
    return [o for o in objects if o.confirmation_status != ConfirmationStatus.REJECTED]


class ReportGenerator:
    def __init__(self, object_map=None):
        self.object_map = object_map

    def generate_json_report(self) -> dict:
        objects = self.object_map.get_all() if self.object_map else []
        # R1, R2: 只计算非 REJECTED
        active_objects = [o for o in objects
                          if o.confirmation_status != ConfirmationStatus.REJECTED]
        rejected_objects = [o for o in objects
                            if o.confirmation_status == ConfirmationStatus.REJECTED]

        confirmed = sum(1 for o in active_objects
                        if o.confirmation_status == ConfirmationStatus.CONFIRMED)
        tentative = sum(1 for o in active_objects
                        if o.confirmation_status == ConfirmationStatus.TENTATIVE)
        uncertain = sum(1 for o in active_objects
                        if o.confirmation_status == ConfirmationStatus.UNCERTAIN)

        # 类别分布
        class_counts: dict[str, int] = {}
        for o in active_objects:
            class_counts[o.class_name] = class_counts.get(o.class_name, 0) + 1

        return {
            "total_objects": len(active_objects),
            "confirmed_count": confirmed,
            "tentative_count": tentative,
            "uncertain_count": uncertain,
            "rejected_count": len(rejected_objects),
            "review_required_count": sum(
                1 for o in active_objects
                if o.confirmation_status == ConfirmationStatus.UNCERTAIN
                or len(o.review_flags) > 0
            ),
            "class_counts": class_counts,
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
                    "observation_frame_count": len({obs.frame_id for obs in obj.observations}),
                    "observation_frame_ids_unique": (
                        len({obs.frame_id for obs in obj.observations}) == obj.observation_count
                    ),
                    "keyframe_count": len(obj.keyframe_ids),
                    "track_count": len(obj.track_ids),
                    "centroid_x": obj.centroid_xy[0],
                    "centroid_y": obj.centroid_xy[1],
                    "review_flags": [f.value for f in obj.review_flags],
                    "best_frame_id": obj.best_frame_id,
                    "uncertainty_reasons": obj.uncertainty_reasons,
                    "rejected_reason": obj.rejected_reason,
                    "merged_into_id": obj.merged_into_id,
                    "track_conflict_count": (
                        1 if "track_conflict" in [f.value for f in obj.review_flags] else 0
                    ),
                }
                for obj in active_objects
            ],
            "rejected_objects": [
                {
                    "provisional_id": obj.provisional_id,
                    "class_name": obj.class_name,
                    "rejected_reason": obj.rejected_reason,
                    "merged_into_id": obj.merged_into_id,
                    "observation_count": obj.observation_count,
                    "rejection_evidence": obj.rejection_evidence,
                }
                for obj in rejected_objects
            ],
        }

    def generate_csv_report(self, objects: list[GlobalObject]) -> str:
        buf = io.StringIO(newline="")
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow([
            "persistent_id", "provisional_id", "class_name", "confirmation_status",
            "visibility_status", "confidence", "observation_count",
            "keyframe_count", "track_count", "centroid_x", "centroid_y",
            "review_flags", "best_frame_id", "rejected_reason", "merged_into_id",
        ])
        for obj in objects:
            writer.writerow([
                obj.persistent_id or "",
                obj.provisional_id,
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
                obj.rejected_reason or "",
                obj.merged_into_id or "",
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
