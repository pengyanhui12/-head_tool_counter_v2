"""Generate consistent JSON/CSV reports and select evidence frames."""
import csv
import io
import json
from dataclasses import dataclass
from typing import Callable

from core.types import ConfirmationStatus, GlobalDetection, GlobalObject, ReviewFlag


REPORT_SCHEMA_VERSION = 1


def empty_report() -> dict:
    """Return a fresh empty payload for every report-shaped output sink."""
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "total_objects": 0,
        "confirmed_count": 0,
        "uncertain_count": 0,
        "review_candidate_count": 0,
        "tentative_count": 0,
        "likely_partial_duplicate_count": 0,
        "rejected_count": 0,
        "review_required_count": 0,
        "class_counts": {},
        "objects": [],
        "review_candidates": [],
        "rejected_objects": [],
    }


@dataclass(frozen=True)
class ObjectPartitions:
    counted: tuple[GlobalObject, ...]
    review_candidates: tuple[GlobalObject, ...]
    rejected: tuple[GlobalObject, ...]


def partition_objects(objects: list[GlobalObject]) -> ObjectPartitions:
    """Partition objects by canonical formal-count policy, preserving order."""
    counted = []
    review_candidates = []
    rejected = []
    for obj in objects:
        if obj.confirmation_status in (
            ConfirmationStatus.CONFIRMED,
            ConfirmationStatus.UNCERTAIN,
        ):
            counted.append(obj)
        elif obj.confirmation_status == ConfirmationStatus.TENTATIVE:
            review_candidates.append(obj)
        elif obj.confirmation_status == ConfirmationStatus.REJECTED:
            rejected.append(obj)
    return ObjectPartitions(
        counted=tuple(counted),
        review_candidates=tuple(review_candidates),
        rejected=tuple(rejected),
    )


def get_counted_objects(objects: list[GlobalObject]) -> list[GlobalObject]:
    """Return objects included in the formal count."""
    return list(partition_objects(objects).counted)


def get_review_candidates(objects: list[GlobalObject]) -> list[GlobalObject]:
    """Return tentative objects that require review."""
    return list(partition_objects(objects).review_candidates)


def get_display_objects(objects: list[GlobalObject]) -> list[GlobalObject]:
    """Return the canonical counted-plus-review collection for visual sinks."""
    partitions = partition_objects(objects)
    return [*partitions.counted, *partitions.review_candidates]


def get_reportable_objects(objects: list[GlobalObject]) -> list[GlobalObject]:
    """Compatibility alias for formally counted confirmed and uncertain objects."""
    return get_counted_objects(objects)


def _to_serializable(obj):
    if hasattr(obj, "value"):
        return obj.value
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        return list(obj)
    return str(obj)


class ReportGenerator:
    def __init__(self, object_map=None):
        self.object_map = object_map

    def generate_json_report(self) -> dict:
        objects = self.object_map.get_all() if self.object_map else []
        partitions = partition_objects(objects)
        counted_objects = list(partitions.counted)
        review_candidates = list(partitions.review_candidates)
        rejected_objects = list(partitions.rejected)
        resolve_id = self._identity_resolver(partitions)

        confirmed = sum(
            obj.confirmation_status == ConfirmationStatus.CONFIRMED
            for obj in counted_objects
        )
        uncertain = sum(
            obj.confirmation_status == ConfirmationStatus.UNCERTAIN
            for obj in counted_objects
        )
        class_counts: dict[str, int] = {}
        for obj in counted_objects:
            class_counts[obj.class_name] = class_counts.get(obj.class_name, 0) + 1

        report = empty_report()
        report.update({
            "total_objects": len(counted_objects),
            "confirmed_count": confirmed,
            "uncertain_count": uncertain,
            "review_candidate_count": len(review_candidates),
            "tentative_count": len(review_candidates),
            "likely_partial_duplicate_count": sum(
                ReviewFlag.LIKELY_PARTIAL_DUPLICATE in obj.review_flags
                for obj in partitions.review_candidates
            ),
            "rejected_count": len(rejected_objects),
            "review_required_count": sum(
                obj.confirmation_status == ConfirmationStatus.UNCERTAIN
                or bool(obj.review_flags)
                for obj in counted_objects
            ) + len(review_candidates),
            "class_counts": class_counts,
            "objects": [
                self._serialize_counted_or_review(obj, True, resolve_id)
                for obj in counted_objects
            ],
            "review_candidates": [
                self._serialize_counted_or_review(obj, False, resolve_id)
                for obj in review_candidates
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
        })
        return report

    @staticmethod
    def _identity_resolver(
        partitions: ObjectPartitions,
    ) -> Callable[[str | None], str | None]:
        persistent_ids = {
            obj.provisional_id: obj.persistent_id
            for obj in partitions.counted
            if obj.persistent_id is not None
        }

        def resolve(candidate_id: str | None) -> str | None:
            if candidate_id is None:
                return None
            return persistent_ids.get(candidate_id, candidate_id)

        return resolve

    @staticmethod
    def _serialize_counted_or_review(
        obj: GlobalObject,
        counted: bool,
        resolve_id: Callable[[str | None], str | None],
    ) -> dict:
        """Serialize the shared counted/review record contract."""
        frame_ids = {obs.frame_id for obs in obj.observations}
        return {
            "persistent_id": obj.persistent_id,
            "provisional_id": obj.provisional_id,
            "class_name": obj.class_name,
            "confirmation_status": obj.confirmation_status.value,
            "visibility_status": obj.visibility_status.value,
            "confidence": obj.confidence,
            "vote_distribution": obj.vote_distribution,
            "observation_count": obj.observation_count,
            "observation_frame_count": len(frame_ids),
            "observation_frame_ids_unique": len(frame_ids) == obj.observation_count,
            "keyframe_count": len(obj.keyframe_ids),
            "track_count": len(obj.track_ids),
            "centroid_x": obj.centroid_xy[0],
            "centroid_y": obj.centroid_xy[1],
            "review_flags": sorted(flag.value for flag in obj.review_flags),
            "best_frame_id": obj.best_frame_id,
            "uncertainty_reasons": obj.uncertainty_reasons,
            "rejected_reason": obj.rejected_reason,
            "merged_into_id": obj.merged_into_id,
            "track_conflict_count": int(ReviewFlag.TRACK_CONFLICT in obj.review_flags),
            "counted": counted,
            "likely_partial_duplicate_of": resolve_id(obj.likely_partial_duplicate_of),
            "duplicate_candidate_ids": sorted(
                resolve_id(candidate_id)
                for candidate_id in obj.duplicate_candidate_ids
            ),
            "duplicate_evidence": ReportGenerator._serialize_duplicate_evidence(
                obj.duplicate_evidence, resolve_id
            ),
        }

    @staticmethod
    def _serialize_duplicate_evidence(
        evidence: dict,
        resolve_id: Callable[[str | None], str | None],
    ) -> dict:
        """Copy advisory evidence and resolve nested candidate identities."""
        serialized = dict(evidence)
        candidates = []
        for candidate in evidence.get("candidates", []):
            serialized_candidate = dict(candidate)
            serialized_candidate["candidate_id"] = resolve_id(
                serialized_candidate.get("candidate_id")
            )
            candidates.append(serialized_candidate)
        if "candidates" in evidence:
            serialized["candidates"] = candidates
        return serialized

    def generate_csv_report(self, objects: list[GlobalObject]) -> str:
        partitions = partition_objects(objects)
        counted_ids = {id(obj) for obj in partitions.counted}
        resolve_id = self._identity_resolver(partitions)
        buf = io.StringIO(newline="")
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow([
            "persistent_id", "provisional_id", "class_name", "confirmation_status",
            "visibility_status", "confidence", "observation_count",
            "keyframe_count", "track_count", "centroid_x", "centroid_y",
            "review_flags", "best_frame_id", "rejected_reason", "merged_into_id",
            "counted", "review_status", "likely_partial_duplicate_of",
            "duplicate_candidate_ids", "duplicate_evidence",
        ])
        for obj in objects:
            duplicate_evidence = self._serialize_duplicate_evidence(
                obj.duplicate_evidence, resolve_id
            )
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
                "|".join(sorted(flag.value for flag in obj.review_flags)),
                obj.best_frame_id or "",
                obj.rejected_reason or "",
                obj.merged_into_id or "",
                "true" if id(obj) in counted_ids else "false",
                self._review_status(obj),
                resolve_id(obj.likely_partial_duplicate_of) or "",
                "|".join(sorted(
                    resolve_id(candidate_id)
                    for candidate_id in obj.duplicate_candidate_ids
                )),
                json.dumps(
                    duplicate_evidence,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=_to_serializable,
                ),
            ])
        return buf.getvalue()

    @staticmethod
    def _review_status(obj: GlobalObject) -> str:
        if obj.confirmation_status == ConfirmationStatus.REJECTED:
            return "rejected"
        decision = obj.duplicate_evidence.get("decision")
        if decision in {
            "likely_partial_duplicate",
            "ambiguous",
            "no_match",
        }:
            return decision
        if ReviewFlag.LIKELY_PARTIAL_DUPLICATE in obj.review_flags:
            return "likely_partial_duplicate"
        if ReviewFlag.AMBIGUOUS_DUPLICATE_CANDIDATE in obj.review_flags:
            return "ambiguous"
        if obj.confirmation_status == ConfirmationStatus.TENTATIVE:
            return "tentative"
        if (obj.confirmation_status == ConfirmationStatus.UNCERTAIN
                or obj.review_flags):
            return "uncertain"
        return ""

    def find_evidence_frames(self, objects: list[GlobalObject]) -> dict[str, int]:
        result = {}
        for obj in objects:
            if not obj.observations:
                continue
            best: GlobalDetection = max(
                obj.observations,
                key=lambda observation: (
                    observation.sharpness * observation.detection_confidence
                ),
            )
            result[obj.provisional_id] = best.frame_id
            obj.best_frame_id = best.frame_id
        return result
