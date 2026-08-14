"""Pure advisory evaluation of tentative partial-duplicate objects."""
from __future__ import annotations

from dataclasses import dataclass
from math import hypot, inf, sqrt

import cv2
import numpy as np

from core.types import ConfirmationStatus, GlobalObject


@dataclass(frozen=True)
class PartialDuplicateConfig:
    min_containment: float = 0.75
    max_normalized_distance: float = 0.75
    max_absolute_distance_px: float = 80.0
    min_mapping_quality: float = 0.50
    max_area_ratio: float = 0.60
    min_candidate_margin: float = 0.15
    min_observations_confirmed: int = 3
    min_keyframes_confirmed: int = 2


@dataclass(frozen=True)
class PartialDuplicateDecision:
    decision: str
    candidate_id: str | None = None
    candidate_ids: tuple[str, ...] = ()
    containment_score: float | None = None
    normalized_distance: float | None = None
    mapping_quality: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class _GeometryResult:
    containment: float | None
    mapping_quality: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class _PassingCandidate:
    candidate_id: str
    score: float
    containment: float
    normalized_distance: float
    mapping_quality: float | None


class PartialDuplicateEvaluator:
    def __init__(self, config: PartialDuplicateConfig | None = None):
        self.config = config or PartialDuplicateConfig()

    def evaluate(
        self,
        tentative: GlobalObject,
        confirmed_candidates: list[GlobalObject],
        co_occurred_pairs: set[frozenset[str]],
    ) -> PartialDuplicateDecision:
        if tentative.confirmation_status != ConfirmationStatus.TENTATIVE:
            return PartialDuplicateDecision("no_match", reason="not_tentative")

        if (
            tentative.observation_count
            >= self.config.min_observations_confirmed
            and len(tentative.keyframe_ids)
            >= self.config.min_keyframes_confirmed
        ):
            return PartialDuplicateDecision(
                "no_match", reason="sufficient_confirmation_evidence"
            )

        candidates = [
            candidate
            for candidate in confirmed_candidates
            if candidate.confirmation_status == ConfirmationStatus.CONFIRMED
            and candidate.class_name == tentative.class_name
        ]
        if not candidates:
            return PartialDuplicateDecision(
                "no_match", reason="no_same_class_confirmed"
            )

        passing: list[_PassingCandidate] = []
        rejection_reasons: list[str] = []
        for candidate in candidates:
            candidate_id = candidate.provisional_id
            pair = frozenset((tentative.provisional_id, candidate_id))
            if pair in co_occurred_pairs:
                rejection_reasons.append("independent_co_occurrence")
                continue

            geometry = self._compare_geometry(tentative, candidate)
            if geometry.containment is None:
                rejection_reasons.append(geometry.reason)
                continue
            if geometry.containment < self.config.min_containment:
                rejection_reasons.append("insufficient_containment")
                continue

            confirmed_max_area = _maximum_area(candidate)
            centroid_values = (*tentative.centroid_xy, *candidate.centroid_xy)
            if not all(np.isfinite(value) for value in centroid_values):
                rejection_reasons.append("distance_exceeded")
                continue
            distance = hypot(
                tentative.centroid_xy[0] - candidate.centroid_xy[0],
                tentative.centroid_xy[1] - candidate.centroid_xy[1],
            )
            normalized_distance = (
                distance / sqrt(confirmed_max_area)
                if np.isfinite(confirmed_max_area) and confirmed_max_area > 0.0
                else inf
            )
            if (
                normalized_distance > self.config.max_normalized_distance
                or distance > self.config.max_absolute_distance_px
            ):
                rejection_reasons.append("distance_exceeded")
                continue

            confirmed_area = _representative_area(candidate)
            tentative_area = _representative_area(tentative)
            area_ratio = (
                tentative_area / confirmed_area
                if np.isfinite(tentative_area)
                and tentative_area >= 0.0
                and np.isfinite(confirmed_area)
                and confirmed_area > 0.0
                else inf
            )
            if area_ratio > self.config.max_area_ratio:
                rejection_reasons.append("not_partial_scale")
                continue

            passing.append(
                _PassingCandidate(
                    candidate_id=candidate_id,
                    score=geometry.containment - normalized_distance,
                    containment=geometry.containment,
                    normalized_distance=normalized_distance,
                    mapping_quality=geometry.mapping_quality,
                )
            )

        if not passing:
            return PartialDuplicateDecision(
                "no_match", reason=_select_rejection_reason(rejection_reasons)
            )

        passing.sort(key=lambda item: (-item.score, item.candidate_id))
        best = passing[0]
        if (
            len(passing) > 1
            and best.score - passing[1].score < self.config.min_candidate_margin
        ):
            second = passing[1]
            return PartialDuplicateDecision(
                "ambiguous",
                candidate_ids=(best.candidate_id, second.candidate_id),
                containment_score=best.containment,
                normalized_distance=best.normalized_distance,
                mapping_quality=best.mapping_quality,
                reason="candidate_margin_below_threshold",
            )

        return PartialDuplicateDecision(
            "attributed",
            candidate_id=best.candidate_id,
            candidate_ids=(best.candidate_id,),
            containment_score=best.containment,
            normalized_distance=best.normalized_distance,
            mapping_quality=best.mapping_quality,
            reason="unique_candidate",
        )

    def _compare_geometry(
        self, tentative: GlobalObject, confirmed: GlobalObject
    ) -> _GeometryResult:
        comparisons: list[tuple[float, float | None]] = []
        low_mapping_quality = False

        for tentative_observation in tentative.observations:
            for confirmed_observation in confirmed.observations:
                if tentative_observation.frame_id == confirmed_observation.frame_id:
                    containment = _rectangle_containment(
                        tentative_observation.bbox_pixels,
                        confirmed_observation.bbox_pixels,
                    )
                    if containment is not None:
                        comparisons.append((containment, None))
                    continue

                if not (
                    _valid_polygon(tentative_observation.projected_corners)
                    and _valid_polygon(confirmed_observation.projected_corners)
                ):
                    continue
                mapping_qualities = (
                    tentative_observation.mapping_quality,
                    confirmed_observation.mapping_quality,
                )
                if not all(
                    np.isfinite(quality)
                    and quality >= self.config.min_mapping_quality
                    for quality in mapping_qualities
                ):
                    low_mapping_quality = True
                    continue
                mapping_quality = min(mapping_qualities)
                comparisons.append(
                    (
                        _polygon_containment(
                            tentative_observation.projected_corners,
                            confirmed_observation.projected_corners,
                        ),
                        mapping_quality,
                    )
                )

        if comparisons:
            containment, mapping_quality = max(
                comparisons,
                key=lambda comparison: (
                    comparison[0],
                    comparison[1] if comparison[1] is not None else inf,
                ),
            )
            return _GeometryResult(containment, mapping_quality)
        if low_mapping_quality:
            return _GeometryResult(None, reason="low_mapping_quality")
        return _GeometryResult(None, reason="no_comparable_geometry")


def _representative_area(obj: GlobalObject) -> float:
    area_min, area_max = obj.area_range
    return (float(area_min) + float(area_max)) / 2.0


def _maximum_area(obj: GlobalObject) -> float:
    return max((float(area) for area in obj.area_range), default=0.0)


def _rectangle_containment(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float | None:
    first_x1, first_y1, first_x2, first_y2 = map(float, first)
    second_x1, second_y1, second_x2, second_y2 = map(float, second)
    if not all(
        np.isfinite(value)
        for value in (
            first_x1,
            first_y1,
            first_x2,
            first_y2,
            second_x1,
            second_y1,
            second_x2,
            second_y2,
        )
    ):
        return None
    first_area = max(0.0, first_x2 - first_x1) * max(0.0, first_y2 - first_y1)
    second_area = max(0.0, second_x2 - second_x1) * max(
        0.0, second_y2 - second_y1
    )
    smaller_area = min(first_area, second_area)
    if smaller_area <= 0.0:
        return None
    intersection_width = max(
        0.0, min(first_x2, second_x2) - max(first_x1, second_x1)
    )
    intersection_height = max(
        0.0, min(first_y2, second_y2) - max(first_y1, second_y1)
    )
    return intersection_width * intersection_height / smaller_area


def _valid_polygon(points: np.ndarray) -> bool:
    polygon = np.asarray(points, dtype=np.float32)
    return (
        polygon.ndim == 2
        and polygon.shape[0] >= 3
        and polygon.shape[1] == 2
        and bool(np.all(np.isfinite(polygon)))
        and abs(float(cv2.contourArea(polygon))) > 0.0
    )


def _polygon_containment(first: np.ndarray, second: np.ndarray) -> float:
    first_hull = cv2.convexHull(np.asarray(first, dtype=np.float32))
    second_hull = cv2.convexHull(np.asarray(second, dtype=np.float32))
    first_area = abs(float(cv2.contourArea(first_hull)))
    second_area = abs(float(cv2.contourArea(second_hull)))
    smaller_area = min(first_area, second_area)
    if smaller_area <= 0.0:
        return 0.0
    intersection_area, _ = cv2.intersectConvexConvex(first_hull, second_hull)
    return min(1.0, max(0.0, float(intersection_area) / smaller_area))


def _select_rejection_reason(reasons: list[str]) -> str:
    priority = (
        "independent_co_occurrence",
        "low_mapping_quality",
        "no_comparable_geometry",
        "insufficient_containment",
        "distance_exceeded",
        "not_partial_scale",
    )
    return next((reason for reason in priority if reason in reasons), "no_match")
