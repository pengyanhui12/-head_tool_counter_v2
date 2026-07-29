"""核心共享类型——模块间通信的统一语言"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class KeyframeDecision(str, Enum):
    SKIP = "skip"
    ACCEPTED = "accepted"
    RECOVERY = "recovery"
    RECOVERY_OK = "recovery_ok"
    LOST = "lost"


class ConfirmationStatus(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    UNCERTAIN = "uncertain"
    REJECTED = "rejected"


class VisibilityStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


class ReviewFlag(str, Enum):
    LIKELY_DUPLICATE = "likely_duplicate"
    CLASS_CONFLICT = "class_conflict"
    EDGE_ONLY = "edge_only"
    LOW_CONFIDENCE = "low_confidence"
    MAPPING_UNSTABLE = "mapping_unstable"
    TRACK_CONFLICT = "track_conflict"


class RecoveryState(str, Enum):
    RECOVERED = "recovered"
    FAILED = "failed"
    LOST = "lost"


@dataclass
class RecoveryResult:
    state: RecoveryState
    anchor_node_id: int | None = None
    H_current_to_anchor: np.ndarray | None = None
    match_result: MatchResult | None = None


@dataclass
class HomographyNode:
    """单应图节点——显式记录 parent 和变换。"""
    node_id: int
    frame_id: int
    parent_node_id: int | None
    H_to_parent: np.ndarray
    H_to_global: np.ndarray


@dataclass
class MergeAudit:
    """合并审计记录。"""
    primary_id: str
    secondary_id: str
    decision: str  # "merged" | "blocked" | "marked_duplicate"
    reason: str
    shared_track_keys: list = field(default_factory=list)
    position_distance: float | None = None
    normalized_distance: float | None = None
    overlapping_frame_ids: list[int] = field(default_factory=list)
    co_occurred: bool = False


@dataclass
class Frame:
    frame_id: int
    timestamp: float
    image: np.ndarray
    is_keyframe: bool = False
    sharpness_score: float = 0.0
    exposure_score: float = 1.0
    mapping_quality: float = 0.0
    camera_profile_id: str = "uncalibrated"


@dataclass
class BufferedFrame:
    frame_id: int
    timestamp: float
    gray: np.ndarray
    sharpness_score: float
    source_frame_id: int


@dataclass
class MatchResult:
    H_source_to_target: np.ndarray | None
    num_keypoints_src: int
    num_keypoints_dst: int
    num_good_matches: int
    num_inliers: int
    inlier_ratio: float
    reprojection_error: float
    occupied_quadrants_src: int
    occupied_quadrants_dst: int
    inlier_bbox_area_ratio_src: float
    inlier_bbox_area_ratio_dst: float
    valid: bool
    failure_reason: str | None = None


@dataclass
class KeyframeTriggerContext:
    max_interval_reached: bool = False
    l2_new_unmatched_detection: bool = False
    track_quality_drop: bool = False
    coverage_growth: float | None = None
    force_end_candidate: bool = False
    l3_required: bool = False
    l3_regions: list[tuple[int, int, int, int]] = field(
        default_factory=list
    )


@dataclass
class KeyframeResult:
    decision: KeyframeDecision
    reason: str
    H_current_to_previous: np.ndarray | None = None
    match_result: MatchResult | None = None


@dataclass(frozen=True)
class DetectionCandidate:
    frame_id: int
    bbox: tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float
    source: str
    image_width: int
    image_height: int


@dataclass
class TrackedDetection:
    candidate: DetectionCandidate
    track_id: int
    track_age: int
    is_new_track: bool = False


@dataclass(frozen=True)
class RawDetection:
    frame_id: int
    keyframe_id: int
    track_id: int
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    corners: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]
    class_id: int
    class_name: str
    confidence: float
    sharpness: float
    mapping_quality: float
    source: str


@dataclass
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float
    age: int = 1
    missed_frames: int = 0
    state: str = "active"
    confidence_history: list[float] = field(default_factory=list)
    detection_history: list[DetectionCandidate] = field(
        default_factory=list
    )
    last_update_frame_id: int = 0
    last_detection_frame_id: int = 0
    generation: int = 0  # logical key = (track_id, generation)


@dataclass
class GlobalDetection:
    frame_id: int
    keyframe_id: int
    track_id: int
    projected_corners: np.ndarray
    projected_center: tuple[float, float]
    polygon_centroid: tuple[float, float]
    polygon_area: float
    class_id: int
    class_name: str
    detection_confidence: float
    sharpness: float
    mapping_quality: float
    edge_quality: float
    size_quality: float
    transform_version: int
    source: str
    bbox_pixels: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # 原始像素坐标 bbox，用于证据帧绘制和 IoU 共现判定


@dataclass
class GlobalObject:
    provisional_id: str
    persistent_id: str | None
    class_name: str
    confirmation_status: ConfirmationStatus
    visibility_status: VisibilityStatus
    review_flags: set[ReviewFlag] = field(default_factory=set)
    confidence: float = 0.0
    vote_distribution: dict[str, float] = field(default_factory=dict)
    observations: list[GlobalDetection] = field(default_factory=list)
    centroid_xy: tuple[float, float] = (0.0, 0.0)
    position_covariance: np.ndarray = field(
        default_factory=lambda: np.eye(2, dtype=float)
    )
    area_range: tuple[float, float] = (0.0, 0.0)
    keyframe_ids: set[int] = field(default_factory=set)
    track_ids: set[int] = field(default_factory=set)
    co_observed_with: set[str] = field(default_factory=set)
    observation_count: int = 0
    uncertainty_reasons: list[str] = field(default_factory=list)
    best_frame_id: int | None = None
    map_version: int = 0
    rejected_reason: str | None = None
    merged_into_id: str | None = None
    rejection_evidence: dict = field(default_factory=dict)


@dataclass(frozen=True)
class LoopConstraint:
    source_node_id: int
    target_node_id: int
    H_source_to_target: np.ndarray
    num_inliers: int
    reprojection_error: float


@dataclass
class TrackerPreview:
    unmatched_detection_indices: list[int] = field(default_factory=list)
    l2_new_unmatched_detection: bool = False
    track_quality_drop: bool = False


@dataclass
class RebuildResult:
    map_version: int
    old_to_new_id: dict[str, str]
    affected_object_ids: list[str]
