"""全局对象地图 — confirmation/visibility 双轴状态 + 双 ID + 审计字段"""
from core.types import (
    GlobalObject,
    GlobalDetection,
    ConfirmationStatus,
    VisibilityStatus,
    ReviewFlag,
)
from core.report_generator import get_counted_objects


class GlobalObjectMap:
    def __init__(self):
        self._objects: list[GlobalObject] = []
        self._provisional_counter = 0
        self._persistent_counter = 0
        self.map_version = 0

    def create_object(self, detection: GlobalDetection) -> GlobalObject:
        self._provisional_counter += 1
        obj = GlobalObject(
            provisional_id=f"P-{self._provisional_counter:04d}",
            persistent_id=None,
            class_name=detection.class_name,
            confirmation_status=ConfirmationStatus.TENTATIVE,
            visibility_status=VisibilityStatus.ACTIVE,
            map_version=self.map_version,
        )
        obj.observations.append(detection)
        obj.observation_count = 1
        obj.centroid_xy = detection.polygon_centroid
        if detection.track_id is not None:
            obj.track_ids.add(detection.track_id)
        obj.keyframe_ids.add(detection.keyframe_id)
        self._objects.append(obj)
        return obj

    def get_all(self) -> list[GlobalObject]:
        return list(self._objects)

    def get_by_provisional(self, provisional_id: str) -> GlobalObject | None:
        for obj in self._objects:
            if obj.provisional_id == provisional_id:
                return obj
        return None

    def get_reportable(self) -> list[GlobalObject]:
        """只返回非 REJECTED 对象（R4: persistent ID 只分配给 reportable）。"""
        return get_counted_objects(self._objects)

    def assign_persistent_ids(self) -> None:
        """只为非 REJECTED 对象分配 persistent ID（R4）。"""
        for obj in self._objects:
            if obj.confirmation_status not in (
                ConfirmationStatus.CONFIRMED,
                ConfirmationStatus.UNCERTAIN,
            ):
                continue
            if obj.persistent_id is None:
                self._persistent_counter += 1
                obj.persistent_id = f"GO-{self._persistent_counter:04d}"

    def set_confirmation(
        self, provisional_id: str, status: ConfirmationStatus, reason: str = ""
    ) -> None:
        obj = self.get_by_provisional(provisional_id)
        if obj:
            obj.confirmation_status = status
            if reason:
                obj.uncertainty_reasons.append(reason)

    def set_visibility(
        self, provisional_id: str, status: VisibilityStatus
    ) -> None:
        obj = self.get_by_provisional(provisional_id)
        if obj:
            obj.visibility_status = status

    def add_review_flag(self, provisional_id: str, flag: ReviewFlag) -> None:
        obj = self.get_by_provisional(provisional_id)
        if obj:
            obj.review_flags.add(flag)

    @property
    def summary(self) -> dict:
        counts = {}
        for obj in self._objects:
            key = obj.confirmation_status.value
            counts[key] = counts.get(key, 0) + 1
        return counts
