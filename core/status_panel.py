"""实时状态面板"""
from dataclasses import dataclass, field


@dataclass
class StatusPanel:
    class_counts: dict[str, int] = field(default_factory=dict)
    confirmation_counts: dict[str, int] = field(default_factory=dict)
    visibility_counts: dict[str, int] = field(default_factory=dict)
    review_flag_counts: dict[str, int] = field(default_factory=dict)
    total_frames: int = 0
    accepted_keyframes: int = 0
    mapping_state: str = "initializing"
    transform_version: int = 0
    map_version: int = 0

    def update(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def get_summary(self) -> dict:
        return {
            "total_frames": self.total_frames,
            "accepted_keyframes": self.accepted_keyframes,
            "mapping_state": self.mapping_state,
            "class_counts": self.class_counts,
            "confirmation_counts": self.confirmation_counts,
            "visibility_counts": self.visibility_counts,
            "review_flag_counts": self.review_flag_counts,
            "transform_version": self.transform_version,
            "map_version": self.map_version,
        }
