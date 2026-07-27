"""在线单应性图——关键帧节点 + 链式累积 + 回环约束记录"""
import numpy as np

from core.types import LoopConstraint


class HomographyGraph:
    def __init__(self):
        self._local_edges: dict[int, np.ndarray] = {}
        self._global_transforms: dict[int, np.ndarray] = {}
        self._loop_constraints: list[LoopConstraint] = []
        self._current_id: int = -1
        self.transform_version: int = 0

    # ── keyframe management ──

    def add_first_keyframe(self, frame_id: int) -> int:
        return self._add_node(frame_id, np.eye(3))

    def add_keyframe(self, frame_id: int, H_current_to_previous: np.ndarray) -> int:
        return self._add_node(frame_id, H_current_to_previous)

    def _add_node(self, frame_id: int, H_curr_to_prev: np.ndarray) -> int:
        node_id = self._current_id + 1
        self._local_edges[node_id] = H_curr_to_prev
        self._current_id = node_id

        if node_id == 0:
            self._global_transforms[node_id] = np.eye(3)
        else:
            self._global_transforms[node_id] = (
                self._global_transforms[node_id - 1] @ H_curr_to_prev
            )
        return node_id

    # ── transforms ──

    def get_transform(self, node_id: int) -> np.ndarray:
        return self._global_transforms.get(node_id, np.eye(3))

    def get_current_transform(self) -> np.ndarray:
        if self._current_id < 0:
            return np.eye(3)
        return self._global_transforms[self._current_id]

    # ── loop constraints ──

    def add_loop_constraint(self, constraint: LoopConstraint) -> None:
        self._loop_constraints.append(constraint)

    # ── optimization ──

    def optimize_homography_graph(self) -> None:
        raise NotImplementedError(
            "Homography graph optimization is not implemented in MVP. "
            "Loop constraints are recorded only."
        )

    # ── query ──

    @property
    def num_keyframes(self) -> int:
        return len(self._local_edges)

    @property
    def loop_constraints(self) -> list[LoopConstraint]:
        return list(self._loop_constraints)
