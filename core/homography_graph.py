"""在线单应性图——关键帧节点 + 链式累积 + 回环约束记录

不变量:
- H1: 无可靠矩阵不得创建图节点
- H2: 不存在 node_id 时不得返回单位阵
- H3: H_current_to_anchor 不得作为 H_current_to_previous
- H4: 节点必须显式记录 parent_node_id
- H5: 加图前必须检查矩阵有限性和投影范围
"""
import numpy as np

from core.types import LoopConstraint, HomographyNode


class HomographyGraph:
    def __init__(self):
        self._nodes: dict[int, HomographyNode] = {}
        self._local_edges: dict[int, np.ndarray] = {}
        self._global_transforms: dict[int, np.ndarray] = {}
        self._frame_ids: dict[int, int] = {}
        self._parent_ids: dict[int, int | None] = {}
        self._loop_constraints: list[LoopConstraint] = []
        self._current_id: int = -1
        self.transform_version: int = 0

    # ── keyframe management ──

    def add_first_keyframe(self, frame_id: int) -> int:
        """添加第一个关键帧（单位阵）。"""
        node_id = 0
        H = np.eye(3)
        node = HomographyNode(
            node_id=node_id,
            frame_id=frame_id,
            parent_node_id=None,
            H_to_parent=H.copy(),
            H_to_global=H.copy(),
        )
        self._nodes[node_id] = node
        self._local_edges[node_id] = H.copy()
        self._frame_ids[node_id] = frame_id
        self._parent_ids[node_id] = None
        self._global_transforms[node_id] = H.copy()
        self._current_id = node_id
        self.transform_version += 1
        return node_id

    def add_keyframe(
        self,
        frame_id: int,
        H_current_to_parent: np.ndarray,
        parent_node_id: int | None = None,
    ) -> int:
        """添加关键帧节点，显式指定 parent。

        Args:
            frame_id: 当前帧编号
            H_current_to_parent: 当前帧到 parent 的单应矩阵
            parent_node_id: parent 节点 ID，None 则使用上一个节点

        Returns:
            node_id

        Raises:
            ValueError: 矩阵无效（None、非有限、接近奇异）
        """
        # H1: 检查矩阵有效性
        self._validate_homography(H_current_to_parent)

        # H4: 显式 parent
        if parent_node_id is None:
            parent_node_id = self._current_id if self._current_id >= 0 else None

        node_id = self._current_id + 1

        # H3: parent 必须存在
        if parent_node_id is not None and parent_node_id not in self._global_transforms:
            raise KeyError(
                f"Parent node {parent_node_id} not found in graph. "
                f"Cannot add keyframe {frame_id}."
            )

        # 计算 H_to_global
        if parent_node_id is not None:
            H_parent_to_global = self._global_transforms[parent_node_id]
            H_to_global = H_parent_to_global @ H_current_to_parent
        else:
            H_to_global = np.eye(3)

        node = HomographyNode(
            node_id=node_id,
            frame_id=frame_id,
            parent_node_id=parent_node_id,
            H_to_parent=H_current_to_parent.copy(),
            H_to_global=H_to_global.copy(),
        )
        self._nodes[node_id] = node
        self._local_edges[node_id] = H_current_to_parent.copy()
        self._frame_ids[node_id] = frame_id
        self._parent_ids[node_id] = parent_node_id
        self._global_transforms[node_id] = H_to_global.copy()
        self._current_id = node_id
        self.transform_version += 1
        return node_id

    # ── transforms ──

    def get_transform(self, node_id: int) -> np.ndarray:
        """获取节点到全局的单应矩阵。

        Raises:
            KeyError: 节点不存在
        """
        if node_id not in self._global_transforms:
            raise KeyError(
                f"Node {node_id} not found in graph. "
                f"Available nodes: {sorted(self._frame_ids.keys())}"
            )
        return self._global_transforms[node_id].copy()

    def get_current_transform(self) -> np.ndarray:
        """获取当前节点的全局变换。无节点时抛出异常。"""
        if self._current_id < 0:
            raise KeyError("No keyframes in graph.")
        return self._global_transforms[self._current_id].copy()

    def get_parent_node_id(self, node_id: int) -> int | None:
        return self._parent_ids.get(node_id)

    def get_node(self, node_id: int) -> HomographyNode | None:
        return self._nodes.get(node_id)

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
        return len(self._nodes)

    @property
    def loop_constraints(self) -> list[LoopConstraint]:
        return list(self._loop_constraints)

    def get_frame_id(self, node_id: int) -> int | None:
        return self._frame_ids.get(node_id)

    @property
    def nodes(self) -> list[tuple[int, int, np.ndarray]]:
        """返回所有节点: [(node_id, frame_id, H_to_global), ...]"""
        return [(nid, self._frame_ids.get(nid, -1),
                 self._global_transforms.get(nid, np.eye(3)))
                for nid in sorted(self._frame_ids.keys())]

    @property
    def homography_nodes(self) -> list[HomographyNode]:
        """返回所有 HomographyNode 对象。"""
        return [self._nodes[nid] for nid in sorted(self._nodes.keys())]

    # ── internal ──

    @staticmethod
    def _validate_homography(H: np.ndarray) -> None:
        """H5: 检查矩阵有限性、形状和投影合理性。"""
        if H is None:
            raise ValueError("Homography matrix is None")
        if not isinstance(H, np.ndarray):
            raise ValueError(f"Homography is not ndarray: {type(H)}")
        if H.shape != (3, 3):
            raise ValueError(f"Homography shape {H.shape} != (3, 3)")
        if not np.all(np.isfinite(H)):
            raise ValueError("Homography contains non-finite values")
        if np.linalg.matrix_rank(H) < 3:
            raise ValueError("Homography is singular (rank < 3)")
        # 检查投影范围合理性
        test_pts = np.array([[0, 0, 1], [640, 480, 1], [320, 240, 1]]).T  # 3x3
        projected = H @ test_pts
        if np.any(np.abs(projected[2, :]) < 1e-8):
            raise ValueError("Homography maps points to infinity")
