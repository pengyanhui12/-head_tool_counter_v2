"""RECOVERY 和 HomographyGraph 测试 — 强制覆盖恢复路径"""
import numpy as np
import pytest

from core.types import (
    Frame, MatchResult, RecoveryState,
)
from core.homography_graph import HomographyGraph
from core.recovery_manager import RecoveryManager


class FailingMatcher:
    """模拟匹配失败"""
    def match(self, src, dst):
        return MatchResult(
            H_source_to_target=None,
            num_keypoints_src=5, num_keypoints_dst=10,
            num_good_matches=3, num_inliers=2,
            inlier_ratio=0.1, reprojection_error=10.0,
            occupied_quadrants_src=1, occupied_quadrants_dst=1,
            inlier_bbox_area_ratio_src=0.05, inlier_bbox_area_ratio_dst=0.05,
            valid=False, failure_reason="too_few_inliers",
        )


class SuccessMatcher:
    """模拟匹配成功"""
    def match(self, src, dst):
        return MatchResult(
            H_source_to_target=np.eye(3),
            num_keypoints_src=100, num_keypoints_dst=100,
            num_good_matches=50, num_inliers=30,
            inlier_ratio=0.6, reprojection_error=1.5,
            occupied_quadrants_src=4, occupied_quadrants_dst=4,
            inlier_bbox_area_ratio_src=0.3, inlier_bbox_area_ratio_dst=0.3,
            valid=True,
        )


def make_frame(frame_id: int):
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return Frame(frame_id=frame_id, timestamp=frame_id / 30.0, image=img,
                 sharpness_score=80.0, exposure_score=0.8)


def test_recovery_fails_returns_lost():
    """全部恢复策略失败 → LOST"""
    mgr = RecoveryManager(matcher=FailingMatcher())
    f = make_frame(1)
    result = mgr.recover(f, None)
    assert result.state == RecoveryState.LOST
    assert result.H_current_to_anchor is None


def test_recovery_bridge_succeeds():
    """bridge 恢复成功"""
    mgr = RecoveryManager(matcher=SuccessMatcher())
    f1 = make_frame(1)
    f5 = make_frame(5)
    f10 = make_frame(10)

    # 缓存 f5 作为 bridge
    mgr.cache_frame(f5)

    result = mgr.recover(f10, f1)
    assert result.state == RecoveryState.RECOVERED


def test_homography_graph_rejects_identity_recovery():
    """H1: 无效矩阵不得创建图节点"""
    g = HomographyGraph()
    g.add_first_keyframe(0)
    # None
    with pytest.raises(ValueError):
        g.add_keyframe(1, None)
    # 非有限
    bad = np.full((3, 3), np.inf)
    with pytest.raises(ValueError):
        g.add_keyframe(1, bad)


def test_homography_graph_parent_chain():
    """显式 parent 链：第二个尾部候选 parent 是第一个尾部关键帧"""
    g = HomographyGraph()
    n0 = g.add_first_keyframe(0)
    H = np.eye(3)

    # 第一个尾部 KF
    n1 = g.add_keyframe(10, H, parent_node_id=n0)
    assert g.get_parent_node_id(n1) == n0

    # 第二个尾部 KF: parent 是 n1
    n2 = g.add_keyframe(20, H, parent_node_id=n1)
    assert g.get_parent_node_id(n2) == n1


def test_graph_get_transform_unknown_node():
    """get_transform 对未知节点抛 KeyError"""
    g = HomographyGraph()
    g.add_first_keyframe(0)
    with pytest.raises(KeyError):
        g.get_transform(999)


def test_homography_validation():
    """H5: 加图前检查矩阵有限性"""
    g = HomographyGraph()
    g.add_first_keyframe(0)
    # 奇异矩阵
    bad = np.zeros((3, 3))
    with pytest.raises(ValueError):
        g.add_keyframe(1, bad)
    # 映射到无穷远
    bad2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    with pytest.raises(ValueError):
        g.add_keyframe(1, bad2)


def test_node_parent_recorded():
    """H4: 节点必须显式记录 parent_node_id"""
    g = HomographyGraph()
    n0 = g.add_first_keyframe(0)
    assert g.get_parent_node_id(n0) is None

    H = np.eye(3)
    n1 = g.add_keyframe(5, H, parent_node_id=n0)
    assert g.get_parent_node_id(n1) == n0

    n2 = g.add_keyframe(10, H)  # default parent = n1
    assert g.get_parent_node_id(n2) == n1
