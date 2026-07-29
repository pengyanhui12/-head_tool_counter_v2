"""HomographyGraph 单元测试"""
import numpy as np
import pytest

from core.homography_graph import HomographyGraph


def test_first_keyframe_identity():
    g = HomographyGraph()
    nid = g.add_first_keyframe(0)
    H = g.get_transform(nid)
    np.testing.assert_allclose(H, np.eye(3), atol=1e-6)
    assert g.num_keyframes == 1


def test_translation_chain():
    g = HomographyGraph()
    g.add_first_keyframe(0)
    H_step = np.array([[1, 0, 10], [0, 1, 0], [0, 0, 1]], dtype=float)
    for i in range(5):
        g.add_keyframe(i + 1, H_step)
    T = g.get_current_transform()
    # 5 steps of +10px in x => +50px
    assert abs(T[0, 2] - 50.0) < 1.0
    assert g.num_keyframes == 6


def test_optimization_not_implemented():
    g = HomographyGraph()
    with pytest.raises(NotImplementedError):
        g.optimize_homography_graph()


def test_get_transform_before_any_keyframe():
    g = HomographyGraph()
    # H2: 不存在 node_id 时不得返回单位阵 → 应该抛出异常
    with pytest.raises(KeyError):
        g.get_current_transform()
    # 同样 get_transform 对未知节点应抛 KeyError
    with pytest.raises(KeyError):
        g.get_transform(999)


def test_get_transform_unknown_node_raises():
    """get_transform 对未知节点抛 KeyError 而不是返回单位阵"""
    g = HomographyGraph()
    g.add_first_keyframe(0)
    with pytest.raises(KeyError):
        g.get_transform(999)


def test_add_keyframe_invalid_matrix():
    """H1: 无效矩阵（None、非有限、奇异）不得创建图节点"""
    g = HomographyGraph()
    g.add_first_keyframe(0)
    # None
    with pytest.raises(ValueError, match="None"):
        g.add_keyframe(1, None)
    # 非有限
    bad = np.array([[np.inf, 0, 0], [0, 1, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match="non-finite"):
        g.add_keyframe(1, bad)
    # 奇异
    bad2 = np.zeros((3, 3))
    with pytest.raises(ValueError):
        g.add_keyframe(1, bad2)


def test_parent_node_id():
    """H4: 节点必须显式记录 parent_node_id"""
    g = HomographyGraph()
    n0 = g.add_first_keyframe(0)
    assert g.get_parent_node_id(n0) is None
    H = np.array([[1, 0, 5], [0, 1, 0], [0, 0, 1]], dtype=float)
    n1 = g.add_keyframe(1, H, parent_node_id=n0)
    assert g.get_parent_node_id(n1) == n0
    n2 = g.add_keyframe(2, H)
    assert g.get_parent_node_id(n2) == n1  # 默认 parent 是上一个


def test_parent_not_found_raises():
    """H3: parent node 不存在时抛异常"""
    g = HomographyGraph()
    g.add_first_keyframe(0)
    H = np.eye(3)
    with pytest.raises(KeyError, match="not found"):
        g.add_keyframe(1, H, parent_node_id=999)
