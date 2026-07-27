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
    H = g.get_current_transform()
    np.testing.assert_allclose(H, np.eye(3))
