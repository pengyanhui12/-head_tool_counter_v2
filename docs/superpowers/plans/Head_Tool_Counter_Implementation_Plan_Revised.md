# Head Tool Counter — 实现计划（修订版）

> 日期：2026-07-27  
> 状态：**核心算法路线已确认，可进入单机离线 MVP 实现**  
> 说明：本计划已修复上一版中的接口冲突、数据流断裂、Tracker 命名不准确、单应矩阵方向歧义、L1/L2/L3 重复更新、错误覆盖率计算、回环“伪优化”等问题。

---

## 0. 目标与范围

### 0.1 Goal

构建轨道交通作业工器具检测计数系统的单机离线 MVP，实现：

- 单目视频读取与质量筛选；
- SIFT + RANSAC 二维关键帧建图；
- YOLO 分层检测；
- 检测帧上的短期跟踪；
- 关键帧检测结果投影到全局二维地图；
- 同帧互斥、历史共现互斥和全局对象关联；
- 多帧类别融合；
- 工具计数与低置信度复核；
- 后处理覆盖率检查；
- JSON、CSV、证据帧和日志输出。

### 0.2 当前阶段不实现

以下内容不作为本轮 MVP 的阻塞项：

- 云端分片上传和 FastAPI；
- 多用户并发和 GPU 调度；
- ALIKED + LightGlue；
- 完整单应图非线性优化；
- NetVLAD 回环检索；
- BoT-SORT/ByteTrack 正式集成；
- 自动相机标定；
- 前端人工复核界面；
- 实时覆盖率提示。

---

## 1. 总体架构

本项目按 **Layer 0～Layer 4，共 5 层**构建。

| 层 | 内容 |
|---|---|
| Layer 0 | 项目骨架、配置、共享类型、日志 |
| Layer 1 | 视频读取、质量评估、帧缓存 |
| Layer 2 | 特征匹配、关键帧、单应性图、回环约束记录 |
| Layer 3 | 检测、短期跟踪、检测融合、投影、全局关联、覆盖 |
| Layer 4 | 编排、证据提取、报告、集成测试、评价 |

### 1.1 Tech Stack

- Python 3.10+
- PyTorch
- Ultralytics YOLO
- OpenCV
- NumPy
- SciPy
- scikit-learn
- PyYAML
- Shapely
- pytest

> FastAPI 暂不属于本轮 MVP 依赖。

---

## 2. 全局约束

### 2.1 场景约束

- 单目、无 IMU、无 AprilTag/ArUco；
- 工具在扫描期间保持静止；
- 工具主要平铺在同一地面平面；
- 全局地图坐标为**拼接像素坐标/任意尺度二维坐标**，不代表真实物理长度；
- 无标定 MVP 默认仅使用中央 ROI 进行几何匹配，边缘观测降权。

### 2.2 检测层级

| Level | 建议分辨率 | 执行对象 | 作用 | 进入全局地图 |
|---|---:|---|---|---|
| L1 | 960/1280 | 已接纳关键帧 | 主检测 | 是 |
| L2 | 640/768 | 每 2～3 帧 | 触发关键帧、短期跟踪 | 否 |
| L3 | 1280/切片 | 已接纳关键帧，按 ROI 触发 | 补充小目标 | 是 |

约束：

1. 所有层先统一输出 `DetectionCandidate`；
2. L2 仅用于触发和短期跟踪，不直接投影；
3. L1/L3 在同一关键帧上必须先融合，不能直接拼接；
4. 融合后只对 Tracker 更新一次；
5. 融合并分配 `track_id` 后，才生成不可变 `RawDetection`；
6. `RawDetection` 只来自已接纳关键帧上的 L1/L3。

### 2.3 关键帧

```text
候选触发
→ 几何质量接纳
→ 接纳失败时 RECOVERY
```

`KeyframeSelector` 接纳时直接返回已经验证过的 `H_current_to_previous`，Pipeline 不得重复运行同一对图像的特征匹配。

### 2.4 单应矩阵方向

全项目唯一约定：

```text
H_source_to_target:
源图像坐标 → 目标图像坐标
```

相邻关键帧：

```text
H_current_to_previous:
当前关键帧 → 上一关键帧
```

累计关系：

```python
H_current_to_global = (
    H_previous_to_global
    @ H_current_to_previous
)
```

接口和成员变量禁止使用含义不明确的 `H`、`local_H`、`transform`。

### 2.5 回环

本轮 MVP：

- 可以发现并记录回环候选；
- 可以计算回环单应约束和残差；
- 不采用线性平均或 diffusion averaging 修改单应矩阵；
- 未实现真正优化器前，不更新 `transform_version`；
- 不触发虚假的全量重投影。

真正的图优化实现后：

```text
矩阵实际发生变化
→ transform_version += 1
→ 全量重投影
→ ObjectAssociator 全量重建
```

### 2.6 状态分离

```text
confirmation_status:
TENTATIVE / CONFIRMED / UNCERTAIN / REJECTED

visibility_status:
ACTIVE / INACTIVE
```

复核标志独立存在：

```text
LIKELY_DUPLICATE
CLASS_CONFLICT
EDGE_ONLY
LOW_CONFIDENCE
MAPPING_UNSTABLE
```

### 2.7 ID

- `provisional_id`：在线处理期间使用；
- `persistent_id`：视频结束、最终对象确定后分配；
- 分配后锁定。

### 2.8 覆盖率

- 拍摄过程中只累计已映射视场；
- 操作员后处理圈定清点区域；
- 覆盖率使用整帧视场多边形计算；
- 不能使用工具框作为覆盖区域。

---

## 3. 项目目录

```text
head_tool_counter/
├── apps/
│   └── offline_scan.py
├── configs/
│   ├── pipeline.yaml
│   ├── detector.yaml
│   ├── tracker.yaml
│   ├── matcher.yaml
│   ├── associator.yaml
│   ├── coverage.yaml
│   └── camera.yaml
├── core/
│   ├── __init__.py
│   ├── types.py
│   ├── config_loader.py
│   ├── video_reader.py
│   ├── quality_evaluator.py
│   ├── frame_buffer.py
│   ├── feature_matcher.py
│   ├── loop_candidate_retriever.py
│   ├── homography_graph.py
│   ├── keyframe_selector.py
│   ├── detector.py
│   ├── detection_fusion.py
│   ├── simple_tracker.py
│   ├── global_projector.py
│   ├── global_object_map.py
│   ├── object_associator.py
│   ├── coverage_map.py
│   ├── status_panel.py
│   ├── evidence_extractor.py
│   ├── report_generator.py
│   └── session_store.py
├── models/
│   └── best.pt
├── tests/
│   ├── fixtures/
│   ├── test_video_reader.py
│   ├── test_quality_evaluator.py
│   ├── test_feature_matcher.py
│   ├── test_homography_graph.py
│   ├── test_keyframe_selector.py
│   ├── test_detector.py
│   ├── test_detection_fusion.py
│   ├── test_simple_tracker.py
│   ├── test_global_projector.py
│   ├── test_global_object_map.py
│   ├── test_object_associator.py
│   ├── test_coverage_map.py
│   ├── test_report_generator.py
│   └── test_pipeline_integration.py
├── tools/
│   ├── evaluate_sessions.py
│   └── inspect_mapping.py
├── outputs/
├── data/
├── pyproject.toml
├── requirements.txt
├── pytest.ini
├── README.md
├── .gitignore
└── environment.yml
```

---

## 4. Layer 0：项目骨架、配置与共享类型

## Task 0.1：项目骨架与依赖锁定

创建：

```text
pyproject.toml
requirements.txt
environment.yml
pytest.ini
README.md
.gitignore
head_tool_counter/__init__.py
head_tool_counter/core/__init__.py
```

Windows PowerShell 示例：

```powershell
New-Item -ItemType Directory -Force head_tool_counter\apps
New-Item -ItemType Directory -Force head_tool_counter\configs
New-Item -ItemType Directory -Force head_tool_counter\core
New-Item -ItemType Directory -Force head_tool_counter\models
New-Item -ItemType Directory -Force head_tool_counter\outputs
New-Item -ItemType Directory -Force head_tool_counter\tests
New-Item -ItemType Directory -Force head_tool_counter\tools
```

依赖：

```text
numpy
opencv-python-headless
scipy
scikit-learn
pyyaml
shapely
ultralytics
pytest
```

---

## Task 0.2：配置文件

### `configs/pipeline.yaml`

```yaml
pipeline:
  max_input_fps: 30
  l2_interval_frames: 3
  max_keyframe_interval_frames: 30
  end_window_frames: 30
  save_debug_artifacts: true
  random_seed: 42
```

### `configs/detector.yaml`

```yaml
model:
  path: models/best.pt
  device: cuda:0

levels:
  L1:
    imgsz: 1280
    conf: 0.15
    iou: 0.65
  L2:
    imgsz: 640
    conf: 0.10
    iou: 0.65
  L3:
    imgsz: 1280
    conf: 0.10
    iou: 0.65

l3:
  enabled: false
  min_box_area_ratio: 0.001
  low_confidence_upper: 0.35
  roi_margin_ratio: 0.20
```

### `configs/tracker.yaml`

```yaml
tracker:
  type: simple_hungarian
  max_missed_detection_frames: 5
  lost_reactivation_frames: 10
  min_iou: 0.20
  max_center_distance_ratio: 0.20
  iou_weight: 0.60
  center_weight: 0.40
```

### `configs/matcher.yaml`

```yaml
matcher:
  mode: sift
  ratio_test: 0.75
  ransac_threshold_px: 3.0
  min_good_matches: 20
  min_inliers: 30
  min_inlier_ratio: 0.30
  max_reprojection_error_px: 3.0
  min_occupied_quadrants: 3
  min_inlier_bbox_area_ratio: 0.15
  roi_center_ratio: 0.70
  max_projected_area_ratio: 10.0
  min_projected_area_ratio: 0.10
```

### `configs/associator.yaml`

```yaml
association:
  max_position_distance_px: 120.0
  position_weight: 0.55
  overlap_weight: 0.20
  size_weight: 0.10
  class_weight: 0.15
  max_cost: 0.75
  min_observations_confirmed: 3
  min_keyframes_confirmed: 2
  min_top_class_ratio: 0.60
  max_votes_per_track: 3

class_compatibility:
  flashlight:
    - flashlight
    - telescopic_voltage_detector
  telescopic_voltage_detector:
    - telescopic_voltage_detector
    - flashlight
```

### `configs/coverage.yaml`

```yaml
coverage:
  grid_resolution: 100
  minimum_valid_polygon_area: 100.0
  target_coverage_ratio: 0.95
```

### `configs/camera.yaml`

```yaml
camera:
  calibration_mode: none
  profile_id: uncalibrated
  central_roi_ratio: 0.70
  allow_edge_detection: true
  allow_edge_mapping: false
  edge_observation_weight: 0.25
  distortion_warning: true
```

---

## Task 0.3：共享类型 `core/types.py`

```python
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
```

---

## 5. Layer 1：视频读取、质量评估和帧缓存

## Task 1.1：VideoReader

### Interface

```python
VideoReader.read() -> Iterator[Frame]
```

### 参考实现

```python
from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np

from core.types import Frame


class VideoReader:
    def __init__(self, path: str, max_fps: float = 30.0):
        if max_fps <= 0:
            raise ValueError("max_fps must be positive")
        self.path = path
        self.max_fps = float(max_fps)

    def read(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise OSError(f"Cannot open video: {self.path}")

        try:
            src_fps = float(cap.get(cv2.CAP_PROP_FPS))
            if not np.isfinite(src_fps) or src_fps <= 0:
                src_fps = 30.0

            output_interval = 1.0 / self.max_fps
            next_output_time = 0.0
            source_frame_id = 0

            while True:
                ok, image = cap.read()
                if not ok:
                    break

                timestamp = source_frame_id / src_fps
                if timestamp + 1e-9 >= next_output_time:
                    yield Frame(
                        frame_id=source_frame_id,
                        timestamp=timestamp,
                        image=image,
                    )
                    next_output_time += output_interval

                source_frame_id += 1
        finally:
            cap.release()
```

### Tests

- `test_video_reader_yields_frames`
- `test_video_reader_rejects_non_positive_max_fps`
- `test_fps_cap_60_to_24`
- `test_invalid_fps_metadata_fallback`
- `test_cannot_open_video`

---

## Task 1.2：QualityEvaluator

```python
import cv2
import numpy as np

from core.types import Frame


class QualityEvaluator:
    def __init__(self, sharpness_threshold: float = 50.0):
        self.sharpness_threshold = sharpness_threshold

    def evaluate(self, frame: Frame) -> Frame:
        image = frame.image
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        frame.sharpness_score = float(
            cv2.Laplacian(gray, cv2.CV_64F).var()
        )
        frame.exposure_score = float(
            np.clip(gray.mean() / 127.5, 0.0, 1.0)
        )
        return frame

    def is_acceptable(self, frame: Frame) -> bool:
        return frame.sharpness_score >= self.sharpness_threshold
```

---

## Task 1.3：FrameBuffer

设计要求：

- 保存灰度缩放图；
- 保存 `source_frame_id`，不保存字节偏移；
- 最大缓存建议 30～60 帧；
- 原始彩色帧按源帧号或时间戳回读。

接口：

```python
push(frame: Frame) -> None
get_range(start_id: int, end_id: int) -> list[BufferedFrame]
get(frame_id: int) -> BufferedFrame | None
```

---

## 6. Layer 2：特征匹配、关键帧和单应性图

## Task 2.1：FeatureMatcher

接口：

```python
match(
    source_image: np.ndarray,
    target_image: np.ndarray,
) -> MatchResult
```

必须实现：

1. 支持 BGR 和灰度输入；
2. KNN 匹配不足两个邻居时跳过；
3. 实际使用中央 ROI 掩膜；
4. 空间分布只基于 RANSAC 内点；
5. 同时检查源图和目标图；
6. 所有质量阈值真正参与 `valid`；
7. 单应矩阵按 `H[2,2]` 归一化；
8. 使用实际图像四角检查投影退化；
9. 检查齐次分母接近 0；
10. 不使用单位正方形评估整幅图像变形。

核心判定：

```python
quality_valid = (
    num_good_matches >= min_good_matches
    and num_inliers >= min_inliers
    and inlier_ratio >= min_inlier_ratio
    and reprojection_error <= max_reprojection_error
    and occupied_quadrants_src >= min_occupied_quadrants
    and occupied_quadrants_dst >= min_occupied_quadrants
    and inlier_bbox_area_ratio_src >= min_bbox_area_ratio
    and inlier_bbox_area_ratio_dst >= min_bbox_area_ratio
)
```

空间分布采用：

```text
至少3个象限包含足够内点
+
内点外接框面积占图像面积达到阈值
```

---

## Task 2.2：LoopCandidateRetriever

MVP 使用间隔采样，只输出候选，不负责优化。

---

## Task 2.3：HomographyGraph

数据：

```python
_local_edges[node_id] = H_current_to_previous
_global_transforms[node_id] = H_current_to_global
_loop_constraints: list[LoopConstraint]
```

接口：

```python
add_first_keyframe(frame_id: int) -> int

add_keyframe(
    frame_id: int,
    H_current_to_previous: np.ndarray,
) -> int

get_transform(node_id: int) -> np.ndarray

get_current_transform() -> np.ndarray

add_loop_constraint(
    constraint: LoopConstraint,
) -> None
```

MVP：

```python
def optimize_homography_graph(self) -> None:
    raise NotImplementedError(
        "Homography graph optimization is not implemented "
        "in MVP. Loop constraints are recorded only."
    )
```

未实际修改矩阵时，`transform_version` 不变。

---

## Task 2.4：KeyframeSelector

接口：

```python
evaluate(
    frame: Frame,
    previous_keyframe: Frame | BufferedFrame | None,
    trigger_context: KeyframeTriggerContext,
) -> KeyframeResult
```

执行逻辑：

1. 第一帧直接接纳；
2. 无触发信号返回 `SKIP`；
3. 有触发信号时执行 FeatureMatcher；
4. `MatchResult.valid=True` 才接纳；
5. 结果直接携带 `H_current_to_previous`；
6. Pipeline 不重复匹配；
7. 失败时从 FrameBuffer 搜索过渡帧；
8. 再失败时尝试历史关键帧重定位；
9. 全部失败返回 `LOST`。

结束窗口采用 `deque(maxlen=end_window_frames)`，视频结束后选择最佳 1～2 帧。

---

## 7. Layer 3：检测、跟踪、融合、投影和关联

## Task 3.1：Detector

所有层统一返回：

```python
list[DetectionCandidate]
```

接口：

```python
detect(
    image: np.ndarray,
    level: str,
    frame_id: int,
    regions: list[tuple[int, int, int, int]] | None = None,
) -> list[DetectionCandidate]
```

约束：

- L1：全图精检；
- L2：全图巡检；
- L3：明确 ROI 高分辨率复检；
- Detector 不创建 `RawDetection`；
- `RawDetection` 在融合和跟踪后创建。

---

## Task 3.2：DetectionFusion

接口：

```python
fuse(
    l1: list[DetectionCandidate],
    l3: list[DetectionCandidate],
) -> list[DetectionCandidate]
```

MVP：

- 同类别 NMS；
- L3 ROI 框转换回全图坐标；
- 兼容类别冲突保留高置信度并记录冲突；
- 禁止直接 `l1.extend(l3)`。

---

## Task 3.3：SimpleDetectionTracker

> 当前只是简化检测跟踪器，不称为 ByteTrack 或 BoT-SORT。

接口：

```python
preview(
    detections: list[DetectionCandidate],
) -> TrackerPreview

update(
    detections: list[DetectionCandidate],
) -> list[TrackedDetection]

get_active_tracks() -> list[Track]

track_quality_dropped(track_id: int) -> bool
```

规则：

- `preview()` 不修改状态；
- 每帧最多一次 `update()`；
- 使用类别兼容门控；
- 使用 IoU + 中心距离代价；
- 使用 Hungarian 匹配；
- 新建轨迹计入本帧已更新集合；
- lost 轨迹在限定窗口内可重新激活；
- 默认 IoU 阈值从 0.2～0.5 实测；
- 新目标指无法与现有轨迹匹配的检测。

同一帧：

```text
L2
→ tracker.preview(L2)
→ 关键帧判定

若关键帧接纳：
    L1/L3融合
    → tracker.update一次
否则：
    tracker.update(L2)
```

---

## Task 3.4：GlobalProjector

接口：

```python
project(
    detection: RawDetection,
    H_keyframe_to_global: np.ndarray,
    transform_version: int,
) -> GlobalDetection
```

设计：

- 四角点构成四边形；
- 中心点单独投影；
- 中心点不是多边形顶点；
- 计算 Polygon 质心和面积；
- 检查齐次分母；
- 边缘框降低 `edge_quality`；
- L2 不调用该模块。

---

## Task 3.5：GlobalObjectMap

接口：

```python
create_object(detection: GlobalDetection) -> GlobalObject
get_all() -> list[GlobalObject]
get_by_provisional(provisional_id: str) -> GlobalObject | None
assign_persistent_ids() -> None
set_confirmation(...) -> None
set_visibility(...) -> None
add_review_flag(...) -> None
```

规则：

- CONFIRMED 离开画面只变为 INACTIVE；
- 不得因长期未见改为 REJECTED；
- 仅 TENTATIVE/UNCERTAIN 可在结束阶段拒绝；
- 最终计数依据 `persistent_id + CONFIRMED`。

---

## Task 3.6：ObjectAssociator

接口：

```python
ingest_frame(
    frame_id: int,
    global_detections: list[GlobalDetection],
) -> list[str]

rebuild_all(
    global_detections: list[GlobalDetection],
    map_version: int,
) -> RebuildResult

final_review() -> None
```

关键规则：

1. 每帧整批调用一次；
2. 同帧互斥和共现关系整批处理；
3. 使用类别兼容矩阵；
4. `C_class` 不使用置信度差；
5. `rebuild_all()` 只接收已重投影结果；
6. Pipeline 负责 RawDetection → GlobalDetection；
7. 回环旧关联只能作为软先验；
8. 同帧互斥、历史共现和人工确认可作为硬约束。

代价：

```text
C =
w_position × C_position
+
w_overlap × C_overlap
+
w_size × C_size
+
w_class × C_class
```

类别：

```text
同类别：0
兼容类别：有限惩罚
不兼容类别：∞
```

类别投票：

```text
w =
p_det × q_sharp × q_map × q_edge × q_size
```

每个 track 最多 3 票。

疑似重复检查在 GlobalObject 层执行，不直接对全部原始观测点做 DBSCAN。

---

## Task 3.7：CoverageMap

接口：

```python
update(
    frame_id: int,
    projected_fov_polygon: Polygon,
) -> None

get_coverage(
    region_polygon: Polygon,
) -> float
```

每个关键帧只更新一次，投影整帧四角，不投影工具框。

---

## 8. Layer 4：编排、证据、报告和评价

## Task 4.1：StatusPanel

```python
from dataclasses import dataclass, field


@dataclass
class StatusPanel:
    class_counts: dict[str, int] = field(default_factory=dict)
    confirmation_counts: dict[str, int] = field(
        default_factory=dict
    )
    visibility_counts: dict[str, int] = field(
        default_factory=dict
    )
    review_flag_counts: dict[str, int] = field(
        default_factory=dict
    )
    total_frames: int = 0
    accepted_keyframes: int = 0
    mapping_state: str = "initializing"
    transform_version: int = 0
    map_version: int = 0
```

---

## Task 4.2：EvidenceExtractor

```text
选择最佳 GlobalDetection
→ 回读原视频
→ 绘制 bbox、persistent_id 和类别
→ 保存 JPG/PNG
```

接口：

```python
select_best(obj: GlobalObject) -> GlobalDetection | None

extract(
    video_path: str,
    objects: list[GlobalObject],
    output_dir: str,
) -> dict[str, str]
```

---

## Task 4.3：ReportGenerator

输出：

```text
report.json
report.csv
evidence/*.jpg
```

JSON 序列化支持：

- dataclass；
- Enum；
- NumPy 数组和数值；
- set；
- Shapely 对象；
- Path。

CSV 字段：

```text
persistent_id
class_name
confirmation_status
visibility_status
confidence
observation_count
keyframe_count
track_count
centroid_x
centroid_y
review_flags
best_frame_id
evidence_path
```

---

## Task 4.4：SessionStore

保存：

```text
原始检测
关键帧
匹配结果
单应矩阵
GlobalDetection
GlobalObject
配置版本
模型版本
软件版本
日志
```

用于复现、调试和后续云端会话扩展。

---

## Task 4.5：offline_scan.py

正确顺序：

```python
for frame in reader.read():
    frame = quality.evaluate(frame)

    if not quality.is_acceptable(frame):
        continue

    frame_buffer.push(to_buffered_frame(frame))
    end_window.append(frame)

    l2_candidates = []

    if should_run_l2(frame.frame_id):
        l2_candidates = detector.detect(
            image=frame.image,
            level="L2",
            frame_id=frame.frame_id,
        )

    preview = tracker.preview(l2_candidates)

    trigger_context = KeyframeTriggerContext(
        max_interval_reached=(
            frame.frame_id - last_keyframe_frame_id
            >= max_interval
        ),
        l2_new_unmatched_detection=(
            preview.l2_new_unmatched_detection
        ),
        track_quality_drop=preview.track_quality_drop,
        l3_required=should_trigger_l3(l2_candidates),
        l3_regions=select_l3_regions(l2_candidates),
    )

    keyframe_result = selector.evaluate(
        frame=frame,
        previous_keyframe=last_keyframe,
        trigger_context=trigger_context,
    )

    if keyframe_result.decision == KeyframeDecision.ACCEPTED:
        if last_keyframe is None:
            keyframe_id = graph.add_first_keyframe(
                frame_id=frame.frame_id,
            )
        else:
            keyframe_id = graph.add_keyframe(
                frame_id=frame.frame_id,
                H_current_to_previous=(
                    keyframe_result.H_current_to_previous
                ),
            )

        H_keyframe_to_global = graph.get_transform(keyframe_id)

        l1_candidates = detector.detect(
            image=frame.image,
            level="L1",
            frame_id=frame.frame_id,
        )

        l3_candidates = []

        if trigger_context.l3_required:
            l3_candidates = detector.detect(
                image=frame.image,
                level="L3",
                frame_id=frame.frame_id,
                regions=trigger_context.l3_regions,
            )

        fused_candidates = detection_fusion.fuse(
            l1=l1_candidates,
            l3=l3_candidates,
        )

        tracked = tracker.update(fused_candidates)

        raw_detections = build_raw_detections(
            tracked_detections=tracked,
            keyframe_id=keyframe_id,
            sharpness=frame.sharpness_score,
            mapping_quality=frame.mapping_quality,
        )

        global_detections = [
            projector.project(
                detection=raw_detection,
                H_keyframe_to_global=H_keyframe_to_global,
                transform_version=graph.transform_version,
            )
            for raw_detection in raw_detections
        ]

        associator.ingest_frame(
            frame_id=frame.frame_id,
            global_detections=global_detections,
        )

        projected_fov = projector.project_frame_corners(
            image_shape=frame.image.shape,
            H_keyframe_to_global=H_keyframe_to_global,
        )

        coverage_map.update(
            frame_id=frame.frame_id,
            projected_fov_polygon=projected_fov,
        )

        last_keyframe = frame
        last_keyframe_frame_id = frame.frame_id

    elif l2_candidates:
        tracker.update(l2_candidates)
```

视频结束：

```text
结束窗口补选关键帧
→ final_review
→ assign_persistent_ids
→ EvidenceExtractor
→ ReportGenerator
```

---

## Task 4.6：端到端集成测试

至少覆盖：

1. 相机固定；
2. 缓慢横向扫描；
3. 离开后返回同一工具；
4. 两件同类工具靠近；
5. 某些帧漏检；
6. 工具位于边缘；
7. 模糊帧和 RECOVERY；
8. L1/L3 重复检测；
9. 有检测但单应失败；
10. 空白区域覆盖；
11. 同帧多个同类目标不合并；
12. CONFIRMED 离开画面仍计数。

---

## Task 4.7：Evaluation Harness

指标：

- 工具级召回率；
- 分类准确率；
- 重复计数率；
- 错误合并率；
- 单类数量准确率；
- 单次完全正确率；
- UNCERTAIN 比例；
- 关键帧接纳率；
- RECOVERY 成功率；
- 建图有效率；
- 处理时间；
- 内存峰值；
- 显存峰值。

---

## 9. 实施顺序

```text
Layer 0
0.1 项目骨架和依赖
0.2 配置文件
0.3 Types和ConfigLoader

Layer 1
1.1 VideoReader
1.2 QualityEvaluator
1.3 FrameBuffer

Layer 2
2.1 FeatureMatcher
2.2 LoopCandidateRetriever
2.3 HomographyGraph
2.4 KeyframeSelector

Layer 3
3.1 Detector
3.2 DetectionFusion
3.3 SimpleDetectionTracker
3.4 GlobalProjector
3.5 GlobalObjectMap
3.6 ObjectAssociator
3.7 CoverageMap

Layer 4
4.1 StatusPanel
4.2 EvidenceExtractor
4.3 ReportGenerator
4.4 SessionStore
4.5 offline_scan
4.6 Integration Test
4.7 Evaluation Harness
```

---

## 10. 依赖图

```text
types/configs
    ↓
video_reader
quality_evaluator
frame_buffer
    ↓
feature_matcher
keyframe_selector
homography_graph
    ↓
detector
detection_fusion
simple_tracker
    ↓
global_projector
global_object_map
object_associator
coverage_map
    ↓
offline_scan
evidence_extractor
report_generator
evaluation
```

---

## 11. 后续任务

| ID | 内容 | 前提 |
|---|---|---|
| 5.1 | FastAPI 云端分片上传 | 离线 Pipeline 稳定 |
| 5.2 | ALIKED + LightGlue | SIFT 基线完成 |
| 5.3 | 真正的单应图非线性优化 | 回环约束稳定 |
| 5.4 | NetVLAD/全局描述子 | 回环候选增加 |
| 5.5 | Panorama 和覆盖热力图 | 全局图稳定 |
| 5.6 | 前端区域圈定和复核 | 报告接口稳定 |
| 5.7 | ByteTrack/BoT-SORT 对比 | Simple Tracker 有基线 |
| 5.8 | 相机标定和 CameraProfile | 设备型号确定 |
| 5.9 | L3 patch inference (high-res crop on small regions) | 3.1 |
| 5.10 | Per-class size priors for L2 trigger tuning | 3.1 |
| 5.11 | Tool-specific appearance ReID | 3.3 |

---

## 12. MVP 完成判据

### 功能判据

- 能读取完整视频；
- 能生成关键帧和全局单应链；
- 能将关键帧工具投影至统一地图；
- 同一工具重新出现时优先关联原对象；
- 同一帧多个工具绝不合并；
- 输出 JSON、CSV 和证据图片；
- 对低置信度、疑似重复目标给出复核标志；
- 有完整集成测试和评价脚本。

### 初始性能目标

```text
工具级召回率 ≥ 95%
重复计数率 ≤ 3%
错误合并率 ≤ 2%
人工复核率 ≤ 15%
单次完全正确率 ≥ 85%
```

> 以上为原型目标，必须通过真实视频重新定标。

### 开发原则

> 先证明“同一工具在移动相机视频中可以稳定只计一次”，再增加 LightGlue、完整回环优化、相机标定和云端系统。
