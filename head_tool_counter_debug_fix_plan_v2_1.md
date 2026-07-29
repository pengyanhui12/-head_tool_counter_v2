# Head Tool Counter Debug 问题修复计划 V2

> 仓库：`pengyanhui12/head_tool_counter`  
> 当前统一基线：本地代码与 GitHub `master` 一致  
> GitHub 默认分支：`master`  
> 当前基线提交：`23c0871f35a2e4ea7c921bff8ee09c377b72cbf2`  
> 依据：当前 GitHub 源码、用户提供的 `debug_pipeline.py`、完整 Debug 日志、第一版 Agent 修改任务书。  
> 本轮禁止引入 LightGlue、BoT-SORT 等大型替代模块。先证明当前 `SIFT + SimpleDetectionTracker + GlobalObjectMap` 的逻辑正确、可测试、可解释。

---

## 1. 本轮目标

本轮不是继续堆功能，也不是让某一段视频“恰好输出正确数量”。必须解决：

1. 未运行 L2 时仍以空列表调用 `tracker.preview()`，可能持续触发 `QUAL_DROP`；
2. 大量连续帧被接纳为关键帧，关键帧机制退化为逐帧处理；
3. 同一 `track_id` 在多个 `GlobalObject` 间反复重绑定；
4. 一个 `GlobalObject` 在同一 `frame_id` 接收多个检测；
5. 后处理合并后出现 `observation_count > unique frame count`；
6. 大量高观测对象被设为 `REJECTED`，但没有明确原因和合并去向；
7. 最终结果依赖激进 `final_review()` 压缩，而不是在线关联稳定；
8. 质量不合格帧直接 `continue`，Tracker 时间状态没有推进；
9. RECOVERY 在 Debug 中没有触发，未被真实验证；
10. `H_current_to_anchor` 可能被错误当作 `H_current_to_previous`；
11. 尾部关键帧连续加入时，匹配参考帧与图中 parent 可能不一致；
12. Debug 脚本与正式 Pipeline 仍可能采用不同入口和配置加载方式；
13. 配置加载分散，Debug 运行时的有效阈值需要明确记录；
14. JSON、CSV、控制台和 persistent ID 口径不统一；
15. Debug 输出目录可能混入上一次结果；
16. 性能统计没有分离特征匹配和图片写盘。

完成后必须满足：

```text
track_binding_conflicts == 0
same_frame_duplicate_observations == 0
unexplained_rejected_objects == 0
fake_identity_recovery == 0
report_count_inconsistency == 0
```

---

## 2. 统一基线确认

用户已确认：

```text
本地代码与 GitHub master 一致
```

因此，本轮不需要处理本地与远端的版本差异，也不应把“尚未推送”作为问题来源。

当前统一基线：

```text
repository: pengyanhui12/head_tool_counter
branch: master
commit: 23c0871f35a2e4ea7c921bff8ee09c377b72cbf2
```

Agent 开始前只需记录可复现信息：

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git log --oneline --decorate -5
pytest tests/ -q
```

要求：

1. `git status --short` 应为空；
2. 记录起始 SHA 和修改前测试结果；
3. 从该统一基线新建修复分支：
   ```bash
   git switch -c fix/debug-correctness-v2
   ```
4. 保存：
   ```text
   debug_output/baseline/git_state.txt
   debug_output/baseline/pytest_before.txt
   debug_output/baseline/effective_config_before.yaml
   ```
5. 不再检查或处理不存在的本地/远程差异；
6. Debug 日志、正式 Pipeline 和后续测试均以该统一基线为准。

---

## 3. Debug 结果确认的问题

### 3.1 关键帧风暴

日志后半段连续出现：

```text
F0102 ACCEPTED ... QUAL_DROP
F0103 ACCEPTED ... QUAL_DROP
F0104 ACCEPTED ... QUAL_DROP
...
```

说明 `track_quality_drop` 变成持续信号，关键帧失去稀疏性。

### 3.2 track-object 重绑定

日志多次出现：

```text
track_id 33 was bound to P-0010, now re-binding to P-0008
track_id 33 was bound to P-0008, now re-binding to P-0010
```

警告后继续覆盖映射是错误行为。

### 3.3 同帧观测冲突

部分对象 `observation_count` 明显大于 `keyframe_count`。当前设计中，一个对象在同一帧最多只能接收一个融合检测。该异常表示：

- 在线关联把同帧多个检测写入同一对象；或
- final review 合并了同帧共现对象。

### 3.4 结果依赖大量 REJECTED

Debug 汇总约为：

```text
provisional objects: 44
reportable objects: 15
confirmed: 12
tentative: 3
rejected: 29
```

最终 15 个不能证明在线关联稳定，必须解释每个对象为何创建、匹配、合并和拒绝。

### 3.5 RECOVERY 没有被验证

```text
recovery_attempts = 0
```

必须通过 mock matcher 强制覆盖失败与恢复路径。

---

## 4. 必须建立的系统不变量

### 4.1 Tracker

```text
T1. 一次 update 中，一个 track 最多匹配一个 detection。
T2. 一个 detection 最多分配给一个 track。
T3. active、inactive、lost 三个集合互斥。
T4. lost 不参与普通重激活。
T5. 未运行检测不等于检测结果为空。
T6. missed 必须按真实 frame_id 差或时间戳推进。
```

### 4.2 ObjectAssociator

```text
O1. 一个 logical_track_key 在未发生合法合并前只能绑定一个对象。
O2. 一个对象在同一 frame_id 最多一个 observation。
O3. 同一帧不同融合检测必须对应不同对象。
O4. 曾同帧共现的对象永久禁止自动合并。
O5. 合并前两个对象的 observation frame 集合不得重叠。
O6. 每个 REJECTED 必须有 rejected_reason。
O7. 合并产生的 REJECTED 必须有 merged_into_id。
O8. observation_count == len(observations)。
O9. 每个对象 observation frame_id 唯一。
```

### 4.3 Homography

```text
H1. 无可靠矩阵不得创建图节点。
H2. 不存在 node_id 时不得返回单位阵。
H3. H_current_to_anchor 不得作为 H_current_to_previous。
H4. 节点必须显式记录 parent_node_id。
H5. 加图前必须检查矩阵有限性和投影范围。
```

### 4.4 报告

```text
R1. total_objects = confirmed + tentative + uncertain。
R2. rejected 不进入 total_objects。
R3. JSON、CSV、控制台、API 使用同一个 reportable_objects。
R4. persistent_id 只分配给最终 reportable objects。
R5. rejected 保留 provisional_id 和完整审计信息。
```

---

## 5. Phase 1：先增加可观测性

新增 `core/debug_events.py`：

```python
@dataclass
class DebugEvent:
    event_type: str
    frame_id: int | None
    payload: dict
```

输出：

```text
debug_output/<run_id>/events.jsonl
```

必须记录：

```text
FRAME_QUALITY
L2_NOT_RUN
L2_RESULT
TRACK_PREVIEW
TRACK_MATCH
TRACK_STATE_CHANGE
TRACK_BIND
TRACK_BIND_CONFLICT
KEYFRAME_TRIGGER
KEYFRAME_ACCEPT
KEYFRAME_SKIP
KEYFRAME_RECOVERY
HOMOGRAPHY_MATCH
OBJECT_CREATE
OBJECT_MATCH_BY_TRACK
OBJECT_MATCH_BY_COST
OBJECT_SAME_FRAME_BLOCK
OBJECT_MERGE_ALLOWED
OBJECT_MERGE_BLOCKED
OBJECT_REJECT
REPORT_SUMMARY
```

统计至少包括：

```python
stats = {
    "l2_run_frames": 0,
    "l2_empty_frames": 0,
    "keyframe_trigger_quality_drop": 0,
    "keyframe_accepted": 0,
    "keyframe_rejected_by_cooldown": 0,
    "consecutive_keyframe_max_run": 0,
    "track_binding_conflicts": 0,
    "same_frame_duplicate_observations": 0,
    "merge_blocked_by_cooccurrence": 0,
    "merge_blocked_by_frame_overlap": 0,
    "objects_created": 0,
    "objects_matched_by_track": 0,
    "objects_matched_by_cost": 0,
    "objects_merged": 0,
    "unexplained_rejected_objects": 0,
    "recovery_attempts": 0,
    "recovery_success": 0,
    "recovery_failed": 0,
}
```

每次运行使用独立目录：

```text
debug_output/20260730_011200_<git_short_sha>/
```

保存：

```text
run_metadata.json
effective_config.yaml
events.jsonl
frame_stats.csv
keyframe_stats.csv
association_events.csv
object_lifecycle.json
reports/report.json
reports/report.csv
```

性能计时拆分：

```text
video_decode_ms
quality_ms
l2_inference_ms
tracker_preview_ms
keyframe_decision_ms
feature_match_ms
l1_inference_ms
l3_inference_ms
fusion_ms
tracker_update_ms
projection_ms
association_ms
debug_image_write_ms
final_review_ms
report_ms
```

分别输出核心算法 FPS 和包含 Debug I/O 的总 FPS。

---

## 6. Phase 2：修复 L2 和关键帧风暴

修改：

```text
apps/offline_scan.py
debug_pipeline.py
core/simple_tracker.py
core/keyframe_selector.py
configs/pipeline.yaml
```

显式区分：

```python
l2_was_run = False
l2_candidates = None
```

运行后：

```python
l2_was_run = True
l2_candidates = detector.detect(...)
```

语义：

```text
None：本帧没运行检测，不更新检测质量和 detection missed。
[]：运行了检测但为空，应推进 detection missed。
非空：有检测。
```

未运行 L2 时不得按普通空检测调用 preview：

```python
if l2_was_run:
    preview = tracker.preview(l2_candidates, frame.frame_id)
else:
    preview = TrackerPreview.no_detection_run()
```

`track_quality_drop` 改为边沿触发，而不是持续触发：

```yaml
tracker:
  quality_drop_trigger_ratio: 0.70
  quality_drop_rearm_ratio: 0.85
  quality_drop_min_history: 5
```

规则：

1. 只有本帧运行 L2 且 track 得到新检测时计算；
2. 从正常跨到下降时触发一次；
3. 连续下降期间不重复触发；
4. 恢复到 rearm ratio 后才能再次触发。

增加关键帧冷却：

```yaml
pipeline:
  min_keyframe_interval_frames: 5
  emergency_keyframe_interval_frames: 2
```

普通 `QUAL_DROP` 和 `NEW_DET` 必须满足最小间隔。首帧、max interval 和可靠 recovery 可例外。

测试：

- L2 未运行不产生 QUAL_DROP；
- L2 空检测推进 missed；
- 质量下降只触发一次；
- cooldown 阻止逐帧关键帧；
- max interval 正常触发；
- 最大连续关键帧长度受控。

---

## 7. Phase 3：修复 Tracker 状态和时间

严格区分：

```python
active_tracks = {tid: t for tid, t in tracks.items() if t.state == "active"}
inactive_tracks = {tid: t for tid, t in tracks.items() if t.state == "inactive"}
lost_tracks = {tid: t for tid, t in tracks.items() if t.state == "lost"}
```

修复类别代价：

```python
def _class_cost(self, track: Track, det: DetectionCandidate) -> float:
    if track.class_id == det.class_id:
        return 0.0
    if det.class_name in self._class_compat.get(track.class_name, []):
        return 0.5
    return float("inf")
```

修改 update：

```python
def update(self, detections, frame_id: int):
    ...
```

Track 增加：

```python
last_update_frame_id: int
last_detection_frame_id: int
```

按真实 frame gap 推进状态。

质量不适合建图的帧不能完全跳过，应调用：

```python
tracker.advance_frame(frame_id)
```

同时区分：

```text
frames_since_detection
detection_opportunities_missed
```

只有真正运行检测但为空时增加后者。

inactive 使用独立重激活门控：

```yaml
tracker:
  inactive_min_iou: 0.30
  inactive_max_center_distance_ratio: 0.12
```

返回前检查 track ID 和 detection index 唯一。

可为 Track 增加：

```python
generation: int = 0
logical_key = (track_id, generation)
```

---

## 8. Phase 4：禁止 track-object 静默重绑定

新增：

```python
def _bind_track_to_object(logical_track_key, object_id):
    existing = self._track_to_object.get(logical_track_key)
    if existing is None:
        self._track_to_object[logical_track_key] = object_id
        return
    if existing == object_id:
        return
    raise TrackBindingConflict(...)
```

不得 warning 后覆盖。

发生冲突时：

1. 不更新旧对象；
2. 不覆盖绑定；
3. 标记涉及对象 `TRACK_CONFLICT`；
4. 保存 frame、old object、candidate object、class、distance、cost、bbox；
5. Debug 模式抛异常；
6. 普通模式可新建 logical track segment，但不得污染旧对象。

匈牙利匹配成功后必须立即绑定：

```python
self._update_object(obj, gd)
self._bind_track_to_object(gd.logical_track_key, obj.provisional_id)
```

对象合法合并后才能迁移 secondary 的 track 映射。

测试：

- 首次绑定；
- 同对象重复绑定；
- 不同对象重绑定失败；
- 匈牙利匹配后映射存在；
- 下一帧走强关联；
- 冲突不污染对象。

---

## 9. Phase 5：一个对象同帧最多一个 observation

添加：

```python
def _has_observation_in_frame(obj, frame_id):
    return any(obs.frame_id == frame_id for obs in obj.observations)
```

任何 `_update_object()` 前检查。

在 `ingest_frame()` 中维护：

```python
assigned_object_ids = set()
assigned_detection_indices = set()
```

track 强关联也必须遵守帧级一对一。

每帧结束检查受影响对象：

```python
frame_ids = [obs.frame_id for obs in obj.observations]
assert len(frame_ids) == len(set(frame_ids))
```

对象合并前：

```python
primary_frames = {obs.frame_id for obs in primary.observations}
secondary_frames = {obs.frame_id for obs in secondary.observations}
if primary_frames & secondary_frames:
    block_merge("observation_frame_overlap")
```

同帧互斥应在最终分配后记录。L1/L3 已融合后的不同 detection，只要最终属于不同对象，就永久 `cannot_merge`。不再要求同类、IoU=0 或距离足够远。

---

## 10. Phase 6：重写 final_review 和 REJECTED 审计

第一阶段禁用：

```text
同类 + centroid < 30 px -> 自动合并
```

改为只标记 `LIKELY_DUPLICATE`。

shared track 也不能无条件硬合并。至少要求：

```text
同类别或兼容
无同帧共现
observation frame 无重叠
track 没有 conflict
空间轨迹连续
mapping quality 达标
```

新增：

```python
@dataclass
class MergeAudit:
    primary_id: str
    secondary_id: str
    decision: str
    reason: str
    shared_track_keys: list
    position_distance: float | None
    normalized_distance: float | None
    overlapping_frame_ids: list[int]
    co_occurred: bool
```

`GlobalObject` 增加：

```python
rejected_reason: str | None = None
merged_into_id: str | None = None
rejection_evidence: dict = field(default_factory=dict)
```

任何 REJECTED 必须有原因。合并产生：

```python
secondary.rejected_reason = "merged_duplicate"
secondary.merged_into_id = primary.provisional_id
```

final review 后执行 `validate_object_map()`，检查：

- rejected reason；
- merged target；
- observation frame 唯一；
- track key 是否绑定多个活动对象；
- persistent ID 是否只属于 reportable 对象。

---

## 11. Phase 7：RECOVERY 和 HomographyGraph

禁止单位阵降级。无可靠匹配时：

```text
不加图节点
不生成 GlobalDetection
不更新 last_keyframe
不进入对象关联
```

图节点显式 parent：

```python
@dataclass
class HomographyNode:
    node_id: int
    frame_id: int
    parent_node_id: int | None
    H_to_parent: np.ndarray
    H_to_global: np.ndarray
```

接口：

```python
graph.add_keyframe(
    frame_id=...,
    parent_node_id=...,
    H_current_to_parent=...,
)
```

RecoveryResult：

```python
@dataclass
class RecoveryResult:
    state: RecoveryState
    anchor_node_id: int | None
    H_current_to_anchor: np.ndarray | None
    match_result: MatchResult | None
```

加入图时必须使用 anchor node。

mapping quality 不得硬编码 0.5，应综合：

```text
inlier ratio
reprojection error
inlier spatial coverage
projected area sanity
```

`get_transform()` 对未知节点抛 `KeyError`。

测试强制覆盖：

- current 到 previous 失败；
- bridge 恢复；
- history anchor 恢复；
- 全部失败 LOST；
- LOST 不投影；
- current-to-anchor 组合方向正确。

---

## 12. Phase 8：尾部关键帧

尾部候选按 frame ID 升序，排除已处理帧。

每成功加入一个，更新：

```python
last_keyframe
last_keyframe_frame_id
last_keyframe_node_id
```

使用统一 `process_accepted_keyframe()` 和显式 parent node。

测试：

```text
两个尾部候选都成功时，第二个候选 parent 必须是第一个尾部关键帧。
```

---

## 13. Phase 9：统一 ConfigLoader

正式 Pipeline 和 Debug Pipeline 必须使用同一 ConfigLoader。

修复顶层 `class_compatibility` 读取。

启动时保存 `effective_config.yaml`，包含默认值合并后的最终配置，并记录：

```text
git SHA
model hash
video hash
matcher thresholds
tracker thresholds
keyframe thresholds
association thresholds
detector thresholds
```

Debug 日志显示的有效阈值与仓库配置文件可能存在加载路径或默认值差异。Agent 必须保存并说明运行时实际生效的配置；先修逻辑、记录 MatchResult 分布，再调匹配阈值。

---

## 14. Phase 10：Detector、L3 和 Fusion

- YOLO 推理传入 device；
- L3 `regions` 真正裁剪；
- 恢复 bbox 原图坐标；
- 多 ROI 去重；
- `L3 enabled=false` 时不调用；
- Fusion 负责合并同一实例的 L1/L3 重复框；
- ObjectAssociator 不再用 bbox IoU 猜测是不是同一实例。

测试：

1. 同一实例 L1/L3 重复框合并；
2. 两个同类轻微重叠实例都保留；
3. 重叠 ROI 的重复框合并；
4. 不同类别不做 class-agnostic 抑制。

---

## 15. Phase 11：报告、ID 和输出

统一：

```python
get_reportable_objects()
```

以下全部使用同一集合：

- JSON；
- CSV；
- 控制台；
- API；
- evidence；
- mosaic。

输出明确区分：

```text
all_provisional_objects
final_reportable_objects
confirmed_count
tentative_count
uncertain_count
rejected_count
review_required_count
```

顺序：

```text
final_review
-> validate_object_map
-> filter reportable
-> assign persistent IDs to reportable only
-> generate reports
```

报告增加：

```json
{
  "observation_frame_count": 0,
  "observation_frame_ids_unique": true,
  "rejected_reason": null,
  "merged_into_id": null,
  "track_conflict_count": 0,
  "mapping_quality_mean": 0.0,
  "mapping_quality_min": 0.0
}
```

验证 JSON、CSV、summary、class_counts 完全一致。

---

## 16. Phase 12：图像质量和时间推进

分离：

```python
is_acceptable_for_mapping()
is_acceptable_for_detection()
```

质量差帧：

- 不进入全局建图；
- 推进 Tracker 时间；
- 可进入 recovery buffer，带低质量标记；
- 不生成全局检测。

曝光使用 underexposed ratio 和 overexposed ratio，不只看灰度均值。

Debug 输出连续质量拒绝区间，便于检查长时间空窗。

---

## 17. 测试文件

新增：

```text
tests/test_keyframe_trigger_logic.py
tests/test_tracker_state_machine.py
tests/test_track_object_binding.py
tests/test_same_frame_invariants.py
tests/test_object_merge_policy.py
tests/test_recovery_parent_graph.py
tests/test_end_window_keyframes.py
tests/test_effective_config.py
tests/test_report_consistency.py
tests/test_debug_output_isolation.py
tests/test_pipeline_debug_regression.py
```

关键回归：

### 关键帧风暴

L2 每 3 帧运行一次，其他帧为 `None`，某 track 发生一次置信度下降：

```text
quality_drop trigger = 1
max consecutive keyframes <= 2
```

### track 重绑定

track 33 已绑定 P-0010，下一帧空间代价偏向 P-0008：

```text
不覆盖绑定
产生 TRACK_BIND_CONFLICT
两个对象均不被错误更新
```

### 同帧两个同类工具

轻微重叠，后续投影质心接近：

```text
最终两个对象
cannot_merge
final_review 后仍为两个
```

### 44 个 provisional 碎片

包含真重复、同帧独立实例、track conflict 和证据不足对象：

- 只合并满足完整安全条件的对象；
- 其余只标记 review；
- 所有 rejected 有原因；
- 不根据目标数量反推合并。

---

## 18. 真实视频验收

同一个 `test_cut.mp4` 重新运行。

硬性条件：

```text
track_binding_conflicts = 0
same_frame_duplicate_observations = 0
unexplained_rejected_objects = 0
invalid_homography_nodes = 0
report_count_inconsistency = 0
```

关键帧要求：

- 不再长时间逐帧 QUAL_DROP；
- 输出最大连续关键帧长度；
- 每个关键帧有 trigger；
- L2 未运行帧不能产生 L2 质量触发；
- 接纳率超过 50% 时输出 warning，但不直接作为算法硬阈值。

每个对象：

```python
assert observation_count == len(observations)
assert observation_count == len({obs.frame_id for obs in observations})
```

每个自动合并：

```text
无 frame overlap
无 co-occurrence
无 track conflict
有 merge audit
```

即使最终仍为 15，也必须同时报告：

```text
在线创建对象数
在线关联次数
自动合并次数
疑似重复数量
rejected 数
rejected 原因分布
```

---

## 19. 参数标定顺序

P0 不变量全部通过后再调：

1. keyframe cooldown；
2. tracker IoU/中心距离；
3. homography 质量；
4. 在线关联 gate；
5. duplicate 标记；
6. L1/L3 fusion；
7. confirmation 阈值。

每次只改一组，保存有效配置和结果。

禁止：

- 通过增大 merge 半径消除重复；
- 通过减小 merge 半径消除误合并；
- 根据真实数量循环试阈值；
- 无解释地把大量对象设为 rejected。

---

## 20. 文件级修改清单

必改：

```text
apps/offline_scan.py
debug_pipeline.py
core/types.py
core/simple_tracker.py
core/keyframe_selector.py
core/object_associator.py
core/global_object_map.py
core/homography_graph.py
core/recovery_manager.py
core/report_generator.py
core/config_loader.py
core/quality_evaluator.py
core/detector.py
core/detection_fusion.py
configs/pipeline.yaml
configs/tracker.yaml
configs/associator.yaml
configs/matcher.yaml
configs/detector.yaml
```

建议新增：

```text
core/debug_events.py
core/exceptions.py
core/merge_policy.py
core/pipeline_result.py
CHANGELOG_DEBUG_V2.md
```

---

## 21. 建议提交顺序

```text
1. chore: snapshot current local agent modifications
2. test: reproduce keyframe storm and track binding conflicts
3. fix: distinguish L2 not-run from empty detections
4. fix: add keyframe cooldown and edge-triggered quality drop
5. fix: correct tracker states, class cost, and frame-gap timing
6. fix: enforce stable logical-track object bindings
7. fix: enforce one observation per object per frame
8. fix: persist all same-frame cannot-merge constraints
9. refactor: replace aggressive final merge with audited merge policy
10. fix: add explicit homography parent nodes and recovery anchors
11. fix: process end-window keyframes with correct parent chain
12. refactor: centralize effective configuration
13. fix: unify reportable objects, IDs, and rejected audit
14. feat: add structured debug logs and isolated run outputs
15. test: add full debug regression coverage
16. docs: document limitations and evaluation procedure
```

---

## 22. Agent 约束

1. 不得删除或放宽失败测试；
2. 不得新增 LightGlue、BoT-SORT；
3. 不得用单位阵 RECOVERY；
4. 不得 warning 后覆盖 track-object 映射；
5. 不得让对象同帧接收多个 observation；
6. 不得合并 frame overlap 对象；
7. 不得仅凭固定 centroid 距离自动合并；
8. 不得仅凭 shared track 合并有 conflict 的对象；
9. 不得产生无 rejected_reason 的 REJECTED；
10. 不得让 Debug 和正式 Pipeline 使用不同配置入口；
11. 不得偏离用户确认的统一 `master` 基线；
12. 不得以“最终数量等于 15”为唯一成功标准；
13. 不得声称 RECOVERY 已验证，除非自动测试强制覆盖；
14. 不得把 Debug 写盘时间计入核心 FPS。

---

## 23. 每阶段报告格式

```text
阶段：
- Phase X

基线 SHA：
- ...

修改文件：
- ...

复现问题：
- ...

根因：
- ...

修改方法：
- ...

新增不变量：
- ...

新增测试：
- ...

测试结果：
- X passed, Y failed

真实视频结果：
- keyframe count
- max consecutive keyframes
- track binding conflicts
- same-frame duplicate observations
- objects created
- objects merged
- unexplained rejected objects
- final reportable objects

仍未解决：
- ...
```

---

## 24. 最终交付

1. 修复后的源码分支；
2. 全部新增测试；
3. `pytest tests/ -q` 完整输出；
4. `CHANGELOG_DEBUG_V2.md`；
5. `run_metadata.json`；
6. `effective_config.yaml`；
7. `events.jsonl`；
8. `keyframe_stats.csv`；
9. `association_events.csv`；
10. `object_lifecycle.json`；
11. JSON/CSV 报告；
12. 同一视频修改前后对比；
13. 起始 SHA、修改后 SHA 与提交记录；
14. 未完成事项清单。

---

## 25. 最终判断标准

本轮完成的标准不是“不报错”，也不是最后恰好输出 15，而是：

> 未运行检测不会制造关键帧触发；同一轨迹不会在不同对象间静默跳转；同一对象不会在同一帧吸收多个实例；曾经同帧存在的工具不会在后处理中合并；所有拒绝和合并都有可复核证据；RECOVERY 使用正确锚点和变换语义。

优先级：

```text
系统不变量
> 在线关联正确性
> 合并可解释性
> 建图正确性
> 参数标定
> 性能
> 可视化
> API
> 新算法替换
```
