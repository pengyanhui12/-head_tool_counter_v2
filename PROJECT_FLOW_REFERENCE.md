# Head Tool Counter 项目流程参考手册
时间：2026-08-19

> 本手册面向后续代码修改、参数调优和问题定位。内容以当前正式流水线为准；正式入口为 `apps/offline_scan.py::run_pipeline()`。

## 1. 项目目标

本项目从移动拍摄的视频中检测并统计作业工具。它不是简单累加每帧检测框，而是通过短期跟踪、关键帧建图、全局坐标投影和跨帧对象关联，将不同画面中的同一件工具合并为一个实体。

最终计数口径为：

```text
total_objects = CONFIRMED + UNCERTAIN
```

`TENTATIVE` 进入人工复核候选，不进入正式计数；`REJECTED` 不计数。

## 2. 快速运行

项目要求 Python 3.10 或更高版本。推荐使用已有 Conda 环境：

```powershell
conda activate head_tool_counter
python -m pytest
python -m apps.offline_scan `
  --video "input.mp4" `
  --config-dir "configs" `
  --output-dir "outputs"
```

启用性能统计：

```powershell
python -m apps.offline_scan --video "input.mp4" --performance
```

API 启动方式：

```powershell
python -m apps.api_server
```

详细诊断入口：

```powershell
python debug_pipeline.py
```

`debug_pipeline.py` 含模块级执行代码，不应被其他模块直接导入。

## 3. 目录与职责

```text
head_tool_counter/
├─ apps/
│  ├─ offline_scan.py       # 正式离线入口和流水线编排
│  └─ api_server.py         # FastAPI 服务封装
├─ core/                    # 算法、领域对象和输出逻辑
│  └─ performance_profiler.py # 离线性能计时、展示和报告保存
├─ configs/                 # YAML 运行配置
├─ models/best.pt           # YOLO 模型权重
├─ tests/                   # pytest 测试
├─ debug_pipeline.py        # 详细诊断脚本
├─ debug_output/            # 调试产物，不是源码
└─ pyproject.toml           # 项目和依赖元数据
```

核心模块分组：

| 阶段 | 模块 | 主要职责 |
|---|---|---|
| 输入 | `video_reader.py` | 解码视频并限制输入 FPS |
| 质量 | `quality_evaluator.py` | 计算清晰度、曝光和帧可用性 |
| 检测 | `detector.py` | 执行 L1/L2/L3 YOLO 推理 |
| 融合 | `detection_fusion.py` | 合并 L1/L3 重复检测 |
| 跟踪 | `simple_tracker.py` | 维护视频坐标内的短期 Track |
| 关键帧 | `keyframe_selector.py` | 决定是否创建关键帧 |
| 匹配 | `feature_matcher.py` | SIFT、RANSAC 和单应矩阵验证 |
| 建图 | `homography_graph.py` | 维护关键帧到全局坐标的变换链 |
| 投影 | `global_projector.py` | 将检测框投影到全局坐标 |
| 全局关联 | `object_associator.py` | 跨关键帧关联、创建、合并对象 |
| 对象存储 | `global_object_map.py` | 管理对象状态和双 ID |
| 覆盖 | `coverage_map.py` | 记录有效关键帧覆盖的全局区域 |
| 恢复 | `recovery_manager.py` | 特征匹配失败后的重新接图 |
| 输出 | `report_generator.py` | 统一 JSON/CSV 计数口径 |
| 证据 | `evidence_extractor.py` | 选择并保存对象证据帧 |

## 4. 完整处理流程

```text
视频输入
  → VideoReader 逐帧读取
  → QualityEvaluator 计算清晰度和曝光
  → 不满足检测质量：推进 tracker 时间并跳过
  → 按间隔执行 L2 低成本检测
  → tracker.preview() 产生关键帧触发信号
  → KeyframeSelector 判断 SKIP / ACCEPTED / RECOVERY
      ├─ SKIP：必要时用 L2 更新 tracker
      ├─ RECOVERY：尝试连接缓存帧或历史图节点
      └─ ACCEPTED：加入 HomographyGraph
          → 执行 L1 高精度检测
          → 可选执行 L3 局部检测
          → DetectionFusion 融合
          → tracker.update() 分配 track_id
          → GlobalProjector 投影到全局坐标
          → ObjectAssociator 跨帧关联
          → CoverageMap 记录视野覆盖
  → 视频结束后补选尾部关键帧
  → ObjectAssociator.final_review()
  → 分配 GO-xxxx 持久 ID
  → 生成报告、拼图、证据图和会话文件
```

## 5. 初始化和配置装配

`run_pipeline()` 使用 `ConfigLoader` 读取配置，并创建各组件。相对配置路径以项目根目录解析。

当前已接入正式流水线的配置包括：

- `pipeline.yaml`：FPS、检测间隔、关键帧间隔和质量阈值；
- `detector.yaml`：模型路径、设备、L1/L2/L3 和融合参数；
- `tracker.yaml`：跟踪门控、丢失与重激活规则；
- `matcher.yaml`：特征匹配和单应矩阵验证阈值；
- `associator.yaml`：全局关联、确认和重复候选规则；
- `coverage.yaml`：网格分辨率、有效面积和目标覆盖率。

`camera.yaml` 是 CameraProfile、标定和边缘映射的预留配置。当前没有完整相机模型，不应将它误映射为普通特征匹配参数。

## 6. 帧质量门控

`QualityEvaluator.evaluate()` 对灰度图计算 Laplacian 方差作为清晰度，并统计过暗、过亮像素得到曝光分数。

默认阈值：

```yaml
detection_sharpness_threshold: 63.0
sharpness_threshold: 89.0
```

行为：

- 清晰度 `< 63`：不检测，不增加 missed detection，只推进时间；
- 清晰度 `63～89`：允许检测和跟踪，不允许正常建图；
- 清晰度 `>= 89` 且曝光合格：允许进入关键帧建图。

正式建图使用 `is_acceptable_for_mapping()`，同时要求曝光分数不低于 `0.3`。首个关键帧长期无法出现时，`InitialKeyframeFallback` 会选择缓存中最清晰的检测级帧。

## 7. L1、L2、L3 检测

### 7.1 L2：侦察与跟踪

L2 默认每 3 帧运行一次，分辨率较低。它用于更新 Track、发现连续新目标和检测跟踪质量下降，不直接进入最终全局计数。

### 7.2 L1：关键帧正式检测

只有被接受的关键帧执行 L1。L1 检测经融合、跟踪和投影后进入全局对象关联，是正式计数的主要检测来源。

### 7.3 L3：局部精细检测

L3 默认关闭。启用后，系统从 L2 中选取低置信度且面积合格的目标区域，扩大 ROI 后执行高分辨率局部检测。

`detector.yaml` 中的 `model.path`、`model.device` 以及各级 `imgsz/conf/iou` 均会传入正式检测器。推理时 `device` 会显式传给 Ultralytics。

## 8. 短期跟踪器

`SimpleDetectionTracker` 维护当前视频坐标内的局部身份：

```text
active → inactive → lost
```

更新分三步：

1. Active Track 与检测做匈牙利匹配；
2. 未匹配检测尝试重激活 Inactive Track；
3. 剩余检测创建新 Track。

匹配代价主要由 IoU、归一化中心距离和类别兼容性组成。`track_id` 只是短期身份；同一件工具可能因遮挡产生多个 Track，最终由 `GlobalObject` 统一。

`preview()` 不修改正式状态，用于判断：

- 是否有连续多次出现的新检测；
- Track 置信度是否明显下降。

这两个信号会触发关键帧判断。

## 9. 关键帧和特征匹配

关键帧触发条件：

- 达到最大关键帧间隔；
- 连续出现未匹配的新检测；
- Track 质量下降；
- 视频结尾强制候选。

普通触发受 `min_keyframe_interval_frames` 冷却限制。触发后，`FeatureMatcher` 使用 SIFT 特征、KNN 匹配、Lowe ratio test 和 RANSAC 估计：

```text
H_current_to_previous
```

匹配结果还要通过内点数、内点率、重投影误差、空间分布、面积变化和矩阵条件数检查。

`FeatureMatcher` 默认缓存最近4张图像的SIFT关键点和描述子，仅当输入是
同一个 `numpy.ndarray` 对象时命中。流水线因此将质量评估后的
`Frame.image` 视为只读数据；若原地修改图像，必须调用
`clear_feature_cache()`。可在 `matcher.yaml` 将
`feature_cache_size` 设为0以关闭缓存并执行消融对比。

视频结束时不会再对尾窗30帧全部执行SIFT。系统按时间把尾窗划分为6段，
每段按清晰度、曝光质量和帧号确定性选择1帧，再对最多6个候选运行完整
SIFT并选择最佳2帧。`pipeline.yaml` 中
`end_window_match_candidates` 控制候选数；设为0或不小于尾窗帧数可恢复
旧的全量匹配行为。

结果：

- `ACCEPTED`：单应矩阵有效；
- `SKIP`：无触发或处于冷却期；
- `RECOVERY`：需要关键帧，但直接匹配失败。

## 10. 单应图和全局投影

`HomographyGraph` 将相邻关键帧的变换串联，得到每个关键帧到第一关键帧坐标系的变换：

```text
H_keyframe_to_global
```

`GlobalProjector` 投影检测框四角和中心，产生：

- 全局四边形；
- 全局中心和多边形质心；
- 投影面积；
- mapping quality；
- transform version。

单应矩阵错误会直接导致全局位置漂移，并表现为同一工具重复计数或不同工具误合并。遇到计数问题时，应先检查建图，再调整全局关联距离。

## 11. Recovery 恢复流程

直接特征匹配失败后，`RecoveryManager` 尝试使用近期缓存帧或历史锚点重新连接到单应图。

- 恢复成功：新建关键帧节点，继续检测、投影和关联；
- 恢复失败：缓存当前帧和 L2 检测，等待后续帧重新建立联系。

恢复关键帧当前使用固定 `mapping_quality=0.5`，因此其空间观测可信度低于高质量直接匹配帧。

## 12. 数据对象生命周期

```text
Frame
  ↓ YOLO
DetectionCandidate
  ↓ Tracker
TrackedDetection
  ↓ 添加关键帧、质量、四角信息
RawDetection
  ↓ 单应矩阵投影
GlobalDetection
  ↓ 跨帧聚合
GlobalObject
  ↓ 报告序列化
JSON / CSV Object
```

关键区别：

| 类型 | 含义 |
|---|---|
| `DetectionCandidate` | 单次 YOLO 检测结果 |
| `Track` | 视频像素坐标内的短期目标 |
| `GlobalDetection` | 已投影到全局平面的单次观测 |
| `GlobalObject` | 多帧、多 Track 聚合后的物理实体 |

修改代码前，先确定当前逻辑处理的是“检测”“Track”还是“全局对象”。

## 13. 全局对象关联

`ObjectAssociator.ingest_frame()` 按以下优先级处理 `GlobalDetection`：

1. **Track 强关联**：已绑定、未断联且仍在空间门控内的逻辑 Track 优先回到原对象；
2. **全局匈牙利匹配**：根据位置、投影多边形重叠、面积和类别代价匹配已有对象；
3. **创建新对象**：无法匹配时创建 `P-xxxx`。

全局关联受位置门控、类别兼容性和 `max_cost` 限制。其中重叠代价为 `1 - IoU`，会取当前检测与对象最近 10 个有效投影轮廓的最大 IoU；投影无效时按最差重叠代价处理。同一帧一个对象最多只能接收一个观测。

已绑定 Track 如果长时间断联或突然跳出对象空间门控，会释放本帧的强绑定并降级到全局空间匹配。位置仍一致时可以重新绑定原对象；匹配到其他对象时，旧对象和新对象都会添加 `TRACK_CONFLICT` 复核标记，最终合并策略禁止它们仅凭共享 Track 自动合并。

对象通过观测数量、关键帧数量和类别投票一致性从 `TENTATIVE` 升级：

```yaml
min_observations_confirmed: 5
min_keyframes_confirmed: 3
min_top_class_ratio: 0.60
```

## 14. 对象状态与 ID

确认状态：

- `TENTATIVE`：证据不足，进入人工复核候选；
- `CONFIRMED`：证据充分，正式计数；
- `UNCERTAIN`：仍计数，但需要关注；
- `REJECTED`：不计数，必须记录原因。

可见性状态与确认状态独立：

- `ACTIVE`：近期仍有观测；
- `INACTIVE`：较长时间没有观测。

ID 分为：

- `P-xxxx`：处理中使用的 provisional ID；
- `GO-xxxx`：最终只分配给 `CONFIRMED` 和 `UNCERTAIN` 的 persistent ID。

## 15. 最终合并与重复复核

视频结束后 `final_review()` 执行：

1. 共享 Track 且满足安全条件的对象自动合并；
2. 空间接近但证据不足的对象只标记 `LIKELY_DUPLICATE`；
3. 检查 Tentative 是否可能是 Confirmed 的局部重复；
4. 记录合并审计和拒绝原因。

`TRACK_CONFLICT` 表示 Track 身份链发生过跳变或冲突，不代表两个对象在空间上疑似重复，因此不会仅因共享冲突 Track 自动添加 `LIKELY_DUPLICATE`。后者只用于确有空间接近或局部重复证据、但不足以安全自动合并的对象。

重要不变量：

- 同一逻辑 Track 不得绑定两个未合法合并的对象；
- 一个对象在同一帧最多一个观测；
- 同一帧的不同融合检测必须对应不同对象；
- 曾同帧独立共现的对象不得自动合并；
- 合并双方的观测帧集合不得重叠；
- `observation_count` 必须等于观测列表长度；
- 每个 `REJECTED` 必须有 `rejected_reason`。

修改关联和合并代码时，必须运行相关不变量测试。

## 16. CoverageMap 覆盖地图

`CoverageMap` 记录成功建图的关键帧在全局平面上覆盖了哪些区域。每次接受关键帧后，系统投影整帧四角形成视野多边形，并调用 `update()`。

配置：

```yaml
coverage:
  grid_resolution: 100
  minimum_valid_polygon_area: 100.0
  target_coverage_ratio: 0.95
```

- `grid_resolution`：覆盖率离散网格精度；
- `minimum_valid_polygon_area`：过滤异常小或退化投影视野；
- `target_coverage_ratio`：目标覆盖率。

`get_coverage(region_polygon)` 计算目标区域的覆盖比例，`is_target_reached()` 判断是否达标。

当前正式流程会收集覆盖多边形，但尚未定义最终目标区域，也未将覆盖率写入报告。因此 CoverageMap 已参与数据收集，尚未参与最终计数决策。

## 17. 视频结束后的处理

主循环结束后：

1. 从尾部窗口选择最多两个质量较好的关键帧；
2. 对未处理的尾部关键帧执行匹配、检测、投影和关联；
3. 执行最终对象审查和持久 ID 分配；
4. 选择每个对象的最佳证据帧；
5. 生成全局拼图；
6. 保存证据图片、会话数据、JSON 和 CSV。

尾部补选可降低工具只在视频末尾出现时的漏检风险。

证据帧选择会综合清晰度、检测置信度、映射质量，以及单次观测与对象全局质心的空间一致性。空间上明显偏离聚合对象的位置即使更清晰，也会受到惩罚，避免相邻同类工具的错误观测成为代表图片。缺少有效全局几何时，系统回退为按清晰度和检测置信度选择。

## 18. 输出结构与报告字段

典型输出：

```text
outputs/
├─ reports/
│  ├─ report.json
│  ├─ report.csv
│  ├─ performance.json
│  └─ performance.txt
├─ global/
│  └─ global_mosaic.jpg
├─ evidence/
│  └─ GO-xxxx_<class>.jpg
└─ sessions/
   └─ <session_id>/objects.json
```

JSON 主要分区：

- `objects`：正式计数对象；
- `review_candidates`：Tentative 对象；
- `rejected_objects`：拒绝对象及审计信息；
- `class_counts`：正式计数的分类统计；
- `review_required_count`：需要人工关注的对象数。

报告一致性规则：

```text
total_objects == confirmed_count + uncertain_count
tentative_count == review_candidate_count
```

## 19. 参数调优指南

### 19.1 运行速度慢

使用命令行 `--performance`，或将 `configs/pipeline.yaml` 中的
`enable_performance_stats` 设为 `true`。启用后，控制台会显示阶段耗时，
并在输出目录的 `reports/` 下保存 `performance.json` 和
`performance.txt`；关闭时不会生成这两个文件。

`Pipeline wall time` 从 `run_pipeline()` 入口开始，包含配置加载、模块
初始化、逐帧识别和 Mosaic/Evidence 等产物生成，但不包含 Python 进程
启动及模块导入。`pipeline_fps` 因此表示完整离线任务吞吐量，不等同于
纯识别循环 FPS。若 `timing_overlap_detected` 为 `true`，说明阶段计时发生
重叠，不应直接使用该次覆盖率进行论文对比。

依次考虑：

1. 降低 `max_input_fps`；
2. 增大 `l2_interval_frames`；
3. 降低 L1/L2 `imgsz`；
4. 关闭 L3；
5. 使用 `--performance` 确认瓶颈。

论文实验必须固定视频、配置、模型、运行设备和软件环境；先进行预热，
重复运行并报告中位数或均值及离散程度。核心算法性能与 Global Mosaic、
Evidence 等产物生成耗时应分别报告。

### 19.2 漏检工具

检查顺序：

1. 原始帧中 YOLO 是否检测到；
2. 帧是否因清晰度或曝光被跳过；
3. L2 是否触发了关键帧；
4. 关键帧匹配是否失败；
5. 对象是否停留在 Tentative；
6. 对象是否被错误合并或拒绝。

可能调整：检测置信度、输入 FPS、L2 间隔、新目标确认次数和关键帧间隔。

### 19.3 同一工具被重复计数

先检查全局投影是否稳定，再检查：

- Track 是否频繁断裂；
- `max_position_distance_px` 和 `online_gate_ratio` 是否过小；
- 类别是否发生不兼容切换；
- 对象是否有共享 Track；
- 是否被同帧共现规则正确阻止合并。

不要仅为减少数量而盲目增大关联距离，否则容易把不同工具误合并。

### 19.4 不同工具被合并

检查：

- Tracker 是否把两个目标串成同一 Track；
- 全局投影是否把两个位置压缩到一起；
- 类别兼容表是否过宽；
- 同帧共现是否被正确记录；
- 关联位置门控是否过大。

### 19.5 经常进入 Recovery

查看匹配失败原因，按顺序检查：

- 图像是否模糊或曝光异常；
- 相邻关键帧视角变化是否过大；
- SIFT 特征数量是否不足；
- 内点数、内点率或空间分布门槛是否过严；
- 单应矩阵是否因非平面场景退化。

## 20. 问题定位决策表

| 现象 | 优先模块 | 优先证据 |
|---|---|---|
| 完全没有检测框 | `detector.py` | YOLO 原始结果、模型路径和 device |
| 检测到但没有关键帧 | `simple_tracker.py`, `keyframe_selector.py` | preview 信号和触发原因 |
| 关键帧频繁失败 | `feature_matcher.py` | matches、inliers、reprojection error |
| 全局坐标漂移 | `homography_graph.py`, `global_projector.py` | 单应矩阵和全局拼图 |
| 同一工具多计 | `object_associator.py` | Track 绑定、距离、merge audit |
| 多件工具少计 | Tracker、共现、合并策略 | 同帧对象和 shared track |
| 内部对象正确但报告错误 | `report_generator.py` | 状态分区和 ID |
| 运行缓慢 | `performance_profiler.py` | `performance.json`、`--performance` 分阶段耗时 |

## 21. 测试与修改安全线

完整测试：

```powershell
python -m pytest
```

常用定向测试：

```powershell
python -m pytest tests/test_detector.py -v
python -m pytest tests/test_feature_matcher.py -v
python -m pytest tests/test_simple_tracker.py -v
python -m pytest tests/test_object_merge_policy.py -v
python -m pytest tests/test_same_frame_invariants.py -v
python -m pytest tests/test_report_consistency.py -v
python -m pytest tests/test_effective_config.py -v
```

修改原则：

1. 先用固定视频、固定配置保存基线；
2. 每次只调整一个阶段或一组强相关参数；
3. 对 Bug 先添加能够复现问题的失败测试；
4. 修改代码时添加简洁、必要的中文注释；
5. 先运行相关测试，再运行完整测试；
6. 比较数量之外，还要比较误合并、漏检、复核候选和耗时；
7. 不要提交视频、模型副本、调试输出和机器专用绝对路径。

## 22. 推荐阅读顺序

按以下顺序阅读可以最快建立完整认识：

1. `apps/offline_scan.py::run_pipeline()`：掌握总流程；
2. `core/types.py`：掌握数据类型；
3. `core/quality_evaluator.py`：理解帧过滤；
4. `core/detector.py` 和 `core/detection_fusion.py`：理解检测；
5. `core/simple_tracker.py`：理解短期 Track；
6. `core/keyframe_selector.py` 和 `core/feature_matcher.py`：理解关键帧；
7. `core/homography_graph.py` 和 `core/global_projector.py`：理解全局坐标；
8. `core/object_associator.py`：理解最终去重；
9. `core/report_generator.py`：理解最终计数；
10. `tests/`：理解不能破坏的业务不变量。

## 23. 类别数量摘要输出

离线入口支持可选开关：

```powershell
python -m apps.offline_scan --class-summary
```

开启后，`core/class_summary.py` 直接复用最终 `report.json` 中的 `total_objects` 和 `class_counts`，在控制台按类别名称排序输出数量，并保存 `reports/class_counts.json`。该摘要遵循正式计数语义，不包含 rejected 对象或仅供审核的 tentative 对象。开关默认关闭，关闭时不会创建额外摘要文件。

## 24. Global Mosaic 生成流程

`core/global_mosaic.py` 先根据正式计数对象和审核候选的投影角点、质心规划最终画布，再按原节点顺序融合抽样关键帧。视频在该阶段只打开一次，每个抽样关键帧最多执行一次 `warpPerspective()`；透明边界的目标缓冲区会显式清零，避免未初始化像素进入融合结果。网格、纹理、对象标注的绘制顺序保持不变。单帧读取失败时跳过该帧，视频无法打开或缺少有效对象坐标时不生成 Mosaic。

## 25. 后续值得独立实施的能力

以下内容目前尚未形成完整闭环，适合分别设计和测试：

- CameraProfile、相机标定和边缘观测权重；
- 定义目标扫描区域，并将覆盖率写入报告；
- 使用覆盖率提示补拍或终止扫描；
- 将 `run_pipeline()` 的统计和路径统一封装为 `PipelineResult` 返回；
- 将调试流水线与正式流水线共享同一组件装配逻辑；
- 增加基于人工标注视频的准确率、召回率和误合并评估工具。

这些能力应逐项实施，避免同时修改检测、建图和关联，使效果无法归因。
