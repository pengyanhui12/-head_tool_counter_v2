# 轨道交通作业工器具检测计数系统 — 设计文档

> 日期：2026-07-24
> 状态：已确认

---

## 1. 项目概述

### 1.1 场景

轨道交通作业前/后，工作人员将工具摆放在地面上，佩戴头戴相机环绕拍摄。系统实时处理视频，自动检测并计数工具，生成可视化产物。人工对低置信度目标进行复核。

### 1.2 约束条件

- 单目头戴相机，无 IMU 数据
- 不允许使用 AprilTag/ArUco 标记清点垫
- 相机距地面约 1.2-1.8m（后期可能变化）
- 实时处理：拍摄完后云端即时返回结果
- 一次扫描约 20 件工具，每帧覆盖约 3/4

### 1.3 精度要求

- 检测精度 ≥ 95%
- 漏检过多影响人工复检效率

### 1.4 现有资源

- 已训练 YOLO 模型可检测 10 类工具，后续扩展更多类别
- 同类工具不同规格、多品牌、新旧不一，同类别内不区分规格

---

## 2. 核心技术路线

**方案：2D 单应性拼接（方案 B）**

基于地面平面假设，通过 ALIKED + LightGlue 做帧间特征匹配 + RANSAC 估计单应性矩阵，构建全局二维坐标地图。所有检测结果投影到全局坐标系后做空间去重。

选择理由：
- 头戴相机 1.2-1.8m 俯视地面，平面假设成立
- 不处理绝对尺度（无 IMU/标记），坐标系为归一化像素坐标
- 比 3D SLAM 快，利于实时返回

---

## 3. Pipeline

```
客户端(头戴相机)  分片上传
  ┌──────────┐ ──────→ 云端处理
  │ 视频录制  │
  │ 边拍边传  │
  └──────────┘

云端流水线（帧级并行）：

              ┌─── 线程A: 建图 ───┐        ┌─── 线程B: 检测 ───┐
              │                   │        │                   │
帧到达        │  清晰度筛选       │        │  关键帧:L1 YOLO    │
  │           │  三步关键帧判定    │        │  (唯一投影源)       │
  ▼           │  (候选→接纳→恢复)  │        │  L2 YOLO(触发用)   │
  │           │  地面ROI特征提取   │        │  L3复检(按需)      │
  ▼           │  (候选→接纳→恢复)  │        │                    │
  ▼           │  地面ROI特征提取   │        │                    │
SharpnessFilter│ ALIKED+LightGlue │        │  BoT-SORT(仅检测帧)│
  │           │  RANSAC单应性     │        │                    │
  ▼           │  链式累积+回环    │        │                    │
KeyframeSelector                   │        │                    │
  │           └────────┬──────────┘        └────────┬───────────┘
  │                    │                            │
  │                    └──────────┬─────────────────┘
  │                               ▼
  │                         投影至全局坐标
  │              (角点→四边形质心)
  │                               │
  │                               ▼
  │               在线门控关联 + 匈牙利匹配
  │              (同帧互斥硬约束)
  │                               │
  │                               ▼
  │                加权多帧类别投票
  │           (五因素权重 + Track内融合)
  │                               │
  │                    ┌──────────┴──────────┐
  │                    ▼                     ▼
  │              TENTATIVE → CONFIRMED/UNCERTAIN/REJECTED
  │              各状态 + review_flags
  │              (主状态与复核标志分离，可并存)
  │                    │                     │
  │                    └──────────┬──────────┘
  │                               ▼
  │                       视频结束时:
  │                 离线 DBSCAN 复核
  │                  (识别疑似重复对象)
  │                               │
  │                               ▼
  │                    操作员框选清点区域
  │                       → 计算覆盖率
  │                 不足 → 提示补充扫描
  │                               │
  │                               ▼
  │                       全局示意图 + 证据帧 + CSV + 覆盖热力图
  └───────────────────────────────────────────
```

### 关键帧策略：候选触发 → 质量接纳 → 失败 RECOVERY

关键帧是整个地图的锚点。错误的关键帧会导致单应矩阵漂移，后续所有工具位置失准。
因此采用**两步机制**：先触发候选，再质量把关。

#### 第一步：候选触发（满足任一）

| 触发条件 | 说明 |
|---------|------|
| 相对于上一关键帧发生足够运动 | 平移/旋转量超过阈值，说明需要新锚点覆盖新区域 |
| 与上一关键帧重叠程度下降但仍可靠 | 内点数仍在接纳阈值之上但在减少，说明正在离开当前区域 |
| 距上一关键帧 ≥ 最大间隔（30帧） | 时间兜底，防止长时间无关键帧 |
| 新增覆盖区域 | 当前帧视场投影到地图，发现有新的地面网格被覆盖 |
| L2 粗检发现疑似新目标 | 新工具出现需要尽快定位 |
| 活跃 track 的 BoT-SORT 置信度骤降 >30% | 跟踪质量下降，需要新的检测锚定 |
| 视频尾部 | 最后 30 帧 → 清晰度筛选 → 配准质量筛选 → 选最佳 1~2 帧作为结束关键帧候选 |

满足任一条件 → 标记为**候选帧**，进入接纳判定。

#### 第二步：质量接纳（候选帧必须**同时满足**所有条件）

| 接纳条件 | 含义 |
|---------|------|
| RANSAC 内点数 ≥ 阈值 | 有足够多的可靠匹配点 |
| 内点比例 ≥ 阈值 | 匹配质量高，不是偶然的虚假匹配 |
| 重投影误差 < 阈值 | 单应矩阵在几何上拟合良好 |
| 匹配点空间分布均匀 | 四个象限均有内点，避免单应性由局部区域主导 |
| 单应矩阵变换合理 | 旋转分量不超限、缩放分量在合理范围、无明显扭曲 |

全部满足 → **接纳为关键帧**，加入 HomographyGraph。

#### 第三步：接纳失败 → RECOVERY

候选帧不满足接纳条件时（如内点不足、重投影误差过大），**不可直接丢弃**：

1. **过渡帧查找**：从帧缓冲中搜索当前帧与上一关键帧之间的中间帧，尝试找到能可靠匹配的过渡帧作为关键帧
2. **历史重定位**：与更早的历史关键帧（不只是上一帧）尝试匹配，看是否能重新建立连接
3. **中断告警**：以上均失败 → 判定扫描路径中断，无法将当前帧接入地图。提示用户"建图中断，请放慢扫描速度或回退到已覆盖区域重新扫描"

RECOVERY 期间，检测和跟踪继续运行（不丢失检测结果），但暂停投影到全局坐标，直到地图恢复。

### 分层检测策略（防止漏检）

> 以下分辨率和频率为**可调节配置**，最终值需通过实际视频测试后确定。

| 层级 | 建议分辨率 | 运行帧 | 目的 | 是否投影到全局地图 |
|------|-----------|--------|------|:---:|
| L1 精检 | 960 或 1280 | 已接纳关键帧 (~1-3次/秒) | 高精度定位 + 分类 + 建图 | ✅ 是 |
| L2 巡检 | 640 或 768 | 每 2~3 帧 (~10-15 次/秒) | 发现新目标、触发 L1/L3、辅助跟踪 | ❌ 否 |
| L3 复检 | 1280 或分块推理 | L2 发现疑似小目标时按需触发 | 高分辨率确认 | ✅ 是（仅在已接纳关键帧上运行，产生 source="L3" 的 RawDetection） |

**数据流规则**：

```
L2 检测帧（非关键帧，无 H 矩阵）
  → 仅用于：发现新目标 / 跟踪 / 触发 L1 或 L3
  → 不投影到全局地图
  → 不参与 global_id 建立和最终计数
  → 不给 RawDetection

L1 检测帧（关键帧，有 H 矩阵）
  → 投影到全局地图
  → 参与 GlobalObject 关联和计数
  → 对应 RawDetection.keyframe_id = 当前关键帧 ID
  → source = "L1"

L3 检测帧（仅在已接纳关键帧上运行，有 H 矩阵）
  → L2 发现疑似小目标时按需触发：在同一关键帧上以更高分辨率(1280)或分块推理方式重检
  → 产生 RawDetection(source="L3") 补充 L1 检测结果
  → 投影到全局地图（source = "L3"）
  → 参与 GlobalObject 关联和计数
  → 与 L1 检测结果共同构成该关键帧的完整检测集
  → 不可在非关键帧上运行（无 H 矩阵无法投影）
```

**L2 发现新目标的处理**：L2 检测到新目标 → 当前帧标记为关键帧候选 → 进入接纳判定 → 若通过质量门禁 → L1 检测该帧 → 投影到全局地图。若接纳失败，该目标等待下一个关键帧覆盖。

BoT-SORT 仅在 L1/L2 检测帧上更新，不在中间非检测帧做卡尔曼预测——头戴相机自身运动使静止工具产生表观运动，纯预测易漂移。非检测帧仅用于建图的特征匹配和质量评估。

---

## 4. 技术选型细节

### 4.1 特征提取与匹配

- **主力**：ALIKED（特征点提取）+ LightGlue（匹配）
- **性能**：端到端耗时需在目标 GPU 和实际分辨率下进行基准测试后确定。性能目标：单次匹配 < 10ms（GPU）。
- **兜底**：SIFT + BFMatcher + ratio test

### 4.2 单应性估计与全局建图

- RANSAC 鲁棒估计帧间单应性矩阵（3×3）

#### 变换方向命名约定（强制执行）

所有 3×3 矩阵命名必须遵循 `H_source_to_target` 格式，禁止使用含义不明确的 `H`、`local_H`、`transform`。

**关键约定**：`FeatureMatcher.match(img_src, img_dst)` 返回 `H_src_to_dst`，即把 src 图像中的点映射到 dst 图像坐标系：

```
p_dst = H_src_to_dst · p_src
```

**帧间匹配**：

```
H_prev_to_curr = match(prev_kf_img, curr_kf_img)
# 含义：将上一关键帧中的点映射到当前帧坐标
```

**链式累积**（全局坐标系 = 第一个关键帧的坐标系）：

```
H_curr_to_global = H_prev_to_global @ H_curr_to_prev
# 注意：H_curr_to_prev = inv(H_prev_to_curr)
```

**代码中**：

```python
# FeatureMatcher: src=当前帧, dst=上一关键帧
H_curr_to_prev = matcher.match(curr_frame, prev_kf)  # 当前→上一

# HomographyGraph 链式累积
H_curr_to_global = H_prev_to_global @ H_curr_to_prev   # 左乘，先映射到上一帧，再累积

# 投影：检测框像素坐标 → 全局坐标
p_global = H_frame_to_global @ p_frame
```

**测试要求**：必须加入非对称变换（旋转+缩放+平移组合），禁止仅用单位矩阵和单方向平移测试。反例：测试中只验证 `abs(H[0,2] - 50) < 15` 无法发现方向性错误。

- 地面 ROI 特征筛选：全图提取特征 → RANSAC 筛选主平面内点 → 根据内点分布动态确定地面区域
  - MVP：默认使用中央 80% 区域的特征；无标定模式下缩小为中央 60%~70%
  - 不固定截取下半部分——头戴相机俯仰变化时地面可能覆盖整幅画面或位于上半部分
  - 检查内点是否覆盖图像多个象限（见关键帧接纳条件中的空间分布均匀性）

#### 回环检测（两级检索）

直接对所有历史关键帧逐一运行 LightGlue 的复杂度为 O(n)，不可随视频长度扩展。

```
当前关键帧
  │
  ▼
① 检索阶段（轻量）
  │  全局图像描述子（如 NetVLAD 或轻量 CNN global descriptor）
  │  召回 Top-K 个候选历史关键帧
  │
  ▼
② 验证阶段（重量）
  │  仅对 K 个候选帧运行 ALIKED + LightGlue
  │
  ▼
③ 几何验证
  │  RANSAC 估计回环单应矩阵
  │  内点数达标 → 建立回环约束 → 触发全局单应图优化
  │
  ▼
④ **回环修正流程（不可变原始数据 + 版本化重建）**

**全局单应图优化**（`optimize_homography_graph()`）：

系统维护的是二维单应性图（非三维位姿图），优化目标是使以下误差最小化：

- 相邻关键帧匹配点的重投影误差
- 回环关键帧匹配点的重投影误差
- 变换平滑性约束（相邻帧变换不应突变）
- 尺度/旋转异常惩罚（单应矩阵不应产生物理不可能的畸变）

优化变量为每个关键帧到全局画布的 3×3 单应矩阵 Hₖ→global。

**MVP 实现要求**：必须实际修正全局矩阵，不能只更新版本号。禁止空操作。

第 1 版使用**扩散平均（diffusion averaging）**：

```
给定回环边 (n_i, n_j) 及其测量的单应矩阵 H_loop，
将误差沿回环路径均匀分配到各帧：

① 计算回环路径上所有节点的累积漂移
② 将漂移量按路径长度加权分配到各中间帧
③ 重新计算所有受影响的 H_k→global

至少保证回环帧的全局变换被实际修正——
transform_version 只在矩阵确实变化时才 +1。
```

后续可用非线性最小二乘（Ceres/g2o 或 LM）替换扩散平均。

> 注意：这不是三维 Bundle Adjustment。不要命名为 `global_ba()`，应使用 `optimize_homography_graph()`。

回环修正不是"通知模块修改几个坐标"，而是**全量重投影 + 全量重关联**。

**不可变原始数据（写入后只读）**：

```python
@dataclass(frozen=True)  # 不可变
class RawDetection:
    """一旦写入永不修改。回环后从中重新投影，而不是修改已有坐标。"""
    bbox: tuple[int,int,int,int]
    center: tuple[float,float]         # 框中心(像素坐标)
    corners: tuple                     # 四角点(像素坐标)
    class_id: int; class_name: str
    confidence: float
    frame_id: int; keyframe_id: int
    sharpness: float
```

**版本化全局状态**：

```python
transform_version: int   # HomographyGraph 每次全局单应图优化后 +1
map_version: int         # GlobalObjectMap 每次完整重关联后 +1
```

**回环修正完整流程**：

```
① HomographyGraph 检测到回环 → 全局单应图优化 → transform_version += 1

② 扫描所有 RawDetection（不可变）：
     用新 H(transform_version) 重新投影 → 新 GlobalDetection
     投影结果带 transform_version 标记，用于判断是否过期

③ ObjectAssociator 接收重投影结果：
     清空旧位置和旧关联（不做增量修改，全量重建）
     但保留不可变约束和证据：
       保留: 同帧互斥、历史共现互斥、track 内连续关系、人工已确认结果
       不保留: 旧 GlobalObject 空间匹配、旧匈牙利分配结果、基于旧单应矩阵的邻近关系
     用新 GlobalDetection 重新运行全量匈牙利匹配
     旧关联最多作为软先验（降低代价矩阵中对应配对的 cost，但不强制匹配）
     → map_version += 1
     → 生成新 GlobalObjectMap

④ GlobalObjectMap 中旧版本对象标记为 superseded，
   保留历史版本（便于对比"回环修正前后变化"）
```

**为什么不能原地修改**：
- 原地修改后无法对比回环前后差异
- 累积误差修复是否正确无从验证
- 增量聚类中的中间状态无法回滚
- 难以调试"这个工具为什么位置跳了 30px"

#### 回环重建后的 ID 稳定性

回环重建后 global_id 可能变化（原 #12 → 重建后 #19），会影响前端、复核记录、审计日志和证据帧引用。

**采用双 ID 机制**：

```python
provisional_id: str     # 在线阶段临时编号（"P-003"），重建后可能变化
persistent_id: str | None  # 最终稳定编号（"GO-001"），仅在全量处理完成后分配，此后永不变化
```

**规则**：
- 在线阶段：所有对象只有 `provisional_id`，`persistent_id = None`
- 回环重建时：通过 `track_ids` 和 `keyframe_ids` 的交集跨版本追踪同一对象，维护 `旧 provisional_id → 新 provisional_id` 映射
- 视频结束 + 最终复核完成后：`persistent_id` 分配，此后永不变化
- 对外接口（前端、审计日志、证据帧）以 `persistent_id` 为准；在线阶段展示 `provisional_id` 并标注"临时"

**跨版本追踪示例**：

```
旧版本 GlobalObject (P-003)
  track_ids = {T5, T8}
  keyframe_ids = {KF_3, KF_7, KF_12}

回环重建 → 新 GlobalObject (P-007)
  track_ids = {T5, T8}          ← 交集非空，继承 P-003 血统
  keyframe_ids = {KF_3, KF_7, KF_12, KF_14}

→ 维护映射: P-003 → P-007
→ 前端和证据帧引用通过映射自动更新
→ 视频结束时分配 persistent_id = "GO-001"
```

第一版短视频可在架构中预留候选检索器接口，暂时使用间隔采样（如每 10 个关键帧取 1 个）作为轻量替代。

### 4.3 投影

- 投影目标：检测框**四个角点**（构成四边形），通过 H 矩阵分别投影到全局坐标
- 四个投影角点构成全局投影四边形，以四边形**质心**作为工具全局位置
- 检测框中心点单独投影，作为辅助参考（不与角点混算多边形）
- 保留四边形面积和方向信息，可用于去重聚类中的距离计算（同类工具预期面积相近）

### 4.4 目标跟踪

- **跟踪器选型**：第一版同时比较 ByteTrack 和 BoT-SORT，在工器具视频上实测 ID 切换率（ID Switch）后选定
- BoT-SORT 的人员 ReID 优势不直接适用于同型号工具（外观高度相似），工具外观特征关联属于可选增强模块，需用工具数据单独验证
- 跟踪仅在 L1/L2 检测帧上运行，不在无检测的中间帧做卡尔曼预测（见 §3 分层检测策略）

### 4.5 空间去重

**在线关联（主力）：门控 + 匈牙利匹配**

新一帧检测投影到全局坐标后：

1. **门控筛选**：按类别和空间范围过滤候选 GlobalObject，只考虑类别相同且质心距离在合理范围内的对象
2. **代价矩阵**：对每个 (GlobalDetection, GlobalObject) 对计算关联代价：

```
C_ij = w_p·C_position + w_o·C_overlap + w_s·C_size + w_c·C_class + w_a·C_appearance

C_position  : 质心欧氏距离
C_overlap   : 投影多边形 IoU（考虑单应性不确定性膨胀）
C_size      : 投影面积差异
C_class     : YOLO 类别置信度差异
C_appearance: 检测框 RoI 特征余弦距离（可选，用于区分同类别紧邻工具）
```

3. **匈牙利匹配**：求解一对一最小代价匹配
4. 未匹配的 Detection → 创建新 GlobalObject
5. 未匹配的 GlobalObject → 保留，记录"未观测"帧数
6. 连续多帧未观测的对象 → 可能已离开画面，标记为 inactive

**同帧互斥硬约束**：同一帧内同时出现的两个检测目标，其关联代价强制设为 ∞，永不允许归为同一个 global_id。

**历史共现互斥**：若两个 GlobalObject 曾在同一帧（或同一关键帧）中同时被观测到，则它们之间的合并代价永久设为 ∞。即使后续某一帧只看到其中一把钳子、空间上也离另一把很近，也不能将它们合并——因为它们曾被同时看到过，确凿是兩件独立工具。

**离线关联复核（辅助）：DBSCAN**

视频结束后，对所有全局观测点运行一次 DBSCAN 聚类复核，识别匈牙利阶段可能遗漏的重复对象。若发现两个对象的观测点在空间中高度混合 → 添加 `LIKELY_DUPLICATE` 复核标志，提醒人工确认。

### 4.6 类别投票

**投票权重要考虑多重质量因素**：

```
w = p_det · q_sharp · q_map · q_edge · q_size

p_det  : YOLO 检测置信度
q_sharp: 该帧清晰度分数（Laplacian 方差，归一化）
q_map  : 该帧单应矩阵质量（RANSAC 内点比例，归一化）
q_edge : 目标是否靠近图像边缘（边缘区域检测框不可靠，权重降低）
q_size : 目标有效像素尺寸（过小的目标分类不可靠，权重降低）
```

**避免连续帧伪重复投票**：

连续相邻帧的检测结果高度相关，不应作为独立证据（同一工具 10 帧检测 ≠ 10 次独立投票）。

```
逐帧检测
  → Track 内融合：同一 track 的 N 帧检测合并为 ≤3 次有效投票
    每次投票 = 该 track 中权重最高的帧；或按视角变化选取代表性帧
  → Track 的有效投票进入 GlobalObject
  → GlobalObject 累积所有关联 track 的投票
  → 投票结束后判定主状态：
     CONFIRMED    : 最高类别票数 > 阈值 且 类别分布集中
     TENTATIVE    : 观测数不足（新对象）
     UNCERTAIN    : 票数分散 或 最高票数不足
     同期检查 review_flags（可与主状态并存）
```

**同帧互斥&历史共现**约束在投票阶段持久生效（见 §4.5）。

### 4.7 覆盖判断

> 当前系统无绝对尺度信息，清点区域边界需操作员在视频结束后指定。拍摄期间不存在"完整区域覆盖率"，只有"已建图范围"。

#### 拍摄期间：实时建图范围显示

- 每帧视场多边形投影到全局网格，累计已覆盖网格
- 实时展示：已建图区域边界（外接矩形）、累计覆盖网格数、最近 N 秒新增覆盖面积
- 不做覆盖率判断——清点区域边界尚未确定，"覆盖率"无定义

#### 视频结束后：后处理覆盖检查

- 系统生成全局地图 → 操作员用鼠标框选或多边形圈定清点区域
- 系统计算：覆盖率 = 已覆盖网格数 / 清点区域总网格数
- 覆盖率 < 95% → 提示"覆盖不完整，建议补充扫描"

#### 清点区域边界

**第一版**：操作员在全景图上手动框选（唯一可靠方式）。
**后续可尝试**：工具分布外接多边形 + margin 自动估计、固定作业区域模板。

#### 尺度恢复（后续升级）

任一条件满足即可将相对网格映射到物理尺寸：已知参照物、人工标注距离、标定相机高度。

---

## 5. 项目结构

```
head_tool_counter/
├── configs/
│   ├── pipeline.yaml          # 全局 pipeline 参数
│   ├── detector.yaml          # YOLO 模型路径、L1/L2/L3分辨率(可配置)、置信度阈值、NMS
│   ├── tracker.yaml           # BoT-SORT 参数
│   ├── matcher.yaml           # ALIKED/LightGlue/SIFT、RANSAC
│   └── associator.yaml        # 门控半径、代价权重、匈牙利匹配阈值、共现约束、DBSCAN复核
│
├── models/
│   └── best.pt                # YOLO26 权重
│
├── core/
│   ├── video_reader.py        # 视频解码 + 分片缓冲
│   ├── sharpness_filter.py    # Laplacian 清晰度 + 运动模糊检测
│   ├── frame_buffer.py        # 帧缓存(只存灰度图+source_offset，~2MB/帧，避免620MB风险)
│   ├── keyframe_selector.py   # 三步关键帧:候选触发→质量接纳→RECOVERY
│   ├── feature_matcher.py     # ALIKED/LightGlue + SIFT兜底 + RANSAC
│   ├── homography_graph.py    # 在线单应性图 + 回环检测(两级检索) + 全局单应图优化
│   ├── loop_candidate_retriever.py  # 回环候选检索(全局描述子/间隔采样)
│   ├── detector.py            # YOLO26 L1/L2/L3 推理封装(仅 L1/L3 产生 RawDetection 并投影)
│   ├── tracker.py             # BoT-SORT/ByteTrack 跟踪
│   ├── global_projector.py    # 角点+中心点 → 全局坐标投影
│   ├── object_associator.py   # 门控+匈牙利关联 + 同帧/共现互斥 + 投票 + DBSCAN复核
│   ├── global_object_map.py   # 确认状态 + 可见性状态(分离) + 双ID管理
│   ├── coverage_map.py        # 建图范围 + 后处理覆盖检查
│   ├── status_panel.py        # 实时状态聚合
│   └── report_generator.py    # 示意图 + 证据帧 + CSV + JSON
│
├── apps/
│   ├── offline_scan.py        # CLI 离线批处理入口
│   └── api_server.py          # HTTP API 在线处理入口
│
└── outputs/
    ├── annotated_video/       # 标注视频
    ├── panorama/              # 全景拼接图
    ├── evidence/              # 证据帧（最佳视角）
    ├── maps/                  # 覆盖热力图
    └── reports/               # CSV + JSON 报告
```

---

## 6. 核心数据类型

```python
# ── 帧级数据 ──
@dataclass
class Frame:
    frame_id: int; timestamp: float; image: np.ndarray
    is_keyframe: bool
    sharpness_score: float          # Laplacian 清晰度
    mapping_quality: float          # 该帧单应矩阵质量(RANSAC 内点比例)
    camera_profile_id: str          # 相机标定配置标识

class KeyframeDecision(Enum):
    SKIP = "skip"
    CANDIDATE = "candidate"
    ACCEPTED = "accepted"
    RECOVERY = "recovery"
    RECOVERY_OK = "recovery_ok"
    LOST = "lost"

@dataclass
class KeyframeResult:
    decision: KeyframeDecision
    reason: str
    homography: np.ndarray | None
    num_inliers: int
    inlier_ratio: float
    reprojection_error: float
    spatial_distribution: float

# ── 不可变原始数据（回环重投影的数据源）──
@dataclass(frozen=True)
class RawDetection:
    """写入后永不修改。回环后从此重新投影。"""
    bbox: tuple[int,int,int,int]
    center: tuple[float,float]
    corners: tuple
    class_id: int; class_name: str
    confidence: float
    frame_id: int
    keyframe_id: int               # 不为 None：只有 L1 关键帧检测才创建 RawDetection
    sharpness: float
    source: str                    # "L1" 或 "L3"（L2 检测不产生 RawDetection）

# ── 跟踪结果 ──
@dataclass
class Track:
    track_id: int; bbox: tuple; class_id: int; class_name: str
    confidence: float; state: str; age: int
    detection_history: list[RawDetection]
    representative_frames: list[int]  # Track 内用于投票的代表性帧

# ── 全局投影结果 ──
@dataclass
class GlobalDetection:
    projected_corners: np.ndarray         # 四角点投影后的全局四边形 (4×2)
    projected_center: tuple[float,float]  # 检测框中心投影（辅助参考）
    polygon_centroid: tuple[float,float]  # 四边形质心（用作工具全局位置）
    polygon_area: float                   # 四边形面积
    class_id: int; class_name: str
    detection_confidence: float
    frame_id: int; keyframe_id: int
    track_id: int | None
    sharpness: float
    mapping_quality: float
    transform_version: int               # 用哪个版本的 H 矩阵投影
    source: str                           # "L1" | "L3"

# ── 全局对象 ──
@dataclass
class GlobalObject:
    # 双 ID 机制
    provisional_id: str                    # 临时编号（"P-003"），回环重建可能变化
    persistent_id: str | None              # 最终编号（"GO-001"），视频结束后锁定，永不变化
    class_name: str
    # 状态与复核标志（分离关注点）
    # 确认状态 — 工具"是什么"的置信度
    confirmation: str  # "TENTATIVE" | "CONFIRMED" | "UNCERTAIN" | "REJECTED"
    # 可见性状态 — 工具"现在是否在画面中"
    visibility: str    # "ACTIVE" | "INACTIVE"

    # 复核标志（与 confirmation/visibility 均可并存）
    review_flags: set[str]  # {"LIKELY_DUPLICATE", "CLASS_CONFLICT", "EDGE_ONLY", "LOW_CONFIDENCE"}
    review_flags: set[str]  # 复核标志: {"LIKELY_DUPLICATE", "CLASS_CONFLICT", "EDGE_ONLY", "LOW_CONFIDENCE"}
    confidence: float
    vote_distribution: dict[str, float]   # 类别 → 加权票数(浮点)
    observations: list[GlobalDetection]
    best_frame_id: int

    centroid_xy: tuple[float,float]        # 聚类中心
    position_covariance: np.ndarray        # 位置协方差(2×2)
    area_range: tuple[float,float]         # 投影多边形面积范围

    keyframe_ids: set[int]                 # 观测到该对象的关键帧
    track_ids: set[int]                    # 关联的 track
    co_observed_with: set[str]            # 历史共现互斥(global_id 集合)
    observation_count: int                 # 总观测次数
    uncertainty_reasons: list[str]         # UNCERTAIN 原因
    map_version: int                       # 由哪个版本的 GlobalObjectMap 生成

# 确认状态转换规则:
#   TENTATIVE  → CONFIRMED   (首次出现 + 多帧验证通过)
#   TENTATIVE  → UNCERTAIN   (观测数不足 或 投票分散)
#   TENTATIVE  → REJECTED    (连续N帧未观测，判定为误检)
#   UNCERTAIN  → CONFIRMED   (补充观测后投票收敛)
#   UNCERTAIN  → REJECTED    (人工确认误检)
#   CONFIRMED → REJECTED     (人工确认误检)

# 可见性状态转换规则:
#   ACTIVE → INACTIVE    (连续多帧未观测到)
#   INACTIVE → ACTIVE    (重新被观测到)

# review_flags（可与任一状态并存）:
#   LIKELY_DUPLICATE  : DBSCAN 复核发现可能重复
#   CLASS_CONFLICT    : 同一对象不同帧投票类别冲突
#   EDGE_ONLY         : 所有观测均在图像边缘
#   LOW_CONFIDENCE    : 综合置信度低于阈值

---

## 7. 关键模块接口摘要

| 模块 | 核心方法 | 说明 |
|------|---------|------|
| KeyframeSelector | `evaluate(frame) → KeyframeDecision` | 候选触发→质量接纳→RECOVERY |
| HomographyGraph | `add_keyframe()`, `get_current_transform()`, `optimize_homography_graph()`, `on_loop_closure(cb)` | 在线建图 + 回环→单应图优化→版本号+1→回调触发全量重投影 |
| Detector | `detect(image, level)` → list | 分层检测 L1/L2/L3。L1/L3 产生 RawDetection 并投影，L2 仅产生触发 dict |
| Tracker | `update(detections)` → list[Track] | ByteTrack / BoT-SORT，实测 ID Switch 后选定 |
| ObjectAssociator | `ingest(detections)`, `rebuild_all(raw, v)`, `final_review()` | 门控+匈牙利在线 + 回环全量重建 + DBSCAN复核 |
| GlobalObjectMap | `get_all()`, `assign_persistent_ids()`, `set_confirmation(id, s)`, `set_visibility(id, v)`, `set_review_flag(id, f)` | 双ID + 确认/可见性分离 + 复核标志 + 跨版本追踪 |
| CoverageMap | `update(frame, transform)`, `get_coverage(region)` → float | 后处理覆盖检查 |
| StatusPanel | `get_summary()` → dict | 实时状态聚合查询 |
| ReportGenerator | `generate(objects, map, coverage)` → dict | 证据帧/全景/CSV/JSON 产物生成 |

---

## 8. 部署模型

- 客户端：头戴相机，边拍边分片上传
- 服务端：云端 GPU，HTTP API 接收，流式处理，处理完即时返回结果
- 产物按需返回（标注视频、全景图、证据帧、CSV、覆盖热力图）

---

## 9. 未解决问题（后续迭代）

_无。设计已确认，可进入实现计划阶段。_
