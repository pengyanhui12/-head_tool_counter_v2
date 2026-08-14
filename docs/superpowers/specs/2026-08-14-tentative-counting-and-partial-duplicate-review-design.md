# Tentative 计数与局部重复复核设计

## 背景

离线扫描结果中，`GO-0024 headlamp` 仅有一次观测且对象置信度为 0。证据框覆盖的是一个已确认头灯 `GO-0007` 的头带局部，但它仍被分配了持久 ID 并计入正式总数。这类低证据局部框会造成重复计数。

当前后处理只自动合并共享 track 的对象。无共享 track 的同类近邻只在质心距离小于固定阈值时标记为疑似重复；`GO-0024` 与 `GO-0007` 的全局质心距离约为 44 px，超过当前 30 px 阈值，因此未被标记。简单扩大阈值会增加相邻同类实物被误合并的风险。

## 目标

1. `TENTATIVE` 对象不再增加正式计数。
2. 保留所有 `TENTATIVE` 对象及其证据，供人工复核。
3. 对疑似属于已确认对象局部的 tentative 给出可审计的候选归因，但不自动合并。
4. 避免把桌面上相邻摆放的多个同类实物误判为同一对象。
5. JSON、CSV、控制台和 API 使用一致的计数口径。

## 非目标

- 本阶段不解决 `GO-0013`、`GO-0019` 的类别误判。
- 本阶段不调整检测器类别模型。
- 本阶段不扩大通用对象合并距离，也不把 tentative 自动合入 confirmed。
- 本阶段不解决全局拼接图的几何漂移。

## 计数模型

对象输出分为两个集合：

- `counted_objects`：正式计数对象，仅包含 `CONFIRMED` 和 `UNCERTAIN`。
- `review_candidates`：待复核候选，仅包含 `TENTATIVE`。

`REJECTED` 仍不属于上述两个集合，并继续通过拒绝记录保留审计信息。

报告中的 `total_objects` 改为 `counted_objects` 的数量。新增 `review_candidate_count`，防止把“正式数量”和“仍保留的候选记录数量”混为一谈。为减少兼容性歧义，所有输出层必须调用相同的分类函数，而不是各自过滤。

对于当前样本，预期汇总为：

```text
Counted objects: 21
Review candidates: 3
Likely partial duplicates: 1
```

这里的 21 是保守自动计数，不代表人工真值。`GO-0013` 和 `GO-0019` 是真实物体但目前证据不足、类别错误，留在复核区而不冒险进入正式计数。

## 局部重复候选判定

判定仅用于添加复核提示，不改变对象状态、不转移观测、不自动合并。

### 候选范围

只比较：

- 一个 `TENTATIVE` 对象；
- 一个 `CONFIRMED` 对象；
- 两者当前类别相同。

### 必要条件

tentative 只有在以下条件全部满足时，才可标记为 `LIKELY_PARTIAL_DUPLICATE`：

1. **低证据条件**：观测数或关键帧数未达到确认阈值。
2. **无独立共现证据**：两对象没有在同一帧以可分离边界框同时出现。若存在可靠同帧共现，必须视为两个独立实物。
3. **局部包含证据**：在可比较的原始帧证据中，较小框的大部分面积落在较大框内。使用 `intersection / smaller_box_area`，避免普通 IoU 因大小差异而过低。
4. **尺度归一化邻近**：全局质心距离相对于 confirmed 对象的代表尺度足够小。固定像素距离只能作为上限保护，不能作为唯一依据。
5. **唯一候选优势**：最佳 confirmed 候选必须明显优于第二候选；否则标记为歧义候选，不指定归属。

### 相邻同类保护

以下任一情况出现时，不得标记为确定的局部重复归因：

- 两对象有可靠的同帧独立共现记录。
- tentative 框与 confirmed 历史框没有包含关系，仅仅质心接近。
- tentative 更像完整尺寸的独立实例，而不是局部框。
- 多个 confirmed 候选得分接近。
- 映射质量或边缘质量低于可配置下限，导致全局位置不可信。

这组保护保证桌面上并排的多个头灯不会因为距离近而被归为同一对象。

## 候选评分与歧义处理

通过独立、可测试的评估函数计算候选证据，输入两个对象及其观测，输出结构化结果：

```text
decision: likely_partial_duplicate | ambiguous | no_match
candidate_id: optional provisional/persistent ID
containment_score
normalized_distance
co_occurrence_blocked
mapping_quality
reason
```

第一版采用明确门限组合，不引入学习模型。若没有可比较的同帧/邻近帧边界框，不能仅凭全局质心做肯定归因，应返回 `no_match` 或 `ambiguous`。

当多个候选均通过必要条件，但最佳分数没有达到配置的领先幅度时：

- 添加 `AMBIGUOUS_DUPLICATE_CANDIDATE`；
- 不填写 `likely_partial_duplicate_of`；
- 保留候选 ID 列表和评分供复核。

## 数据模型与输出

为 tentative 复核记录增加以下审计字段：

```json
{
  "persistent_id": null,
  "provisional_id": "P-0024",
  "confirmation_status": "tentative",
  "counted": false,
  "review_flags": ["likely_partial_duplicate"],
  "likely_partial_duplicate_of": "GO-0007",
  "duplicate_evidence": {
    "containment_score": 0.86,
    "normalized_distance": 0.31,
    "reason": "low_evidence_partial_box_near_confirmed_object"
  }
}
```

示例分数仅用于说明字段格式；实际输出必须填写本次判定的计算值。

JSON 顶层至少区分：

- `total_objects`：正式计数；
- `confirmed_count`；
- `uncertain_count`；
- `review_candidate_count`；
- `likely_partial_duplicate_count`；
- `objects`：正式计数对象；
- `review_candidates`：tentative 对象；
- `rejected_objects`。

CSV 可使用单表，但必须包含 `counted`、`review_status`、`likely_partial_duplicate_of` 等列。控制台需要分别打印正式计数和复核候选数。

`TENTATIVE` 不分配正式 `GO-*` 持久 ID，仅保留 `P-*` 临时 ID。证据文件使用临时 ID 命名，避免文件名暗示其已进入正式对象集合。

## 配置

新增配置应集中在关联或复核子系统，并说明行为影响：

- 局部包含比例下限；
- 归一化质心距离上限；
- 绝对距离保护上限；
- 最低映射质量；
- 最佳候选相对第二候选的最小领先幅度。

默认值应从当前样本和现有测试数据推导，不能仅针对 `GO-0024` 手工拟合。配置调整只影响复核标记，不影响对象自动合并。

## 处理流程

1. 完成所有帧的在线关联和既有安全合并。
2. 重新评估对象确认状态。
3. 对每个 tentative 搜索同类 confirmed 候选。
4. 应用共现、包含、尺度、映射质量和唯一性保护。
5. 写入复核标记及结构化证据，不改变对象归属。
6. 将对象划分为正式计数、复核候选和拒绝对象。
7. JSON、CSV、控制台及 API 从同一划分结果生成输出。

## 测试策略

至少覆盖以下回归场景：

1. 单帧头带局部框邻近既有头灯，标记为局部重复候选且不计数。
2. 两个同类头灯在同一帧独立出现，即使距离很近也不得归因或合并。
3. 两个同类对象没有同帧共现，但边界框不存在包含关系，不得仅凭距离归因。
4. 一个 tentative 同时接近两个 confirmed 且得分接近，标记为歧义。
5. 低映射质量观测不得产生肯定归因。
6. tentative 保留在 JSON、CSV 和证据输出中，但不分配持久 ID、不进入 `total_objects`。
7. `total_objects == confirmed_count + uncertain_count`。
8. JSON、CSV、控制台与 API 的正式计数和复核候选数量一致。
9. 现有共享 track 安全合并和同帧不变量测试继续通过。

## 验收标准

在目标视频上：

- `GO-0024` 不进入正式计数；
- 它保留为 tentative 复核候选，并指向 `GO-0007`，同时附带可解释证据；
- 相邻摆放的四个独立头灯仍保持为四个 confirmed 对象；
- `GO-0013`、`GO-0019` 保留为普通复核候选，不因本功能被错误合并；
- 正式计数、候选数及各输出载体之间不存在口径差异。
