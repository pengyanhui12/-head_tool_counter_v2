# SIFT Cache and End-Window Prefilter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 缓存近期图像的SIFT特征，并将尾窗全量30帧匹配缩减为最多6个时间分段候选。

**Architecture:** `FeatureMatcher` 内部维护按图像对象身份校验的有界LRU缓存，对外 `match()` 契约不变。`KeyframeSelector` 先执行确定性的时间分段质量预筛，再使用原有SIFT评分选择最终尾窗关键帧。

**Tech Stack:** Python 3.10+、OpenCV SIFT、NumPy、pytest、YAML。

## Global Constraints

- 不改变SIFT、BFMatcher、RANSAC、单应矩阵验收和对象计数规则。
- 新增或修改的代码必须添加解释业务意图和边界的中文注释。
- 默认 `feature_cache_size=4`、`end_window_match_candidates=6`，均支持配置回退。
- 不修改Global Mosaic和用户在 `offline_scan.py` 中的本地默认路径。
- 不执行Git提交或推送。

---

### Task 1: FeatureMatcher有界LRU缓存

**Files:**
- Modify: `core/feature_matcher.py`
- Modify: `tests/test_feature_matcher.py`

**Interfaces:**
- Extend: `FeatureMatcher.__init__(..., feature_cache_size: int = 4)`。
- Produce: `clear_feature_cache() -> None`。
- Preserve: `match(source_image, target_image) -> MatchResult`。

- [x] **Step 1: 写失败测试**

用可计数的fake detector替换 `_detector`，验证同一数组只调用一次 `detectAndCompute()`、容量超限按LRU淘汰、清空后重新提取、容量0每次重新提取，并验证负容量抛出 `ValueError`。

- [x] **Step 2: 验证RED**

Run: `python -m pytest tests/test_feature_matcher.py -q`

Expected: FAIL，原因是构造器不接受 `feature_cache_size` 或没有缓存行为。

- [x] **Step 3: 实现缓存**

使用 `OrderedDict[int, tuple[np.ndarray, tuple]]`。查询时同时检查 `id(image)` 和缓存图像 `is image`；命中后移到末尾。插入后按容量从头淘汰。容量0直接调用现有特征提取，`clear_feature_cache()` 清空字典。

- [x] **Step 4: 验证GREEN**

Run: `python -m pytest tests/test_feature_matcher.py -q`

Expected: 全部PASS。

---

### Task 2: 尾窗六段质量预筛

**Files:**
- Modify: `core/keyframe_selector.py`
- Modify: `tests/test_end_window_keyframes.py`

**Interfaces:**
- Extend: `KeyframeSelector.__init__(..., end_window_match_candidates: int = 6)`。
- Produce: `_prefilter_end_frames(end_frames: list[Frame]) -> list[Frame]`。

- [x] **Step 1: 写失败测试**

增加计数Matcher，构造30个有序帧，断言只匹配6次；验证6个候选覆盖6个连续时间分段、同分时选择较晚帧、配置30时匹配30次、空列表返回空、非正配置执行全量匹配。

- [x] **Step 2: 验证RED**

Run: `python -m pytest tests/test_end_window_keyframes.py -q`

Expected: FAIL，原因是仍对30帧逐一匹配或构造器缺少新参数。

- [x] **Step 3: 实现确定性预筛**

先按 `frame_id` 排序；当 `0 < candidate_count < frame_count` 时，用 `numpy.array_split` 等价的整数边界把序列分成候选数个非空连续分段。每段按 `(sharpness * exposure, sharpness, exposure, frame_id)` 取最大帧，随后仅对候选执行原有SIFT评分和最终top-2排序。

- [x] **Step 4: 验证GREEN**

Run: `python -m pytest tests/test_end_window_keyframes.py tests/test_keyframe_selector.py tests/test_keyframe_trigger_logic.py -q`

Expected: 全部PASS。

---

### Task 3: 配置接入、文档与对比验证

**Files:**
- Modify: `configs/matcher.yaml`
- Modify: `configs/pipeline.yaml`
- Modify: `apps/offline_scan.py`
- Modify: `tests/test_effective_config.py`
- Modify: `PROJECT_FLOW_REFERENCE.md`

**Interfaces:**
- Wire: `matcher.feature_cache_size -> FeatureMatcher`。
- Wire: `pipeline.end_window_match_candidates -> KeyframeSelector`。

- [x] **Step 1: 写配置失败测试**

验证正式配置加载后两个新值分别为4和6，并验证构建路径把配置传入对应对象。

- [x] **Step 2: 验证RED**

Run: `python -m pytest tests/test_effective_config.py -q`

Expected: FAIL，原因是新配置尚不存在或未接入。

- [x] **Step 3: 接入配置并更新手册**

添加两个YAML字段并在 `offline_scan.py` 构造器调用中传入；参考手册记录缓存只对同一数组对象有效、Frame图像只读约束、尾窗预筛算法及配置回退方法。

- [x] **Step 4: 运行相关与完整测试**

Run: `python -m pytest -q`

Expected: 全部PASS。

- [x] **Step 5: 同视频实际对比**

Run: `python -m apps.offline_scan --output-dir .test-output-sift-optimized --performance`

Expected: 计数仍为20；Feature Matching calls约19；生成的报告状态和分类计数与优化前临时验证一致；保存实际耗时差异。

- [x] **Step 6: 检查并清理**

运行 `git diff --check`，安全删除 `.test-output-sift-optimized` 和pytest临时目录，标记计划完成。
