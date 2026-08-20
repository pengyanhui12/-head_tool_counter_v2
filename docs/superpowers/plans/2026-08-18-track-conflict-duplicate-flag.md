# Track 冲突重复标记修复实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 防止共享 Track 的冲突对象仅因合并被共现或帧重叠规则阻止而错误获得 `LIKELY_DUPLICATE`。

**Architecture:** 保留 `MergePolicy` 的安全检查顺序和审计原因，只调整 `ObjectAssociator` 对被阻止合并对象的复核标记。空间接近规则仍可独立添加 `LIKELY_DUPLICATE`。

**Tech Stack:** Python 3.10+、pytest、NumPy。

## Global Constraints

- 新增或修改的代码必须添加解释业务意图的中文注释。
- 不改变计数、Track 绑定、合并安全条件和空间重复阈值。
- 不修改用户已有的无关工作区变更。

---

### Task 1: 修正共享 Track 冲突对象的重复标记

**Files:**
- Modify: `core/object_associator.py`
- Test: `tests/test_object_merge_policy.py`

**Interfaces:**
- Consumes: `GlobalObject.review_flags`、`MergePolicy.can_merge()` 的合并判定。
- Produces: `_merge_by_shared_track_safe()` 中准确区分 Track 冲突与疑似重复的行为。

- [x] **Step 1: 写入失败回归测试**

构造两个共享 Track、均带 `TRACK_CONFLICT` 且曾同帧共现的已确认对象；执行 `final_review()` 后断言保留 `TRACK_CONFLICT`，但不添加 `LIKELY_DUPLICATE`。

- [x] **Step 2: 验证测试因当前错误行为失败**

Run: `python -m pytest tests/test_object_merge_policy.py::test_track_conflict_with_cooccurrence_does_not_imply_likely_duplicate -v`

Expected: FAIL，原因是对象仍包含 `ReviewFlag.LIKELY_DUPLICATE`。

- [x] **Step 3: 编写最小实现**

在 `_merge_by_shared_track_safe()` 中以对象当前是否带 `TRACK_CONFLICT` 为准；任一对象存在该标记时，共享 Track 合并阻止路径不追加 `LIKELY_DUPLICATE`。不改变 `_mark_close_duplicates()`。

- [x] **Step 4: 验证目标测试和相关测试**

Run: `python -m pytest tests/test_object_merge_policy.py tests/test_track_object_binding.py -q`

Expected: 全部 PASS。

- [x] **Step 5: 验证完整测试套件**

Run: `python -m pytest -q`

Expected: 全部 PASS。
