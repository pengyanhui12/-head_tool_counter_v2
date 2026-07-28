# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

轨道交通作业工器具检测计数系统。头戴单目相机拍摄地面摆放的工具 → YOLO 检测 + SIFT 建图 + 跟踪 + 全局空间去重 → 输出工具类别及数量，支持人工复核低置信度目标。

## 常用命令

```bash
# 运行全部测试
pytest tests/ -q

# 运行单个测试文件
pytest tests/test_feature_matcher.py -v

# 离线处理（需先将模型放入 models/best.pt）
python apps/offline_scan.py --video <path.mp4>

# 安装依赖 + 可编辑模式
pip install -e .
```

## 核心架构

**按层构建**，下层不依赖上层：

```
Layer 0: types.py (所有 dataclass) + config_loader.py (YAML 读取) + configs/*.yaml
Layer 1: video_reader.py → quality_evaluator.py → frame_buffer.py
Layer 2: feature_matcher.py → homography_graph.py + keyframe_selector.py
Layer 3: detector.py → detection_fusion.py → simple_tracker.py
         → global_projector.py → global_object_map.py + object_associator.py → coverage_map.py
Layer 4: status_panel.py + report_generator.py + evidence_extractor.py + session_store.py
         ├── apps/offline_scan.py (CLI 编排)
         └── apps/api_server.py  (FastAPI 服务)
```

## 关键约定

**检测层级**：L1(1280, 关键帧, 主检测) / L2(640, 每3帧, 触发+跟踪) / L3(1280, 按需, 小目标补充)。
L2 不投影、不参与计数。L1+L3 在关键帧上先融合 (`DetectionFusion.fuse`)，再生成不可变 `RawDetection`。

**单应矩阵方向**：全项目统一 `H_source_to_target` 命名。`FeatureMatcher.match(src, dst)` 返回 `H_src_to_dst`。
`HomographyGraph` 链式累积: `H_curr_to_global = H_prev_to_global @ H_curr_to_prev`。

**状态分离**：`ConfirmationStatus`(TENTATIVE/CONFIRMED/UNCERTAIN/REJECTED) + `VisibilityStatus`(ACTIVE/INACTIVE) + `ReviewFlag`(LIKELY_DUPLICATE等) 三轴独立。

**双 ID**：`provisional_id`(在线变化) → `persistent_id`(视频结束后锁定)。

**关键帧三步**：候选触发(interval/L2/tracker信号) → FeatureMatcher 质量接纳 → RECOVERY。

**回环**：MVP 只记录 `LoopConstraint`，`optimize_homography_graph()` 抛出 `NotImplementedError`，不做空操作。

**配置**：所有阈值/分辨率/权重都在 `configs/*.yaml` 中，代码不写死。

## 当前已知问题

1. 投票全为 0 — `_update_object` 中投票权重计算后 `vote_distribution` 累积为 0，导致无 CONFIRMED
2. 关键帧判定不均 — 同一个 30s 视频 149 个 KF(过密)，另一个 280 帧仅 2 个(过少)
3. 对象碎片化 — 空间关联未有效合并，同类同名对象重复创建
4. 大部分 `.py` 文件缺少中文注释
