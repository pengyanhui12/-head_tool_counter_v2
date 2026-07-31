# Head Tool Counter 项目结构与程序入口分析

> 分析日期：2026-07-30  
> 分析范围：当前仓库的目录、Python 模块依赖、可执行入口与主要处理链路。  
> 本次仅新增本文档，未修改任何现有源码、配置或数据文件。

## 1. 项目概览

该项目是一个面向视频的“作业工具检测与计数”系统。其核心流程是：

```text
读取视频
  -> 图像质量评估
  -> 分级目标检测（L1/L2/L3）
  -> 目标跟踪
  -> 关键帧选择与特征匹配
  -> 单应性图构建及全局坐标投影
  -> 跨帧对象关联、合并和确认
  -> 覆盖率/证据/全局拼图处理
  -> 生成 JSON、CSV 等结果
```

项目使用 Python 3.10 及以上版本，核心依赖包括 OpenCV、NumPy、SciPy、scikit-learn、PyYAML、Shapely 和 Ultralytics。`pyproject.toml` 将 `core*` 与 `apps*` 声明为可打包模块，但没有声明 `console_scripts` 命令行入口。

## 2. 顶层目录结构

```text
head_tool_counter/
├─ apps/                         # 应用层入口
│  ├─ offline_scan.py            # 正式离线扫描入口、主处理流程
│  └─ api_server.py              # FastAPI 在线服务入口
├─ core/                         # 核心算法与领域逻辑
├─ configs/                      # YAML 配置
├─ models/
│  └─ best.pt                    # Ultralytics/YOLO 模型权重
├─ tests/                        # pytest 单元与一致性测试
├─ docs/                         # 设计说明和实施计划
├─ debug_output/                 # 历史调试运行产物
├─ data/                         # 数据预留目录，目前为空
├─ tools/                        # 工具脚本预留目录，目前为空
├─ head_tool_counter.egg-info/   # 本地安装/打包生成的元数据
├─ debug_pipeline.py             # 独立的详细诊断流水线脚本
├─ pyproject.toml                # 项目与打包配置
├─ requirements.txt              # pip 依赖清单
├─ environment.yml
├─ environment.yaml              # Conda 环境配置
├─ pytest.ini                    # pytest 配置
└─ CLAUDE.md 等                  # 项目说明、变更记录和调试计划
```

说明：

- `.git/`、`.idea/`、`.pytest_cache/`、`.claude/` 属于版本控制、IDE、测试缓存或辅助工具目录，不参与业务运行。
- `debug_output/` 中包含帧截图、关键帧、匹配图、事件日志、报告等历史产物，不是源码。
- `data/` 与 `tools/` 当前没有业务文件。

## 3. 各主要目录职责

### 3.1 `apps/`：应用入口与流程编排

#### `apps/offline_scan.py`

这是正式的离线处理入口，同时承担主要的流水线装配和编排职责。

关键函数：

- `run_pipeline(video_path, config_dir, output_dir)`：创建各核心组件，逐帧处理视频，并生成最终结果。
- `main()`：解析 `--video`、`--config-dir` 和 `--output-dir` 参数，然后调用 `run_pipeline()`。

该文件末尾有标准入口保护：

```python
if __name__ == "__main__":
    main()
```

#### `apps/api_server.py`

这是在线 HTTP 服务入口。它创建 FastAPI 应用，并复用 `apps.offline_scan.run_pipeline()` 完成实际视频处理。

主要接口：

- `POST /scan`：上传视频并创建后台任务。
- `GET /job/{job_id}`：查询任务状态。
- `GET /job/{job_id}/report`：读取已完成任务的 JSON 报告。
- `GET /health`：健康检查。

直接执行该文件时，会通过 Uvicorn 监听 `0.0.0.0:8000`。

### 3.2 `core/`：核心算法与领域模块

`core/` 可以按职责划分为以下几组：

| 类别 | 主要模块 | 职责 |
|---|---|---|
| 基础数据模型 | `types.py`, `pipeline_result.py`, `exceptions.py` | 数据类、枚举、结果对象和领域异常 |
| 输入与质量 | `video_reader.py`, `quality_evaluator.py`, `frame_buffer.py` | 视频解码、帧质量判断、近期帧缓存 |
| 检测与融合 | `detector.py`, `detection_fusion.py` | 调用模型执行 L1/L2/L3 检测并融合结果 |
| 跟踪 | `simple_tracker.py` | 基于匈牙利算法、IoU 和中心距离的检测跟踪 |
| 关键帧与匹配 | `keyframe_selector.py`, `feature_matcher.py` | 决定关键帧、执行特征匹配并评估匹配质量 |
| 空间映射 | `homography_graph.py`, `global_projector.py`, `global_mosaic.py` | 管理单应性变换、映射到全局坐标、生成全局拼图 |
| 对象聚合 | `object_associator.py`, `global_object_map.py`, `merge_policy.py` | 跨关键帧关联对象、去重合并、维护全局对象 |
| 恢复机制 | `recovery_manager.py`, `loop_candidate_retriever.py` | 匹配失败后的恢复及候选检索 |
| 业务输出 | `report_generator.py`, `evidence_extractor.py`, `session_store.py` | 报告、证据图和会话数据持久化 |
| 状态与覆盖 | `coverage_map.py`, `status_panel.py` | 覆盖范围及运行状态信息 |
| 配置与调试 | `config_loader.py`, `debug_events.py` | 读取配置、记录结构化调试事件和性能统计 |

### 3.3 `configs/`：运行参数

配置文件按领域拆分：

- `pipeline.yaml`：输入帧率、检测间隔、关键帧间隔、质量阈值等。
- `detector.yaml`：模型路径、设备、L1/L2/L3 推理参数。
- `matcher.yaml`：特征匹配、RANSAC、内点和投影约束。
- `tracker.yaml`：跟踪器、丢失重激活、质量下降触发参数。
- `associator.yaml`：全局对象关联、确认条件和类别兼容性。
- `camera.yaml`：相机标定、中心区域和边缘映射策略。
- `coverage.yaml`：覆盖网格和目标覆盖率。

正式离线入口通过 `ConfigLoader` 读取这些配置。模型实际在 `apps/offline_scan.py` 中按项目根目录下的 `models/best.pt` 构造。

### 3.4 `tests/`：自动化测试

测试覆盖视频读取、帧缓存、特征匹配、关键帧选择、单应性图、全局投影、检测融合、跟踪状态机、对象绑定与合并、恢复关系、报告一致性等核心行为。

测试框架为 pytest；`pytest.ini` 将当前目录加入 `pythonpath`。

### 3.5 `debug_pipeline.py` 与 `debug_output/`

`debug_pipeline.py` 是根目录下的独立诊断脚本。它复刻并增强主流水线，额外输出：

- 逐帧和关键帧图像；
- 特征匹配图；
- `events.jsonl`；
- 关键帧、关联事件统计；
- 对象生命周期；
- 有效配置快照；
- 详细日志和一致性检查。

需要注意：该脚本没有 `main()` 和 `if __name__ == "__main__"` 保护，模块级代码会在文件被执行或被导入时立即运行。因此它应视为调试工具，而不是可安全导入的库模块。

`debug_output/` 是该脚本的历史输出目录。

## 4. 程序入口结论

项目存在三个可执行入口，但用途不同。

### 4.1 主要/正式入口：离线扫描

文件：`apps/offline_scan.py`

调用链：

```text
python -m apps.offline_scan
  -> main()
  -> argparse 解析参数
  -> run_pipeline(video_path, config_dir, output_dir)
  -> core 各组件
  -> output_dir/reports/report.json
     output_dir/reports/report.csv
     以及拼图、证据和会话文件
```

推荐从项目根目录以模块方式启动：

```powershell
python -m apps.offline_scan `
  --video "视频文件路径.mp4" `
  --config-dir "configs" `
  --output-dir "outputs"
```

因此，如果问题是“这个项目的主程序入口在哪里”，最准确的答案是：

> **主入口位于 `apps/offline_scan.py` 的 `main()`；核心业务入口是同文件中的 `run_pipeline()`。**

### 4.2 在线 API 入口

文件：`apps/api_server.py`

直接运行方式：

```powershell
python -m apps.api_server
```

也可以由 Uvicorn 导入应用对象：

```powershell
uvicorn apps.api_server:app --host 0.0.0.0 --port 8000
```

调用链：

```text
HTTP POST /scan
  -> scan()
  -> BackgroundTasks
  -> _run_job()
  -> apps.offline_scan.run_pipeline()
  -> 保存并返回报告
```

API 层不是另一套算法实现，它只是对离线流水线的服务化封装。

### 4.3 调试入口

文件：`debug_pipeline.py`

运行方式：

```powershell
python debug_pipeline.py
```

它是面向诊断和产物采集的脚本，并非正式应用入口。由于参数和处理代码主要位于模块顶层，运行前应先检查脚本开头定义的视频路径、配置和输出位置。

## 5. 正式离线流水线的组件关系

```text
VideoReader
  -> QualityEvaluator
  -> FrameBuffer
  -> Detector(L2)
  -> SimpleDetectionTracker.preview()
  -> KeyframeSelector
       ├─ ACCEPTED
       │    -> HomographyGraph
       │    -> Detector(L1/L3)
       │    -> DetectionFusion
       │    -> SimpleDetectionTracker.update()
       │    -> GlobalProjector
       │    -> ObjectAssociator
       │    -> CoverageMap
       ├─ RECOVERY
       │    -> RecoveryManager
       │    -> 恢复成功后进入投影与关联流程
       └─ 普通帧
            -> 必要时仅更新跟踪器

视频结束
  -> 末尾窗口关键帧补选
  -> ObjectAssociator.final_review()
  -> 分配持久对象 ID
  -> Global Mosaic / Evidence Extraction
  -> SessionStore
  -> ReportGenerator(JSON/CSV)
```

## 6. 结构上的重要观察

1. `apps/offline_scan.py` 同时承担“应用入口”和“流水线编排”两种职责，核心算法本身已较好地拆分到 `core/`。
2. 在线 API 完全复用 `run_pipeline()`，因此离线入口是整个系统最核心的可调用边界。
3. 项目未在 `pyproject.toml` 中配置安装后的命令行命令，所以当前应通过 `python -m ...` 或 Uvicorn 启动。
4. `debug_pipeline.py` 有导入即执行的副作用，不应被其他模块直接导入。
5. `data/` 和 `tools/` 是空的预留目录；`models/best.pt` 是实际运行所需的模型资产。
6. `debug_output/` 和 `head_tool_counter.egg-info/` 是生成内容，不属于核心源码结构。
7. `api_server.py` 使用 FastAPI、Uvicorn 和文件上传能力，但这些依赖未出现在当前 `pyproject.toml` 的主依赖列表中；部署 API 时需确保环境另外安装了相关包。

