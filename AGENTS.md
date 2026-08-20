# Repository Guidelines

## 项目结构与模块组织

- `apps/` 存放应用入口：`offline_scan.py` 负责离线视频处理，`api_server.py` 提供 FastAPI 服务。
- `core/` 存放检测、跟踪、特征匹配、投影、对象关联、恢复及报告生成等核心逻辑。可复用的领域逻辑应放在此处，流程编排保留在 `apps/`。
- `configs/` 存放按子系统拆分的 YAML 配置，`models/` 存放检测模型权重，`tests/` 存放 pytest 测试。
- `debug_pipeline.py` 是独立诊断脚本。`debug_output/`、`apps/outputs*/`、报告、拼接图和证据图片均属于生成产物，不应视为源码。

## 构建、测试与本地开发

项目要求 Python 3.10 或更高版本。可运行 `conda env create -f environment.yaml` 创建 Conda 环境，或运行 `python -m pip install -r requirements.txt` 安装依赖。

- `python -m pytest`：按照 `pytest.ini` 执行完整测试套件。
- `python -m pytest tests/test_simple_tracker.py -v`：执行单个测试模块。
- `python -m apps.offline_scan --video input.mp4 --config-dir configs --output-dir outputs`：处理本地视频。
- `python -m apps.api_server`：启动 HTTP API。FastAPI、Uvicorn 和 multipart 支持尚未写入主依赖文件，API 环境可能需要单独安装。
- `python debug_pipeline.py`：运行详细诊断流程并生成调试产物。

## 编码风格与命名约定

遵循常规 Python 风格，使用四个空格缩进。模块、函数和变量使用 `snake_case`，类使用 `PascalCase`，常量使用 `UPPER_CASE`。公共接口应添加类型标注，路径操作优先使用 `pathlib.Path`，导入顺序与相邻文件保持一致。仓库暂未强制使用格式化器或静态检查器，因此不要进行无关的格式调整。

后续新增或修改的代码必须添加中文注释。注释应解释业务意图、关键算法、边界条件或不易理解的取舍；不要逐行翻译代码，也不要为显而易见的赋值和控制流程添加冗余注释。公共类、函数及复杂方法可使用简洁的中文文档字符串。

## 测试指南

pytest 按 `test_*.py` 文件、`Test*` 类和 `test_*` 函数进行发现。每项行为变更都应新增或更新测试，重点覆盖跟踪状态转换、同帧不变量、合并策略、单应性验证、输出身份和报告一致性。测试应保持确定性，并使用 pytest 临时目录隔离文件输出。当前没有强制覆盖率门槛。

## 提交与 Pull Request 指南

近期提交采用简短的 Conventional Commits 风格前缀，如 `fix:`、`feat:` 和 `docs:`。提交主题应使用祈使语气并保持单一职责，例如 `fix: preserve track binding during recovery`。Pull Request 应说明问题与解决方案，列出配置或输出契约变化，关联相关 Issue，并提供实际执行的测试命令及结果。可观察输出发生变化时，应附示例报告或调试图片。

## 安全与配置建议

不要提交凭据、私有视频、机器专用绝对路径、额外模型权重或生成输出。在新环境运行前，应检查 YAML 默认值和模型路径。
