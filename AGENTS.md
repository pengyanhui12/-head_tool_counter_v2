# Repository Guidelines

## 项目结构与模块组织

- `apps/`：应用入口。`offline_scan.py` 用于离线视频处理，`api_server.py` 提供 FastAPI 服务。
- `core/`：核心业务代码，包括检测、跟踪、特征匹配、单应性映射、对象关联、恢复和报告生成。
- `configs/`：按子系统拆分的 YAML 配置。修改参数时应说明对运行行为的影响。
- `models/best.pt`：检测模型权重。不要随意提交额外的大型模型或生成文件。
- `tests/`：pytest 测试，文件统一命名为 `test_*.py`。
- `debug_pipeline.py`：独立诊断脚本；`debug_output/` 保存运行产物，不属于源码。

## 构建、测试与本地运行

项目要求 Python 3.10 或更高版本。安装依赖：

```powershell
python -m pip install -r requirements.txt
```

运行全部测试或单个测试文件：

```powershell
python -m pytest
python -m pytest tests/test_simple_tracker.py -v
```

执行离线扫描：

```powershell
python -m apps.offline_scan --video input.mp4 --config-dir configs --output-dir outputs
```

使用 `python -m apps.api_server` 启动 API。FastAPI、Uvicorn 和文件上传支持目前未列入主依赖文件，运行前可能需要单独安装。

## 编码风格与命名约定

使用四空格缩进并遵循常规 Python 风格：函数和变量使用 `snake_case`，类使用 `PascalCase`，常量使用 `UPPER_CASE`。公共接口应添加类型标注。领域逻辑放在 `core/`，`apps/` 主要负责组件装配和流程编排。路径操作优先使用 `pathlib.Path`。仓库目前没有强制格式化器或静态检查器，因此应保持与相邻代码一致，并合理组织导入。

## 测试指南

pytest 配置位于 `pytest.ini`。测试文件、测试类和测试函数分别遵循 `test_*.py`、`Test*` 和 `test_*` 命名。每项行为变更都应新增或更新测试，重点覆盖跟踪状态转换、同帧不变量、对象合并、单应性验证及报告一致性。当前没有强制覆盖率门槛。

## 提交与 Pull Request

近期提交常用 `fix:`、`docs:` 等简短前缀。提交信息应使用祈使式摘要，例如 `fix: preserve track binding during recovery`，并保持单一职责。Pull Request 应说明问题、实现方案、配置变化以及实际执行的测试命令和结果。关联相关 Issue；若输出行为变化，应附示例报告或调试图片。

## 配置与生成文件

不要提交凭据、私有视频、临时输出或仅适用于个人机器的绝对路径。在新环境运行前检查 YAML 默认值和模型路径。当目录结构、CI、格式化工具或开发流程变化时，应同步更新本指南。
