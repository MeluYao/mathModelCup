# 数学建模协作脚手架设计

## 目标

为 `mathModelCup` 建立一个轻量、可复现的 Python 数学建模项目基础，适用于赛题尚未确定的阶段。

## 范围

- 使用 Conda 管理 Python 3.11 环境与常用数值计算、建模及可视化依赖。
- 提供通用源码包、实验 Notebook 目录、数据与输出目录，以及最小运行示例。
- 初始化 Git，并将代码、配置、文档纳入版本控制。
- 不跟踪现有赛题目录、评估报告、赛题 Word 文档、临时目录、数据、生成结果和本地环境文件。

## 目录结构

```text
src/math_model_cup/   # 通用数据、模型和评估工具
scripts/              # 可直接运行的实验入口
notebooks/            # 探索性分析
data/raw/             # 原始数据（不跟踪）
data/processed/       # 处理后数据（不跟踪）
outputs/              # 图表、模型、报告等产物（不跟踪）
tests/                # 通用工具的测试
```

## 协作规则

- `main` 只接收经过 Pull Request 审查的变更。
- 每个赛题方案使用独立功能分支；Notebook 用于探索，稳定逻辑迁移到 `src/`。
- 依赖仅通过 `environment.yml` 更新，避免成员间环境漂移。

## 验证

在新建环境中运行 `python scripts/run_example.py`，应输出一项可重复计算的示例指标；测试命令为 `pytest`。
