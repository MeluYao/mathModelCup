# Math Model Cup

一个轻量、可复现的数学建模协作脚手架。

## 快速开始

```powershell
conda env create -f environment.yml
conda run -n math-model-cup python scripts/run_example.py
conda run -n math-model-cup pytest
```

赛题资料、原始数据和实验输出不纳入 Git。稳定的通用逻辑应从 Notebook 迁移到 `src/math_model_cup/`。
