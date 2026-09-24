# Math Model Cup

一个轻量、可复现的数学建模协作脚手架。

## 快速开始

```powershell
conda env create -f environment.yml
conda run -n math-model-cup python scripts/run_example.py
conda run -n math-model-cup pytest
```

赛题资料和原始数据不纳入 Git；需要交付的实验输出可纳入 `outputs/`。稳定的通用逻辑应从 Notebook 迁移到 `src/math_model_cup/`。
