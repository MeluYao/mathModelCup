# 问题一结果入口

- [完整模型建立与求解报告](问题一_模型建立与求解.md)：物理模型、方法推导、45项安全载荷、18架次逐箱方案、目标权衡与安全余量敏感性。
- [离线图文概览](index.html)：可在浏览器打开。
- [Excel结果工作簿](../outputs/01a0d71b-q1/问题一结果.xlsx)：包括提交模板同名九列表、最大安全载荷、策略与余量对照、逐箱归属、逐区汇总及计算口径。
- [独立核验记录](verification.json)：105项通过；15个服务区MILP均为最优且最优间隙为0。

基准返航安全余量20%，主目标依次为架次、能耗、累计作业时间。主方案18架次，能耗59.1322603269 kWh，累计作业32776.5167827秒（9.1045879952小时），B型9架次、C型9架次。最低返航SOC为23.0892483877%。

累计作业时间包含准备、装载、飞行、交接，不是第二问多机并行调度的完工时间。能耗推导约定和适用范围见完整报告。

## 复现

在项目根目录执行 `./run_q1.ps1`，依次求解、独立整数规划核验及绘图。也可使用已有Python运行：

```text
python solve_q1.py
python verify_q1.py
python visualize_q1.py
```

依赖与预处理相同：numpy、pandas、scipy、matplotlib等，版本见preprocessing/environment.json。本项目已安装的补充依赖在.preprocess_deps。

Excel使用`@oai/artifact-tool`生成，入口为`q1_results/workbook_build/build.mjs`；其node_modules链接到本机Codex提供的Node依赖。先完成上述Python流程，再用配套Node执行此文件即可更新工作簿。

`tables/`保留完整计算精度；`figures/`包含PNG与SVG；输入校验和见input_manifest.json。主方案完整货箱列表位于tables/Q1_单点组批.csv，未修改原始附件或原始提交模板。
