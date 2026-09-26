# D Problem 1 Three-Method Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and run three reproducible solvers for D 题第一问 and produce separate feasible batch plans with a cross-method comparison.

**Architecture:** One focused module owns input loading, DEM geometry, physical calculations, candidate generation, three solvers and validation. A thin runner writes UTF-8 reports and performs safety-margin sensitivity analysis.

**Tech Stack:** Python, NumPy, pandas, Pillow, openpyxl, SciPy `milp`, pytest.

## Tasks

- [x] Add physical-model dataclasses, Excel/DEM loading, equivalent range, energy, time and safe-payload bisection.
- [x] Enumerate count-vector candidate trips and validate exact box coverage, capacity, energy and SOC.
- [x] Implement best-fit decreasing with deterministic merge improvement.
- [x] Implement exact count-state dynamic programming.
- [x] Implement three-stage lexicographic set-partitioning MILP.
- [x] Add synthetic and real-data tests comparing exact methods.
- [x] Add the real-data runner, separate trip CSV files, comparison, safe-payload, sensitivity and Markdown reports.
- [x] Run the full test suite and independently reconcile generated totals before completion.
