# Math Model Cup Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Python 3.11 mathematical-modeling project scaffold while keeping supplied contest materials out of version control.

**Architecture:** A small `math_model_cup` package exposes deterministic helper functions. A runnable script consumes that package as an environment smoke test. Configuration, documentation, data folders, and generated outputs remain separate so exploratory work does not pollute stable code.

**Tech Stack:** Python 3.11, Conda, NumPy, Pandas, SciPy, scikit-learn, Matplotlib, Pytest, Git.

## Global Constraints

- Use Python 3.11 through `environment.yml`.
- Never track `A题/`, `B题/`, `C题/`, `D题/`, `F题/`, `评估报告/`, supplied `.docx` files, `tmp/`, `data/`, `outputs/`, or `.conda/`.
- Keep reusable code under `src/math_model_cup/`; use `scripts/` only for executable entry points.
- Run tests through `conda run -n math-model-cup pytest` after the environment is created.

---

### Task 1: Repository boundaries and Conda configuration

**Files:**
- Create: `.gitignore`, `environment.yml`, `README.md`
- Create: `data/raw/.gitkeep`, `data/processed/.gitkeep`, `outputs/.gitkeep`

**Interfaces:** Produces a reproducible environment named `math-model-cup` and an empty tracked project layout.

- [ ] **Step 1: Write the environment definition**

```yaml
name: math-model-cup
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - matplotlib
  - seaborn
  - pytest
```

- [ ] **Step 2: Create ignore rules and tracked empty folders**

```gitignore
A题/
B题/
C题/
D题/
F题/
评估报告/
*.docx
tmp/
.conda/
.venv/
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep
outputs/*
!outputs/.gitkeep
__pycache__/
.pytest_cache/
*.py[cod]
```

- [ ] **Step 3: Document bootstrap commands**

```markdown
conda env create -f environment.yml
conda run -n math-model-cup python scripts/run_example.py
conda run -n math-model-cup pytest
```

- [ ] **Step 4: Verify the staging boundary**

Run: `git add . && git status --short`

Expected: supplied contest directories and Word documents do not appear.

- [ ] **Step 5: Commit**

```bash
git add .gitignore environment.yml README.md data/raw/.gitkeep data/processed/.gitkeep outputs/.gitkeep
git commit -m "chore: set up reproducible project structure"
```

### Task 2: Deterministic baseline utility and smoke test

**Files:**
- Create: `src/math_model_cup/__init__.py`, `src/math_model_cup/metrics.py`
- Create: `scripts/run_example.py`, `tests/test_metrics.py`

**Interfaces:** Produces `mean_absolute_error(actual: Sequence[float], predicted: Sequence[float]) -> float`.

- [ ] **Step 1: Write the failing test**

```python
from math_model_cup.metrics import mean_absolute_error

def test_mean_absolute_error_returns_average_absolute_difference() -> None:
    assert mean_absolute_error([1.0, 2.0, 3.0], [2.0, 2.0, 5.0]) == 1.0
```

- [ ] **Step 2: Verify it fails**

Run: `conda run -n math-model-cup pytest tests/test_metrics.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'math_model_cup'`.

- [ ] **Step 3: Implement the minimal utility**

```python
from collections.abc import Sequence

def mean_absolute_error(actual: Sequence[float], predicted: Sequence[float]) -> float:
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    if not actual:
        raise ValueError("actual and predicted must not be empty")
    return sum(abs(value - estimate) for value, estimate in zip(actual, predicted)) / len(actual)
```

- [ ] **Step 4: Implement and run the smoke test**

```python
from math_model_cup.metrics import mean_absolute_error

if __name__ == "__main__":
    score = mean_absolute_error([1.0, 2.0, 3.0], [2.0, 2.0, 5.0])
    print(f"Example MAE: {score:.2f}")
```

Run: `conda run -n math-model-cup python scripts/run_example.py && conda run -n math-model-cup pytest`

Expected: `Example MAE: 1.00` and all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/math_model_cup scripts/run_example.py tests/test_metrics.py
git commit -m "feat: add reproducible baseline utility"
```

### Task 3: Publish the initialized repository

**Files:**
- Modify: `.git/config`
- Modify: Git remote `origin`

**Interfaces:** Produces a `main` branch pushed to `https://github.com/MeluYao/mathModelCup.git`.

- [ ] **Step 1: Configure commit identity**

```bash
git config user.name "MeluYao"
git config user.email "2101036267@qq.com"
```

- [ ] **Step 2: Configure and verify remote**

```bash
git remote add origin https://github.com/MeluYao/mathModelCup.git
git remote -v
```

- [ ] **Step 3: Push the main branch**

```bash
git branch -M main
git push -u origin main
```

Expected: GitHub displays the scaffold without supplied contest materials.

