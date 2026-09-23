# AI Collaboration Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a standalone Chinese instruction file that tells Codex and Claude Code how to operate this repository for two non-Git users.

**Architecture:** A single Markdown document is the source of truth and addresses AI directly. It maps each member account to one persistent branch, mandates Chinese commit messages, and defines repository-specific safety and recovery behavior.

**Tech Stack:** Markdown, Git, Conda, Codex, Claude Code.

## Global Constraints

- Use simplified Chinese and tool-neutral language.
- Include no account credentials, tokens, or passwords.
- Treat the AI as the only reader; do not teach members to run Git commands.
- Map `yf82289878` to branch `yf82289878` and `lyn18302` to branch `lyn18302`.
- Require every commit message to use simplified Chinese.
- Never instruct users or AI to upload `A题/`, `B题/`, `C题/`, `D题/`, `F题/`, `评估报告/`, `.docx`, `tmp/`, data, or output artifacts.
- Use the existing Python 3.11 Conda convention in `environment.yml`.

---

### Task 1: Rewrite the AI collaboration guide

**Files:**
- Modify: `docs/AI_COLLABORATION_GUIDE.md`

**Interfaces:**
- Consumes: one of the member accounts `yf82289878` or `lyn18302` and their local clone of `MeluYao/mathModelCup`.
- Produces: a safe AI-assisted workflow from local work through Pull Request.

- [ ] **Step 1: Replace member-facing teaching with direct AI instructions**

```markdown
你是本仓库的 AI 执行者。先识别成员账号：yf82289878 使用同名分支 yf82289878；lyn18302 使用同名分支 lyn18302。你负责所有 Git 操作，不要求成员输入 Git 命令。
```

- [ ] **Step 2: Document fixed account branches, Chinese commits, and PR handoff**

Include: synchronize `main`; create or switch to the account-named branch; never create task branches; run tests; show `git status`; use a simplified-Chinese commit message; push; create a PR; request review; do not merge without approval.

- [ ] **Step 3: Document project-specific modeling and environment conventions**

Include: exploratory work belongs in `notebooks/`; stable logic belongs in `src/math_model_cup/`; Conda commands use `environment.yml`; data and outputs remain untracked.

- [ ] **Step 4: Document recovery prompts**

Include prompts for merge conflicts, accidental staging, a failed Conda environment creation, and a failed test.

- [ ] **Step 5: Validate required safety terms**

Run: `rg -n "yf82289878|lyn18302|中文|赛题|Pull Request|Conda|冲突|确认" docs/AI_COLLABORATION_GUIDE.md`

Expected: each workflow and safety topic appears at least once.

- [ ] **Step 6: Commit and push**

```bash
git add docs/AI_COLLABORATION_GUIDE.md docs/superpowers/plans/2026-09-23-ai-collaboration-guide.md
git commit -m "文档：调整 AI 协作与账号分支规范"
git push
```
