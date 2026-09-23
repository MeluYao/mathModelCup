# AI Collaboration Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a standalone Chinese guide that lets Codex and Claude Code safely help non-Git users collaborate on this repository.

**Architecture:** A single Markdown document is the source of truth. It provides copyable AI prompts, a plain-language workflow, repository-specific safety rules, and recovery instructions.

**Tech Stack:** Markdown, Git, Conda, Codex, Claude Code.

## Global Constraints

- Use simplified Chinese and tool-neutral language.
- Include no account credentials, tokens, or passwords.
- Never instruct users or AI to upload `A题/`, `B题/`, `C题/`, `D题/`, `F题/`, `评估报告/`, `.docx`, `tmp/`, data, or output artifacts.
- Use the existing Python 3.11 Conda convention in `environment.yml`.

---

### Task 1: Write the AI collaboration guide

**Files:**
- Create: `docs/AI_COLLABORATION_GUIDE.md`

**Interfaces:**
- Consumes: a member's natural-language request and their local clone of `MeluYao/mathModelCup`.
- Produces: a safe AI-assisted workflow from local work through Pull Request.

- [ ] **Step 1: Add a startup prompt for either Codex or Claude Code**

```markdown
请先阅读 AI_COLLABORATION_GUIDE.md 和 README.md。你负责执行 Git 操作，但在删除文件、发送邀请、修改权限、公开仓库或推送前必须向我确认。赛题资料、数据和输出不得加入 Git。
```

- [ ] **Step 2: Document roles, branch workflow, and PR handoff**

Include: synchronize `main`; use an intent-based branch name; run tests; let AI show `git status`; commit; push; create a PR; request review; do not merge without approval.

- [ ] **Step 3: Document project-specific modeling and environment conventions**

Include: exploratory work belongs in `notebooks/`; stable logic belongs in `src/math_model_cup/`; Conda commands use `environment.yml`; data and outputs remain untracked.

- [ ] **Step 4: Document recovery prompts**

Include prompts for merge conflicts, accidental staging, a failed Conda environment creation, and a failed test.

- [ ] **Step 5: Validate required safety terms**

Run: `rg -n "赛题|不.*Git|Pull Request|Conda|冲突|确认" docs/AI_COLLABORATION_GUIDE.md`

Expected: each workflow and safety topic appears at least once.

- [ ] **Step 6: Commit and push**

```bash
git add docs/AI_COLLABORATION_GUIDE.md docs/superpowers/plans/2026-09-23-ai-collaboration-guide.md
git commit -m "docs: add AI collaboration guide"
git push
```
