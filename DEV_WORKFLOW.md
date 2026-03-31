# DEV_WORKFLOW.md — amaebi 开发流程（可移植版）

## 概览

```
需求 → Claude Code 开发 → 独立测试 → 通过？ → push + PR → CI → merge
              ↑                           ↓ 不通过
              └──── Claude Code 修复 ────┘
```

## 关键原则

- **Claude Code**：只负责开发（写代码 + commit）
- **测试 agent**：只负责测试/review（独立会话）
- **主 agent**：只负责协调、push、PR
- **职责隔离**：开发不测、测试不改、协调不写代码
- 编译/测试统一通过项目脚本执行（避免环境漂移）

---

## 目录与分支策略（git worktree）

- 主仓库保持在 `master/main`，不直接开发
- 每个任务使用独立 worktree：`<worktree-base>/<task-name>`
- 分支命名建议：
  - `feat/<name>`
  - `fix/<name>`
  - `refactor/<name>`

### 创建 worktree（新分支）

```bash
git fetch origin
git worktree add <worktree-base>/<task-name> -b feat/<task-name> origin/master
```

### 使用已有远程分支

```bash
git fetch origin
git worktree add <worktree-base>/<task-name> origin/feat/<task-name> \
  || (cd <worktree-base>/<task-name> && git pull)
```

### 清理 worktree

```bash
git worktree remove <worktree-base>/<task-name>
git branch -d feat/<task-name>
```

---

## 标准阶段

### 1) 需求定义
输出：目标、范围、约束、验收标准。

### 2) 开发（Claude Code）
只做：
- 读代码、实现功能、提交 commit

不做：
- 不跑测试
- 不 push
- 不创建 PR

### 3) 测试（独立测试 agent）
在独立会话执行统一测试脚本，输出测试报告，至少包括：
1. 编译检查
2. 单元/集成测试
3. 静态检查（clippy/等价）
4. 边界场景与逻辑核对
5. commit message 规范检查（conventional commits）

### 4) Review 与修复循环
测试不通过 → Claude Code 修复并提交 → 回到阶段 3，直到通过。

### 5) Push + PR
- push 到任务分支
- 新建或更新 PR

### 6) CI + Merge
- CI 通过
- 人工 review 通过
- merge

---

## 测试规则

统一使用项目脚本，不手拼命令：

```bash
# 标准测试（CI 对齐）
./scripts/test.sh

# 需要额外集成测试时（例如 docker/ignored）
./scripts/test.sh --docker
```

> 若仓库里有 `#[ignore]` 测试，必须有明确触发方式（如 `--docker`），并纳入提交前检查。

---

## 并行开发

- 多任务并行时，每个任务独立 worktree
- 不同任务可并行开发/测试，互不干扰

---

## 提交规范

- commit 使用 conventional commits
- 一次提交只做一类改动
- PR 描述聚焦：改了什么、为什么、如何验证
