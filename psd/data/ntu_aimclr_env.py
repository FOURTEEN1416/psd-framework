"""external/AimCLR 定位器 — worktree 检出环境下的只读回退解析。

背景: external/ 属 gitignore（AGENTS.md 硬规则 5），git worktree 检出不包含
AimCLR 克隆；NTU 链路（训练/评估/协议保真测试）全部只读消费该目录。

约定: 与 `.venv 统一用主仓绝对路径`同构——本地没有时回退主检出
（D:\\Desktop\\psd-framework），多候选时精确名优先、其余排序稳定遍历。
禁止对 external/ 做 Junction（卸窗脚本不识别，存在误删风险）。
"""
from __future__ import annotations

from pathlib import Path


def candidate_aimclr_roots(repo_root: Path) -> list[Path]:
    """生成有序候选列表：本仓优先，其次同名主检出，再其余 psd-framework* 兄弟目录。"""
    parent = repo_root.parent
    siblings = [
        p
        for p in sorted(parent.glob("psd-framework*"))
        if p.is_dir() and p.resolve() != repo_root.resolve()
    ]
    ordered = []
    main_exact = parent / "psd-framework"
    if main_exact in siblings:
        siblings.remove(main_exact)
        siblings.insert(0, main_exact)
    for base in [repo_root, *siblings]:
        ordered.append(base / "external" / "AimCLR")
    return ordered


def resolve_aimclr_root(repo_root: Path | None = None) -> Path:
    """返回第一个实际存在的 external/AimCLR；全缺时 FileNotFoundError 列出已查候选。"""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    repo_root = Path(repo_root)

    candidates = candidate_aimclr_roots(repo_root)
    for cand in candidates:
        if cand.is_dir():
            return cand

    raise FileNotFoundError(
        "external/AimCLR 未找到（只读消费目录，gitignore 不随 worktree 走）。\n"
        "已检索候选:\n  " + "\n  ".join(str(c) for c in candidates)
    )
