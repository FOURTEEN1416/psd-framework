#!/usr/bin/env python
"""W9 Phase A 补充件 — NTU60 不稳代理下的选择性稳健下载。

背景：fetch_ntu_data.py 的整树 download_folder 在不稳代理下随机中断
（2026-08-24 实测两次分别断在不同子目录）。本脚本两阶段：
  1) 枚举官方根目录（skip_download，带整体重试），落盘清单 manifest；
  2) 仅筛选 ntu60_frame50 子树文件逐个 gdown.download（resume=True 断点续传，
     每文件最多 N 次重试），已存在且大小吻合的自动跳过——可反复运行直至补齐。

用法：
  python scripts/ntu_selective_fetch.py --proxy http://127.0.0.1:17890 \
      [--attempts 6] [--include released_model]
默认只下 ntu60_frame50 子树；--include 可追加其他前缀（如 released_model）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT_URL = "https://drive.google.com/drive/folders/1VnD3CLcD7bT5fMGI3tDGPlcWZmBbXS0m"
MANIFEST = Path("data/ntu60_frame50/_manifest.json")


def enumerate_root(proxy: str | None, max_tries: int = 8) -> list[dict]:
    import gdown

    last_err: Exception | None = None
    for i in range(1, max_tries + 1):
        try:
            print(f"[enum] 第 {i}/{max_tries} 次枚举根目录…")
            items = gdown.download_folder(url=ROOT_URL, proxy=proxy, skip_download=True, quiet=True)
            out = []
            for it in items or []:
                out.append(
                    {
                        "id": getattr(it, "id", None),
                        "path": str(getattr(it, "path", "")).replace("\\", "/"),
                        "size": int(getattr(it, "size", 0) or 0),
                        "url": getattr(it, "url", None),
                    }
                )
            print(f"[enum] 成功：{len(out)} 个条目")
            return out
        except Exception as e:  # noqa: BLE001 — 枚举失败需整体重试
            last_err = e
            print(f"[enum] 失败：{type(e).__name__}: {e}；{5 * i}s 后重试")
            time.sleep(5 * i)
    raise RuntimeError(f"枚举在 {max_tries} 次后仍失败：{last_err}")


def fetch_one(gdown, url: str, dest: Path, proxy: str | None, tries: int) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for i in range(1, tries + 1):
        try:
            got = gdown.download(
                url=url,
                output=str(dest),
                proxy=proxy,
                quiet=True,
                resume=True,  # 断点续传：失败重试从已下载字节继续
            )
            if got:
                print(f"[dl] OK {dest}")
                return True
            print(f"[dl] 空返回（第 {i} 次）：{dest.name}")
        except Exception as e:  # noqa: BLE001 — 单文件失败重试
            print(f"[dl] 第 {i}/{tries} 次失败 {dest.name}: {type(e).__name__}: {e}")
        time.sleep(4 * i)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--proxy", default=None, help="HTTP 代理，如 http://127.0.0.1:17890")
    ap.add_argument("--attempts", type=int, default=6, help="每文件最大尝试次数")
    ap.add_argument("--include", action="append", default=[], help="额外包含的路径前缀（可多次）")
    ap.add_argument("--dest", type=Path, default=Path("data/ntu60_frame50"), help="数据根目录")
    args = ap.parse_args()

    try:
        import gdown
    except ImportError:
        print("缺少 gdown：pip install gdown", file=sys.stderr)
        return 2

    prefixes = ["ntu60_frame50"] + [p.strip("/") for p in args.include]

    # 阶段 1：枚举（有清单缓存则直接用）
    if MANIFEST.exists():
        items = json.loads(MANIFEST.read_text(encoding="utf-8"))
        print(f"[enum] 使用已有清单 {MANIFEST}（{len(items)} 条）；删除该文件可强制重新枚举")
    else:
        items = enumerate_root(args.proxy)
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[enum] 清单已存 {MANIFEST}")

    targets = [it for it in items if it["id"] and any(p in it["path"] for p in prefixes)]
    print(f"[plan] 命中前缀 {prefixes}：{len(targets)} 个文件")

    ok = fail = skip = 0
    for it in targets:
        rel = it["path"]
        # 官方根目录含顶层 action_dataset/ 前缀；落盘时剥掉，使 dest 直接就是数据根
        rel_stripped = rel.split("/", 1)[1] if "/" in rel else rel
        dest = args.dest / rel_stripped
        if dest.with_suffix(dest.suffix + ".ok").exists():
            skip += 1
            continue
        url = it["url"] or f"https://drive.google.com/uc?id={it['id']}"
        if fetch_one(gdown, url, dest, args.proxy, args.attempts):
            dest.with_suffix(dest.suffix + ".ok").write_text("done", encoding="ascii")
            ok += 1
        else:
            fail += 1
            print(f"[FAIL] {rel}")

    print(f"\n[summary] 新下载 {ok} / 跳过 {skip} / 失败 {fail}")
    print("[next] 全部成功后运行校验：python scripts/fetch_ntu_data.py --verify --dest data/ntu60_frame50")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
