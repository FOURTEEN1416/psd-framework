#!/usr/bin/env python
"""W9 Phase A — NTU60 官方预处理数据获取与校验（纯 CPU）。

渠道优先级（依据 dev-docs/handovers/W9-ntu-repro.md §2 与官方仓 README/issues #2）：
  A. Google Drive 官方预处理包（作者 Levigty 维护，根目录含 action_dataset/ 与 released_model/）
     - 本机若无法直连 drive.google.com（2026-08-24 W9 实测不可达），用 --proxy 指定可用本地代理
     - gdown 为可选依赖，不进 requirements.txt：pip install gdown 后运行
  B. 百度网盘官方镜像（作者 2022-02-11 在 issue #2 提供，提取码 0211）
     - 需用户账号交互；本脚本打印转存指引，下载完成后用 --verify 校验

用法：
  python scripts/fetch_ntu_data.py --channel baidu              # 打印手动转存指引
  python scripts/fetch_ntu_data.py --channel gdrive [--proxy http://127.0.0.1:PORT] \
         [--folder-id <action_dataset子目录ID>]                  # 自动下载
  python scripts/fetch_ntu_data.py --verify --dest data/ntu60_frame50

数据契约（来源 external/AimCLR config/ntu60/*.yaml + feeder/ntu_feeder.py，只读引用）：
  {xsub,xview}/{train,val}_position.npy : (N,3,50,25,2) float32（建议 mmap 可读）
  {xsub,xview}/{train,val}_label.pkl    : pickle 二元组 (sample_names, labels)，labels ∈ [0,60)
说明：motion/bone 流由官方 processor 从 position 现算（issues #2 作者答复），无需单独文件。
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

BAIDU_URL = "https://pan.baidu.com/s/1NRK1ksRHgng_NkOO1ZYTcQ"
BAIDU_CODE = "0211"
GDRIVE_ROOT_ID = "1VnD3CLcD7bT5fMGI3tDGPlcWZmBbXS0m"  # 官方 README「action_dataset + released_model」根目录


def print_baidu_guide(dest: Path) -> None:
    print("【渠道 B：百度网盘手动转存指引】")
    print(f"  1. 打开链接: {BAIDU_URL}  提取码: {BAIDU_CODE}")
    print("  2. 只需转存其中的 ntu60_frame50 子树（released_model / PKU 相关可忽略）")
    print("  3. 用网盘客户端下载到本机后，将目录结构整理为：")
    print(f"     {dest}/{{xsub,xview}}/{{train,val}}_position.npy + 同名 label.pkl")
    print("  4. 运行校验：")
    print(f"     python scripts/fetch_ntu_data.py --verify --dest {dest.as_posix()}")


def fetch_gdrive(dest: Path, proxy: str | None, folder_id: str | None) -> int:
    try:
        import gdown  # 可选依赖，懒加载
    except ImportError:
        print("缺少 gdown：请先 `.\\.venv\\Scripts\\python.exe -m pip install gdown`", file=sys.stderr)
        return 2
    fid = folder_id or GDRIVE_ROOT_ID
    url = f"https://drive.google.com/drive/folders/{fid}"
    print(f"[gdown] 下载 {url} -> {dest}" + (f" (proxy={proxy})" if proxy else ""))
    # gdown >= 4/5 移除了 CLI 风格的 main(args)；改用 download_folder 编程接口
    # （v6 签名：url/id/output/quiet/proxy/speed/use_cookies/verify/user_agent/skip_download/resume）
    out = gdown.download_folder(url=url, output=str(dest), proxy=proxy, quiet=False)
    rc = 0 if out else 1
    if rc == 0:
        print("[gdown] 完成。注意：根目录含 released_model/，如只需数据可删除该子目录。")
        print(f"[next] 校验：python scripts/fetch_ntu_data.py --verify --dest {dest.as_posix()}")
    return rc


def verify(dest: Path) -> int:
    ok = True
    total = 0
    print(f"校验目标: {dest.resolve()}")
    for bench in ("xsub", "xview"):
        for part in ("train", "val"):
            pos = dest / bench / f"{part}_position.npy"
            lab = dest / bench / f"{part}_label.pkl"
            if not pos.exists() or not lab.exists():
                print(f"  [缺失] {pos.name} / {lab.name} @ {bench}/{part}")
                ok = False
                continue
            import numpy as np

            arr = np.load(pos, mmap_mode="r")
            with open(lab, "rb") as f:
                names, labels = pickle.load(f)
            n = arr.shape[0]
            total += n
            shape_ok = arr.ndim == 5 and tuple(arr.shape[1:]) == (3, 50, 25, 2)
            label_ok = len(labels) == n and all(0 <= int(x) < 60 for x in labels[: min(n, 100)])
            status = "OK" if (shape_ok and label_ok) else "FAIL"
            if not (shape_ok and label_ok):
                ok = False
            print(
                f"  [{status}] {bench}/{part}: N={n}, shape={arr.shape}, dtype={arr.dtype}, "
                f"label数={len(labels)} (期望形状 (N,3,50,25,2) float32)"
            )
    print(f"合计样本数: {total}（论文口径 56,578 序列；官方全集 56,880 减缺失骨架样本，允许小幅出入）")
    print("结果: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="NTU60 AimCLR 预处理数据获取/校验（W9 Phase A）")
    ap.add_argument("--channel", choices=["gdrive", "baidu"], help="获取渠道")
    ap.add_argument("--proxy", default=None, help="HTTP 代理，如 http://127.0.0.1:7890")
    ap.add_argument("--folder-id", default=None, help="Google Drive 子目录 ID（默认官方根目录）")
    ap.add_argument("--dest", type=Path, default=Path("data/ntu60_frame50"), help="数据落盘目录（gitignore 内）")
    ap.add_argument("--verify", action="store_true", help="仅校验已有数据")
    args = ap.parse_args()

    if args.verify:
        return verify(args.dest)
    if args.channel == "baidu":
        print_baidu_guide(args.dest)
        return 0
    if args.channel == "gdrive":
        return fetch_gdrive(args.dest, args.proxy, args.folder_id)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
