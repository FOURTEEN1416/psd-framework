# -*- coding: utf-8 -*-
"""P2'/P7 看门狗 — 计划任务每 10 分钟冷启动一轮, 保障两线断点续跑。

设计(2026-09-05, ADR 0009 并行线保障):
  - P2' YOLO11x-pose 续训(x_pose_dog-3, 低内存 batch=2/workers=2):
      活性 = 目录 max mtime < 40min(>1 epoch 周期); 死/卡 →
      last.pt 存在 → resume=True 续跑(epoch 计数/lr 调度保留);
      last.pt 不存在 → 重跑 p18_resume_lowmem.py(X2 权重重建 X3)。
  - P7 PanAf 下载(aria2 16 连接):
      .aria2 在且 zip mtime < 20min → ALIVE; 否则杀残留 aria2c 后 -c 续传;
      .aria2 消失且 zip 在 → 后台 tar 解压(磁盘 185G 充足, 不删 zip);
      解压完成写 EXTRACT_DONE 标记。DLC 推理需 GPU, 由主会话在
      YOLO DONE 后启动(watchdog 只报状态不抢 GPU)。
  - 单实例锁(pid 存活即退出); 恢复动作 DETACHED_PROCESS 拉起, 看门狗退出不影响。
  - 教训内嵌: 计划任务禁重复注册(注册前 /query); NextRun 空即守护死(注册后必验);
    后台长任务禁 tail 管道(一律 -u + 落盘日志)。

状态: runs/watchdog_p2p7_status.json(每轮覆盖) + runs/watchdog_p2p7.log(追加)
手动执行: .venv/Scripts/python.exe scripts/watchdog_p2p7.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = REPO / ".venv" / "Scripts" / "python.exe"
RUNS = REPO / "runs"
K9 = Path(r"D:\Desktop\k9-training-system\runs\pose\runs\p18_yolox_finetune")
X3 = K9 / "x_pose_dog-3"
X2_LAST = K9 / "x_pose_dog-2" / "weights" / "last.pt"
PANAF_DIR = RUNS / "p7_asbar"
ZIP = PANAF_DIR / "panaf_dataset.zip"
ARIA2_CTRL = PANAF_DIR / "panaf_dataset.zip.aria2"
EXTRACT_DIR = PANAF_DIR / "panaf_extract"  # 与 run_p21_panaf_pipeline.py 的 EXTRACT_DIR 一致
EXTRACT_DONE = PANAF_DIR / "EXTRACT_DONE"
URL = "https://data.bris.ac.uk/datasets/tar/1h73erszj3ckn2qjwm4sqmr2wt.zip"
ARIA2C = r"C:\Users\FOUR\AppData\Local\Microsoft\WinGet\Links\aria2c.exe"
LOCK = RUNS / "watchdog_p2p7.lock"
STATUS = RUNS / "watchdog_p2p7_status.json"
LOG = RUNS / "watchdog_p2p7.log"

YOLO_STALE_MIN = 40      # > 2 epoch 周期(~30min/ep)
DL_STALE_MIN = 20
PANAF_N_TOTAL = 89       # x_pose_dog-3 目标 epoch 数


def log(msg: str):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def detached(cmd: list, logpath: Path, cwd: Path | None = None):
    """独立于本进程拉起长任务; stdout/stderr 追加到日志(-u 落盘, 禁管道)。"""
    logf = open(logpath, "ab")
    subprocess.Popen(
        cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=cwd,
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        close_fds=True,
    )


def dir_fresh(d: Path, minutes: int) -> tuple[bool, float]:
    """目录内所有文件 max mtime 距今 < minutes → 活。返回 (是否活, 最旧分钟数)。"""
    if not d.exists():
        return False, -1.0
    newest = 0.0
    for p in d.rglob("*"):
        try:
            if p.is_file():
                newest = max(newest, p.stat().st_mtime)
        except OSError:
            continue
    if newest == 0.0:
        return False, -1.0
    age_min = (time.time() - newest) / 60
    return age_min < minutes, age_min


def pid_alive(pid: int) -> bool:
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True,
                             text=True, timeout=30).stdout
        return str(pid) in out
    except Exception:
        return False


def yolo_epoch_done() -> bool:
    csv = X3 / "results.csv"
    if not csv.exists():
        return False
    n = sum(1 for _ in open(csv, encoding="utf-8", errors="ignore"))
    return n >= PANAF_N_TOTAL + 1  # header + 89


def check_yolo(actions: list):
    if yolo_epoch_done():
        return "DONE"
    alive, age = dir_fresh(X3, YOLO_STALE_MIN)
    if alive:
        return f"ALIVE (newest file {age:.0f}min ago)"
    # 死/卡 → 恢复
    if (X3 / "weights" / "last.pt").exists():
        log(f"[yolo] stale({age:.0f}min) → resume from x_pose_dog-3/last.pt")
        detached([str(PY), "-u", "-c",
                  f"from ultralytics import YOLO; YOLO(r'{X3 / 'weights' / 'last.pt'}').train(resume=True)"],
                  RUNS / "p18_watchdog_train.log", cwd=str(REPO))
        actions.append("yolo: resumed from x_pose_dog-3/last.pt")
        return "RESTARTED (resume)"
    if X2_LAST.exists():
        log(f"[yolo] stale({age:.0f}min) & no X3 ckpt → rerun p18_resume_lowmem.py")
        detached([str(PY), "-u", str(REPO / "scripts" / "p18_resume_lowmem.py")],
                 RUNS / "p18_watchdog_train.log", cwd=str(REPO))
        actions.append("yolo: re-seeded x_pose_dog-3 from x_pose_dog-2 weights")
        return "RESTARTED (fresh low-mem)"
    log("[yolo] FATAL: no checkpoint anywhere (X3/X2 last.pt missing) — manual fix required")
    actions.append("yolo: NO CHECKPOINT, manual fix required")
    return "FATAL_NO_CKPT"


def kill_aria2():
    subprocess.run(["taskkill", "/F", "/IM", "aria2c.exe"], capture_output=True, timeout=30)


def tar_running() -> bool:
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq tar.exe"], capture_output=True,
                             text=True, timeout=30).stdout
        return "tar.exe" in out
    except Exception:
        return False


def check_panaf(actions: list):
    if not ARIA2_CTRL.exists():
        # 下载已结束: 校验后进入解压分支
        if ZIP.exists() and ZIP.stat().st_size > 40 * 1024**3:
            if EXTRACT_DONE.exists():
                return "EXTRACTED"
            extracting = EXTRACT_DIR.exists() and any(EXTRACT_DIR.iterdir())
            if extracting and tar_running():
                return "EXTRACTING"
            if extracting:
                alive, age = dir_fresh(EXTRACT_DIR, 30)
                if alive:
                    return f"EXTRACTING (files updated {age:.0f}min ago, tar still writing)"
                # tar 已退出且目录静默 → 视为解压完成
                n_files = sum(1 for p in EXTRACT_DIR.rglob("*") if p.is_file())
                if n_files > 100:
                    EXTRACT_DONE.write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")
                    log(f"[panaf] extraction complete ({n_files} files) → EXTRACT_DONE")
                    actions.append("panaf: extraction verified complete")
                    return "EXTRACTED"
                log(f"[panaf] tar exited but only {n_files} files → redo extraction")
                actions.append("panaf: extraction redone (incomplete)")
            else:
                log("[panaf] download complete → starting tar extraction")
                actions.append("panaf: extraction started")
            EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
            detached(["tar", "-xf", str(ZIP), "-C", str(EXTRACT_DIR)],
                     PANAF_DIR / "panaf_extract.log", cwd=str(PANAF_DIR))
            return "EXTRACTING (started)"
        return "ERROR (no ctrl file, zip missing/small)"
    # 下载中: 用 zip 自身 mtime 判活(目录内日志文件会污染 max-mtime 判据)
    age = (time.time() - ZIP.stat().st_mtime) / 60
    if age < DL_STALE_MIN:
        return f"DOWNLOADING (zip updated {age:.0f}min ago)"
    log(f"[panaf] download stale({age:.0f}min) → kill stale aria2c + resume with -c")
    kill_aria2()
    time.sleep(2)
    detached([ARIA2C, "-x16", "-s16", "-c", "-d", str(PANAF_DIR), "-o", "panaf_dataset.zip",
              "--check-certificate=false", URL], PANAF_DIR / "aria2_watchdog.log", cwd=str(PANAF_DIR))
    actions.append("panaf: aria2 resumed (-c)")
    return "RESTARTED (aria2 -c)"


def main():
    # 单实例锁
    if LOCK.exists():
        try:
            old = int(LOCK.read_text().strip())
            if pid_alive(old):
                print(f"[watchdog] instance {old} still running, exit")
                return
        except Exception:
            pass
    LOCK.write_text(str(os.getpid()))
    try:
        actions: list = []
        yolo_state = check_yolo(actions)
        panaf_state = check_panaf(actions)
        for a in actions:
            log(a)
        status = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "yolo": {"state": yolo_state, "epoch_target": PANAF_N_TOTAL,
                     "last_pt": str(X3 / "weights" / "last.pt")},
            "panaf": {"state": panaf_state,
                      "extract_done": EXTRACT_DONE.exists(),
                      "extract_dir": str(EXTRACT_DIR)},
            "actions": actions,
            "next_action_hint": (
                "DLC inference (run_p21) after BOTH: yolo DONE + panaf EXTRACTED"
            ),
        }
        STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(status, ensure_ascii=False))
    finally:
        try:
            LOCK.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
