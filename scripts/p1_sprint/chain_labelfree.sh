#!/bin/bash
# P1 协议回归链 v2 — 断点续训版。
# 背景: v1 在 NTU120 epoch25 因系统 commit 内存 99% 被 OOM 杀死。P01 每 5ep 落盘
# checkpoint，处理器支持 --weights + --start_epoch，故本版每阶段用「最近 ckpt 续训 +
# 失败自动重试」的循环，最多 10 次尝试；两次尝试间隔 90s 让系统回收内存。
# 串行: NTU120(150ep)→三臂 → UCF101(150ep)→三臂。任一阶段重试耗尽即中止（不静默断链）。
set -u
REPO="D:/Desktop/PR论文/psd-framework"
cd "$REPO" || exit 1
PY="$REPO/.venv/Scripts/python.exe"
export PYTHONUNBUFFERED=1

latest_epoch() {  # latest_epoch <pretext_dir> → 最大 epoch 号（无则 0）
  local best=0 f b n
  for f in "$1"/epoch*_model.pt; do
    [ -e "$f" ] || continue
    b=$(basename "$f"); n=${b#epoch}; n=${n%%_*}
    [ "$n" -gt "$best" ] 2>/dev/null && best=$n
  done
  echo "$best"
}

train_with_resume() {  # train_with_resume <tier> <cfg> <max_attempts>
  local tier=$1 cfg=$2 max=$3 attempt start ckpt
  local dir="runs/p1_sprint/labelfree_$tier/pretext"
  local log="runs/p1_sprint/labelfree_$tier/pretext.log"
  for attempt in $(seq 1 "$max"); do
    start=$(latest_epoch "$dir")
    if [ "$start" -ge 150 ]; then
      echo "[$(date +%H:%M:%S)] $tier pretext already complete (epoch $start)" >> "$log"
      return 0
    fi
    echo "[$(date +%H:%M:%S)] $tier pretext attempt $attempt start_epoch=$start" >> "$log"
    if [ "$start" -gt 0 ]; then
      "$PY" -u scripts/run_ntu_phaseb.py --config "$cfg" \
        --weights "$dir/epoch${start}_model.pt" --start_epoch "$start" >> "$log" 2>&1
    else
      "$PY" -u scripts/run_ntu_phaseb.py --config "$cfg" >> "$log" 2>&1
    fi
    local code=$?
    local now; now=$(latest_epoch "$dir")
    echo "[$(date +%H:%M:%S)] $tier pretext attempt $attempt END exit=$code latest_epoch=$now" >> "$log"
    if [ "$now" -ge 150 ]; then
      echo "[$(date +%H:%M:%S)] $tier pretext COMPLETE" >> "$log"
      return 0
    fi
    sleep 90
  done
  echo "[$(date +%H:%M:%S)] $tier pretext GAVE_UP after $max attempts" >> "$log"
  return 1
}

run_arms() {  # run_arms <tier>
  local tier=$1
  local log="runs/p1_sprint/labelfree_$tier/replicate.log"
  echo "[$(date +%H:%M:%S)] $tier arms START" >> "$log"
  "$PY" -u scripts/p1_sprint/labelfree_tier_replication.py --tier "$tier" >> "$log" 2>&1
  local code=$?
  echo "[$(date +%H:%M:%S)] $tier arms END exit=$code" >> "$log"
  return $code
}

train_with_resume ntu120 configs/labelfree_ntu120_pretext.yaml 10 || { echo "CHAIN_ABORTED ntu120"; exit 1; }
run_arms ntu120 || { echo "CHAIN_ABORTED ntu120_arms"; exit 1; }
train_with_resume ucf101 configs/labelfree_ucf101_pretext.yaml 10 || { echo "CHAIN_ABORTED ucf101"; exit 1; }
run_arms ucf101 || { echo "CHAIN_ABORTED ucf101_arms"; exit 1; }

echo "[$(date +%H:%M:%S)] CHAIN_ALL_DONE" >> runs/p1_sprint/labelfree_ntu120/pretext.log
echo "CHAIN_ALL_DONE"
