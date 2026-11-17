#!/bin/bash
# UCF101 无标签物理层协议回归 — 分段链（6 段 × 25ep，每段一个进程、跑完即退）。
# 遵守用户约束：不挂长进程；段间进程退出；checkpoint 每 5ep 落盘，段失败可续。
# 段完成 → 三臂（单进程，跑完即退）。任一阶段重试耗尽即中止。
set -u
REPO="D:/Desktop/PR论文/psd-framework"
cd "$REPO" || exit 1
PY="$REPO/.venv/Scripts/python.exe"
export PYTHONUNBUFFERED=1
CFG="configs/labelfree_ucf101_pretext.yaml"
DIR="runs/p1_sprint/labelfree_ucf101"
LOG="$DIR/pretext.log"
BOUNDS=(25 50 75 100 125 150)

latest_epoch() {
  local best=0 f b n
  for f in "$DIR"/pretext/epoch*_model.pt; do
    [ -e "$f" ] || continue
    b=$(basename "$f"); n=${b#epoch}; n=${n%%_*}
    [ "$n" -gt "$best" ] 2>/dev/null && best=$n
  done
  echo "$best"
}

for target in "${BOUNDS[@]}"; do
  start=$(latest_epoch)
  if [ "$start" -ge "$target" ]; then
    echo "[$(date +%H:%M:%S)] slice ->$target skipped (already at $start)" >> "$LOG"
    continue
  fi
  echo "[$(date +%H:%M:%S)] slice $start->$target START" >> "$LOG"
  if [ "$start" -gt 0 ]; then
    "$PY" -u scripts/run_ntu_phaseb.py --config "$CFG" \
      --weights "$DIR/pretext/epoch${start}_model.pt" --start_epoch "$start" \
      --num_epoch "$target" >> "$LOG" 2>&1
  else
    "$PY" -u scripts/run_ntu_phaseb.py --config "$CFG" --num_epoch "$target" >> "$LOG" 2>&1
  fi
  code=$?
  now=$(latest_epoch)
  echo "[$(date +%H:%M:%S)] slice ->$target END exit=$code latest=$now" >> "$LOG"
  if [ "$now" -lt "$target" ]; then
    echo "[$(date +%H:%M:%S)] slice FAILED (latest=$now < $target), retrying once after 60s" >> "$LOG"
    sleep 60
    start=$now
    if [ "$start" -gt 0 ]; then
      "$PY" -u scripts/run_ntu_phaseb.py --config "$CFG" \
        --weights "$DIR/pretext/epoch${start}_model.pt" --start_epoch "$start" \
        --num_epoch "$target" >> "$LOG" 2>&1
    else
      "$PY" -u scripts/run_ntu_phaseb.py --config "$CFG" --num_epoch "$target" >> "$LOG" 2>&1
    fi
    now=$(latest_epoch)
    if [ "$now" -lt "$target" ]; then
      echo "[$(date +%H:%M:%S)] slice GAVE_UP (latest=$now < $target)" >> "$LOG"
      exit 1
    fi
  fi
done

echo "[$(date +%H:%M:%S)] PRETEXT_150_COMPLETE" >> "$LOG"
echo "[$(date +%H:%M:%S)] arms START" >> "$LOG"
"$PY" -u scripts/p1_sprint/labelfree_tier_replication.py --tier ucf101 >> "$DIR/replicate.log" 2>&1
code=$?
echo "[$(date +%H:%M:%S)] arms END exit=$code" >> "$LOG"
[ $code -eq 0 ] && echo "UCF101_CHAIN_ALL_DONE" || echo "UCF101_CHAIN_ARMS_FAILED"
