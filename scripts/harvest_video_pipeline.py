"""C1/W25 公开视频主动抓取管线（Bilibili + YouTube → 格式 A 片段池）。

收敛契约: dev-docs/handovers/DATA-CAMPAIGN-plan.md §0 格式 A
  - 目录: runs/data_campaign/<channel>/fragments/*.mp4, 单段 <=30s
  - manifest 每行: {fragment_id, source_channel, origin_url_or_path,
                    capture_context, species_note, license_note, collected_at}
领地: scripts/harvest_* + runs/data_campaign/video/ + configs/harvest_*
纪律: 全程 CPU/网络; 禁触 GPU 队列(relay Q1-Q3c); 标签=规则引擎自产(待提点),
      禁止外部标签直用(df_action.xlsx 教训)。

用法:
  python scripts/harvest_video_pipeline.py --config configs/harvest_video_w25.yaml --stage selftest
  python scripts/harvest_video_pipeline.py --config ... --stage search
  python scripts/harvest_video_pipeline.py --config ... --stage download
  python scripts/harvest_video_pipeline.py --config ... --stage split
  python scripts/harvest_video_pipeline.py --config ... --stage filter
  python scripts/harvest_video_pipeline.py --config ... --stage manifest
  python scripts/harvest_video_pipeline.py --config ... --stage all

各阶段状态落盘于 <out_root>/_runtime/*.jsonl, 重跑自动跳过已完成条目。
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # cwd 不在仓根时也能 import psd(本脚本自身不依赖 psd 包)

# ---------------------------------------------------------------- 工具


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def stable_id(*parts: str, n: int = 12) -> str:
    h = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:n]
    return f"w25v-{h}"


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def log(msg: str) -> None:
    print(f"[harvest {now_iso()}] {msg}", flush=True)


class Pipeline:
    def __init__(self, cfg_path: str):
        self.cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
        self.out_root = REPO / self.cfg["out_root"]
        self.rt = self.out_root / "_runtime"
        self.rt.mkdir(parents=True, exist_ok=True)
        (self.out_root / "fragments").mkdir(parents=True, exist_ok=True)
        self.f_cand = self.rt / "candidates.jsonl"
        self.f_dl = self.rt / "downloaded.jsonl"
        self.f_frag = self.rt / "fragments_raw.jsonl"   # 切分产物(含被拒者)
        self.f_stats = self.rt / "filter_stats.jsonl"   # YOLO 粗筛结果
        self.f_man = self.out_root / "manifest.jsonl"
        self.ua = " ".join(self.cfg["http_user_agent"].split())

    # ------------------------------------------------ 搜索阶段

    def _ydl_opts(self, platform: str, playlistend: int | None = None) -> dict:
        import yt_dlp

        p = self.cfg["platforms"][platform]
        opts: dict = {
            "quiet": True,
            "no_warnings": True,
            "socket_timeout": 20,
            "extract_flat": True,   # 搜索只拿列表不逐条解析(单条坏视频不再拖死整查询)
            "http_headers": {"User-Agent": self.ua},
        }
        if playlistend:
            opts["playlistend"] = playlistend
        if p.get("proxy"):
            opts["proxy"] = p["proxy"]
        if platform == "bilibili":
            ck = self._bili_cookie_file()
            if ck:
                opts["cookiefile"] = str(ck)
        return opts

    def _bili_cookie_file(self) -> Path | None:
        """访问 B 站首页获取游客 cookie(buvid3 等), Netscape 格式落盘。412 对策。"""
        import http.cookiejar

        path = self.rt / "cookies_bili_guest.txt"
        if path.exists() and path.stat().st_size > 50:
            return path
        cj = http.cookiejar.MozillaCookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
        opener.addheaders = [("User-Agent", self.ua), ("Referer", "https://www.bilibili.com/")]
        try:
            resp = opener.open("https://www.bilibili.com/", timeout=15)
            resp.read(1024)
        except Exception as e:  # noqa: BLE001 —— 网络失败不致命, 记录后继续
            log(f"[bili-cookie] 首页访问失败: {e}")
            return None
        cj.save(str(path), ignore_discard=True, ignore_expires=True)
        log(f"[bili-cookie] 游客 cookie 已获取: {[c.name for c in cj]}")
        return path

    def stage_search(self) -> None:
        import yt_dlp

        seen_ids: set[str] = set()
        for c in load_jsonl(self.f_cand):
            seen_ids.add(c["video_key"])
        n_new = 0
        import time
        for qi, q in enumerate(self.cfg["search"]["queries"]):
            if qi:
                time.sleep(2.5)  # 平台礼貌间隔, 降低 B 站 412 风控概率
            for platform in self.cfg["search"]["platform_pref"][q["lang"]]:
                if not self.cfg["platforms"][platform].get("enabled"):
                    continue
                prefix = "bilisearch" if platform == "bilibili" else "ytsearch"
                query = f"{prefix}{self.cfg['search']['results_per_query']}:{q['kw']}"
                try:
                    with yt_dlp.YoutubeDL(self._ydl_opts(platform)) as ydl:
                        info = ydl.extract_info(query, download=False)
                except yt_dlp.utils.DownloadError as e:
                    msg = str(e)
                    if platform == "bilibili" and "412" in msg:
                        # cookie 失效 → 刷新重试一次
                        (self.rt / "cookies_bili_guest.txt").unlink(missing_ok=True)
                        try:
                            with yt_dlp.YoutubeDL(self._ydl_opts(platform)) as ydl2:
                                info = ydl2.extract_info(query, download=False)
                            log(f"[search] bili 412 后刷新 cookie 重试成功: {q['kw']}")
                        except Exception as e2:  # noqa: BLE001
                            log(f"[search] FAIL {platform}:{q['kw']} -> {str(e2)[:120]}")
                            continue
                    else:
                        log(f"[search] FAIL {platform}:{q['kw']} -> {msg[:120]}")
                        continue
                entries = list(info.get("entries") or [])
                kept = 0
                for e in entries:
                    vid = e.get("id") or ""
                    url = e.get("url") or e.get("webpage_url") or ""
                    if not vid or not url:
                        continue
                    key = f"{platform}:{vid}"
                    if key in seen_ids:
                        continue
                    seen_ids.add(key)
                    title = e.get("title") or e.get("raw_title") or ""
                    dur = e.get("duration")
                    append_jsonl(self.f_cand, {
                        "video_key": key, "platform": platform,
                        "video_id": str(vid), "url": url,
                        "title": str(title)[:200], "duration_s": dur,
                        "query_kw": q["kw"], "lang": q["lang"],
                        "found_at": now_iso(),
                    })
                    kept += 1
                    n_new += 1
                log(f"[search] {platform}:'{q['kw']}' 候选 +{kept}/{len(entries)}")
        total = len(load_jsonl(self.f_cand))
        log(f"[search] 完成, 候选总数 {total} (新增 {n_new})")

    # ------------------------------------------------ 下载阶段

    def _pick_candidates(self) -> list[dict]:
        """按查询轮转选取候选, 尊重每查询下载上限与时长过滤; 跳过已下载。"""
        sf = self.cfg["source_filters"]
        done = {r["video_key"] for r in load_jsonl(self.f_dl)}
        per_q: dict[str, list[dict]] = {}
        for r in load_jsonl(self.f_cand):
            dur = r.get("duration_s")
            if dur is not None and not (sf["min_duration_s"] <= dur <= sf["max_duration_s"]):
                continue
            per_q.setdefault(r["query_kw"], []).append(r)
        picked: list[dict] = []
        used_per_q: dict[str, int] = {}
        order = [q["kw"] for q in self.cfg["search"]["queries"]]
        # 轮转保证关键词覆盖均匀
        lists = [(k, per_q[k]) for k in order if k in per_q]
        idx = 0
        while any(lst for _, lst in lists):
            k, lst = lists[idx % len(lists)]
            cnt = used_per_q.get(k, 0)
            cap = self.cfg["search"]["download_per_query"]
            if cnt >= cap or not lst:
                lists[idx % len(lists)] = (k, [])
                idx += 1
                if all(not l for _, l in lists):
                    break
                continue
            cand = lst.pop(0)
            if cand["video_key"] in done:
                continue
            picked.append(cand)
            used_per_q[k] = cnt + 1
            idx += 1
        return picked

    def stage_download(self, limit: int | None = None) -> None:
        import yt_dlp

        target = self.cfg["target_fragments"] + self.cfg["safety_margin"]
        accepted = sum(1 for r in load_jsonl(self.f_stats) if r.get("accepted"))
        if accepted >= target:
            log(f"[download] 已有 {accepted} 片段过筛 >= 目标 {target}, 无需继续下载")
            return
        picked = self._pick_candidates()
        if limit:
            picked = picked[:limit]
        log(f"[download] 本轮拟下载 {len(picked)} 个源视频")
        dl_dir = self.rt / "cache"

        def fmt_spec() -> str:
            return "bv*[height<=480]/b[height<=480]/bv*/b/w"

        for i, cand in enumerate(picked, 1):
            cur_acc = sum(1 for r in load_jsonl(self.f_stats) if r.get("accepted"))
            if cur_acc >= target:
                log(f"[download] 达到目标 {cur_acc}>={target}, 提前停机")
                break
            if any(r["video_key"] == cand["video_key"] for r in load_jsonl(self.f_dl)):
                continue
            # ---- 下载前轻量预探: 时长/分辨率越界者直接跳过(省带宽) ----
            try:
                probe_opts = self._ydl_opts(cand["platform"])
                probe_opts.update({"skip_download": True})
                with yt_dlp.YoutubeDL(probe_opts) as ydl:
                    meta = ydl.extract_info(cand["url"], download=False)
                dur0 = meta.get("duration")
                h0 = int(meta.get("height") or 0)
                sf = self.cfg["source_filters"]
                if dur0 and not (sf["min_duration_s"] <= dur0 <= sf["max_duration_s"]):
                    append_jsonl(self.f_dl, {
                        "video_key": cand["video_key"], "platform": cand["platform"],
                        "video_id": cand["video_id"], "url": cand["url"],
                        "title": str(meta.get("title") or "")[:200],
                        "query_kw": cand["query_kw"], "lang": cand["lang"],
                        "error": f"pre-probe skip: duration {dur0}s 越界",
                        "downloaded_at": now_iso(),
                    })
                    log(f"[download] 预探跳过 {cand['video_key']} (dur={dur0:.0f}s)")
                    continue
                if h0 and h0 < sf["min_height"]:
                    append_jsonl(self.f_dl, {
                        "video_key": cand["video_key"], "platform": cand["platform"],
                        "video_id": cand["video_id"], "url": cand["url"],
                        "title": str(meta.get("title") or "")[:200],
                        "query_kw": cand["query_kw"], "lang": cand["lang"],
                        "error": f"pre-probe skip: height {h0} < 下限",
                        "downloaded_at": now_iso(),
                    })
                    log(f"[download] 预探跳过 {cand['video_key']} (h={h0})")
                    continue
            except Exception as e:  # noqa: BLE001 — 预探失败不拦截, 交由正式下载兜底
                log(f"[download] 预探失败(继续) {cand['video_key']}: {str(e)[:80]}")
            outtmpl = str(dl_dir / cand["platform"] / f"{cand['video_id']}.%(ext)s")
            opts = {
                "quiet": True, "no_warnings": True, "noprogress": True,
                "socket_timeout": 30,
                "format": fmt_spec(),
                "outtmpl": outtmpl,
                "concurrent_fragment_downloads": 4,
                "retries": 2,
                "http_headers": {"User-Agent": self.ua},
                "noplaylist": True,
            }
            p = self.cfg["platforms"][cand["platform"]]
            if p.get("proxy"):
                opts["proxy"] = p["proxy"]
            if cand["platform"] == "bilibili":
                ck = self._bili_cookie_file()
                if ck:
                    opts["cookiefile"] = str(ck)
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(cand["url"], download=True)
                fp = info.get("requested_downloads") or []
                fpath = fp[0]["filepath"] if fp else ydl.prepare_filename(info)
                if not Path(fpath).exists():
                    raise FileNotFoundError(f"下载后未找到文件: {fpath}")
                dur = info.get("duration") or cand.get("duration_s")
                if dur and not (self.cfg["source_filters"]["min_duration_s"]
                                <= dur <= self.cfg["source_filters"]["max_duration_s"]):
                    Path(fpath).unlink(missing_ok=True)
                    log(f"[download] 时长越界({dur:.0f}s)丢弃 {cand['video_key']}")
                    continue
                append_jsonl(self.f_dl, {
                    **{k: cand[k] for k in
                       ("video_key", "platform", "video_id", "url", "title",
                        "query_kw", "lang")},
                    "filepath": str(Path(fpath).resolve()),
                    "duration_s": dur,
                    "downloaded_at": now_iso(),
                })
                log(f"[download] ({i}/{len(picked)}) OK {cand['platform']}:"
                    f"{cand['video_id']} dur={dur and round(dur)}s 累计过筛={cur_acc}")
            except Exception as e:  # noqa: BLE001 —— 单源失败不阻塞全局
                append_jsonl(self.f_dl, {
                    "video_key": cand["video_key"], "platform": cand["platform"],
                    "video_id": cand["video_id"], "url": cand["url"],
                    "title": cand.get("title", ""), "query_kw": cand["query_kw"],
                    "lang": cand["lang"], "error": str(e)[:300],
                    "downloaded_at": now_iso(),
                })
                log(f"[download] FAIL {cand['video_key']}: {str(e)[:120]}")

    # ------------------------------------------------ 切分阶段

    def _ffprobe_duration(self, path: Path) -> float | None:
        try:
            out = subprocess.run(
                ["ffmpeg", "-hide_banner", "-i", str(path)],
                capture_output=True, text=True, timeout=60).stderr
            m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", out)
            if m:
                h, mi, s = m.groups()
                return int(h) * 3600 + int(mi) * 60 + float(s)
        except Exception:  # noqa: BLE001
            return None
        return None

    def _scene_times(self, path: Path) -> tuple[list[float], float]:
        """ffmpeg 场景切点检测(降采样解码省 CPU)。返回 (切点时间列表, 总时长)。"""
        sp = self.cfg["split"]
        cmd = [
            "ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
            "-vf", f"select='gt(scene,{sp['scene_threshold']})',showinfo",
            "-an", "-f", "null", "-",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        times: list[float] = []
        for line in proc.stderr.splitlines():
            m = re.search(r"pts_time:([\d.]+)", line)
            if m:
                times.append(float(m.group(1)))
        dur = self._ffprobe_duration(path) or (times[-1] if times else 0.0)
        # 去抖: 相邻切点最小间隔
        dedup: list[float] = []
        for t in times:
            if not dedup or t - dedup[-1] >= sp["min_scene_gap_s"]:
                dedup.append(t)
        return dedup, dur

    @staticmethod
    def budget_segments(bounds: list[float], max_len: float, min_len: float) -> list[tuple[float, float]]:
        """把场景边界预算成 <=max_len 的片段序列(纯函数, selftest 覆盖)。

        规则: 从 start 出发取不超过 max_len 的最远边界收段; 无边界可用时
        在 start+max_len 强制收段; 末尾不足 min_len 的残段丢弃。
        """
        pts = sorted({round(b, 3) for b in bounds if b > 0})
        segs: list[tuple[float, float]] = []
        start = 0.0
        end_total = pts[-1] if pts else 0.0
        while end_total - start > 1e-6:
            candidates = [p for p in pts if start + 1e-6 < p <= start + max_len + 1e-6]
            if candidates:
                close = candidates[-1]
            else:
                close = min(start + max_len, end_total)
            segs.append((start, close))
            start = close
            # 跳过与 start 重合的边界, 防死循环
            pts = [p for p in pts if p > start + 1e-6]
            if not pts:
                break
            end_total = max(end_total, pts[-1])
        return [(a, b) for a, b in segs if b - a >= min_len]

    def stage_split(self, limit: int | None = None) -> None:
        done_frags = {r["fragment_id"] for r in load_jsonl(self.f_frag)}
        enc = self.cfg["split"]["encode"]
        sources = [r for r in load_jsonl(self.f_dl) if r.get("filepath")]
        if limit:
            sources = [s for s in sources
                       if not any(not r.get("error")
                                  for r in load_jsonl(self.f_frag)
                                  if r["source_video_key"] == s["video_key"])][:limit]
        log(f"[split] 待处理源视频 {len(sources)} 个")
        n_new = 0
        for src in sources:
            already = [r for r in load_jsonl(self.f_frag) if r["source_video_key"] == src["video_key"]]
            if any(not r.get("error") for r in already):
                continue
            fpath = Path(src["filepath"])
            if not fpath.exists():
                continue
            try:
                times, dur = self._scene_times(fpath)
                if not dur:
                    raise RuntimeError("无法探测时长")
                bounds = [0.0] + times + [dur]
                segs = self.budget_segments(bounds, self.cfg["split"]["max_len_s"],
                                            self.cfg["split"]["min_len_s"])
                scale_h = enc["height"]
                for si, (a, b) in enumerate(segs):
                    fid = stable_id(src["video_key"], f"{a:.2f}-{b:.2f}")
                    if fid in done_frags:
                        continue
                    out = self.out_root / "fragments" / f"{fid}.mp4"
                    vf = f"scale=-2:'min({scale_h},ih)'"
                    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                           "-ss", f"{a:.3f}", "-i", str(fpath), "-t", f"{b - a:.3f}",
                           "-vf", vf, "-c:v", "libx264",
                           "-preset", enc["preset"], "-crf", str(enc["crf"]),
                           "-an", "-movflags", "+faststart", str(out)]
                    subprocess.run(cmd, check=True, timeout=300)
                    row = {
                        "fragment_id": fid,
                        "source_video_key": src["video_key"],
                        "origin_url_or_path": src["url"],
                        "title": src.get("title", ""),
                        "query_kw": src.get("query_kw", ""),
                        "lang": src.get("lang", ""),
                        "start_s": round(a, 2), "end_s": round(b, 2),
                        "duration_s": round(b - a, 2),
                        "fragment_path": str(out.resolve()),
                    }
                    append_jsonl(self.f_frag, row)
                    done_frags.add(fid)
                    n_new += 1
                log(f"[split] {src['video_key']} -> {len(segs)} 片段 (累计新片段 {n_new})")
            except Exception as e:  # noqa: BLE001
                append_jsonl(self.f_frag, {
                    "fragment_id": stable_id(src["video_key"], "ERR"),
                    "source_video_key": src["video_key"],
                    "origin_url_or_path": src["url"], "title": src.get("title", ""),
                    "query_kw": src.get("query_kw", ""), "lang": src.get("lang", ""),
                    "start_s": 0, "end_s": 0, "duration_s": 0,
                    "fragment_path": "", "error": str(e)[:300],
                })
                log(f"[split] FAIL {src['video_key']}: {str(e)[:150]}")

    # ------------------------------------------------ 粗筛阶段(YOLO COCO, CPU)

    def _ensure_yolo_weights(self) -> Path:
        from ultralytics import YOLO  # noqa: F401 — 触发其自带权重解析前先手动落盘

        name = self.cfg["filter"]["weights"]
        local = self.rt / name
        if local.exists():
            return local
        # 直连失败则走 YouTube 同款代理再试(GitHub release 可能被墙)
        urls = [
            f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{name}",
        ]
        proxy_candidates: list[str | None] = [None, "http://127.0.0.1:17890"]
        for attempt, proxy in enumerate(proxy_candidates):
            req = urllib.request.Request(urls[0], headers={"User-Agent": self.ua})
            opener = urllib.request.build_opener()
            if proxy:
                opener = urllib.request.build_opener(
                    urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
            try:
                log(f"[filter] 下载权重 {name} (attempt={attempt}, proxy={proxy})")
                with opener.open(req, timeout=60) as resp, local.open("wb") as fo:
                    while chunk := resp.read(1 << 20):
                        fo.write(chunk)
                return local
            except Exception as e:  # noqa: BLE001
                log(f"[filter] 权重下载失败(attempt={attempt}): {str(e)[:100]}")
        raise RuntimeError(f"COCO 权重获取失败: {name}")

    def stage_filter(self) -> None:
        import cv2  # ultralytics 依赖自带
        from ultralytics import YOLO

        fc = self.cfg["filter"]
        weights = self._ensure_yolo_weights()
        model = YOLO(str(weights))
        dog_cls = 16  # COCO 'dog'
        frag_rows = [r for r in load_jsonl(self.f_frag) if not r.get("error")]
        done = {r["fragment_id"] for r in load_jsonl(self.f_stats)}
        todo = [r for r in frag_rows if r["fragment_id"] not in done]
        log(f"[filter] 待粗筛片段 {len(todo)} 个 (device={fc['device']})")
        for r in todo:
            path = Path(r["fragment_path"])
            if not path.exists():
                continue
            cap = cv2.VideoCapture(str(path))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ks = fc["frames_per_fragment"]
            idxs = sorted({int(n_frames * (i + 0.5) / ks) for i in range(ks)} & set(range(max(n_frames, 1))))
            hits, max_conf = 0, 0.0
            for fi in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, frame = cap.read()
                if not ok:
                    continue
                res = model.predict(frame, device=fc["device"], conf=fc["conf"],
                                    classes=[dog_cls], verbose=False)
                boxes = res[0].boxes
                if boxes is not None and len(boxes) > 0:
                    hits += 1
                    max_conf = max(max_conf, float(boxes.conf.max()))
                # CPU 铁律: 不做任何 cuda 调用; predict(device='cpu') 已保证
            cap.release()
            rate = hits / max(len(idxs), 1)
            accepted = rate >= fc["min_dog_rate"]
            append_jsonl(self.f_stats, {
                "fragment_id": r["fragment_id"],
                "dog_rate": round(rate, 3), "hits": hits, "n_sampled": len(idxs),
                "max_conf": round(max_conf, 3),
                "width": w, "height": h, "fps_src": round(fps, 2),
                "accepted": bool(accepted),
                "judged_at": now_iso(),
            })
            if not accepted and path.exists():
                path.unlink()  # 未过筛不留盘, 只留统计
        acc = sum(1 for x in load_jsonl(self.f_stats) if x.get("accepted"))
        log(f"[filter] 完成, 过筛累计 {acc}")

    # ------------------------------------------------ manifest 阶段(契约格式 A)

    REQUIRED_FIELDS = ["fragment_id", "source_channel", "origin_url_or_path",
                       "capture_context", "species_note", "license_note", "collected_at"]

    def stage_manifest(self) -> None:
        stats = {r["fragment_id"]: r for r in load_jsonl(self.f_stats) if r.get("accepted")}
        frags = {r["fragment_id"]: r for r in load_jsonl(self.f_frag) if not r.get("error")}
        rows = []
        for fid, st in stats.items():
            fr = frags.get(fid)
            if not fr:
                continue
            rows.append({
                "fragment_id": fid,
                "source_channel": self.cfg["channel"],
                "origin_url_or_path": fr["origin_url_or_path"],
                "capture_context": (
                    f"query='{fr.get('query_kw','')}' lang={fr.get('lang','')} "
                    f"title='{fr.get('title','')[:80]}' "
                    f"clip={fr.get('start_s')}s-{fr.get('end_s')}s"),
                "species_note": self.cfg["manifest"]["species_note"],
                "license_note": self.cfg["manifest"]["license_note"],
                "collected_at": now_iso(),
                # ---- 扩展字段(契约 7 字段之外的质量元数据, 便于下游审查) ----
                "label_status": "rule_seed_pending",   # W6 引擎自产, 待 Q3a/Q3b 提点
                "duration_s": fr.get("duration_s"),
                "dog_rate": st.get("dog_rate"),
                "max_conf": st.get("max_conf"),
                "resolution": f"{st.get('width')}x{st.get('height')}",
                "fps_src": st.get("fps_src"),
            })
        rows.sort(key=lambda x: x["fragment_id"])
        tmp = self.f_man.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in rows:
                missing = [k for k in self.REQUIRED_FIELDS if k not in r]
                if missing:
                    raise ValueError(f"manifest 行缺契约字段 {missing}: {r['fragment_id']}")
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(self.f_man)
        log(f"[manifest] 写入 {len(rows)} 行 -> {self.f_man}")

    # ------------------------------------------------ 自检(纯函数验证)

    def stage_selftest(self) -> None:
        cases = [
            # (bounds, max_len, min_len, expected_segments)
            ([0, 10, 20, 40], 30, 1.5, [(0, 20), (20, 40)]),
            ([0, 5, 70], 30, 1.5, [(0, 5), (5, 35), (35, 65), (65, 70)]),
            ([0, 29, 31, 90], 30, 1.5, [(0, 29), (29, 31), (31, 61), (61, 90)]),
            ([0, 100], 30, 1.5, [(0, 30), (30, 60), (60, 90), (90, 100)]),
            ([0, 1.0], 30, 1.5, []),                      # 全部短于 min_len → 空
            ([0], 30, 1.5, []),
        ]
        for bounds, mx, mn, exp in cases:
            got = self.budget_segments([float(b) for b in bounds], mx, mn)
            assert all(b - a <= mx + 1e-6 for a, b in got), \
                f"超上限! bounds={bounds} -> {got}"
            assert got == exp, f"bounds={bounds}\n got={got}\n exp={exp}"
        log("[selftest] budget_segments 全部用例通过 (<=max_len 硬约束成立)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO / "configs/harvest_video_w25.yaml"))
    ap.add_argument("--stage", required=True,
                    choices=["search", "download", "split", "filter",
                             "manifest", "all", "selftest"])
    ap.add_argument("--limit", type=int, default=None,
                    help="download/split 阶段单次调用处理上限(分批防超时)")
    args = ap.parse_args()
    pipe = Pipeline(args.config)
    if args.stage in ("download", "split"):
        getattr(pipe, f"stage_{args.stage}")(limit=args.limit)
        return
    stages = ([args.stage] if args.stage != "all"
              else ["search", "download", "split", "filter", "manifest"])
    for s in stages:
        getattr(pipe, f"stage_{s}")()


if __name__ == "__main__":
    main()
