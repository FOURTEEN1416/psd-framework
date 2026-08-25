# W28 → 协调窗 问题上报单（2026-08-25）

> 上报窗口: W28 (wt/W28, C4 合成保真度)
> 呈递对象: 主检出协调窗（歆歆）
> 处置优先级: P1 = 阻塞下游/数据不实 | P2 = 门禁缺陷 | P3 = 教训固化
> 同步渠道: 本文档 + 共享记忆广播（shared 命名空间, 关键词 `W28上报`）

---

## P1【数据不实】partialclass4_T30.pkl 是冒烟残留，且 Q3b 从未执行

**现象（当次实勘证据）**
- `runs/public_real_dataset/partialclass4_T30.pkl` 实际内容 **1 clip × 30 帧 × 17 关节(COCO 拓扑)**, 6.5KB;
  同目录 manifest 却登记 172 视频样本计划; `partialclass4_extract_quality.json` 缺失。
- 主仓 `runs/relay_exec/state.json`: `"steps": []`; transcript 显示 relay 自 2026-08-25 09:05 起因 GPU 被 NTU 任务占满(≈7.6GiB)持续排队, **Q1-Q3c 五步均未开跑**。
- 17 点拓扑说明冒烟期用的是 COCO 权重默认值(`--weights yolo11s-pose.pt`), 与管线契约 `EXPECTED_KPTS=24`(dog-pose)不符。

**影响面**
- 任何把该 pkl 当"Q3b 真实提点产物"消费的任务都会拿到 n=1 的失真参考: C4 保真度参考(本窗已参数化规避)、Q3c 公开真实层微调、论文 tab2 中间列数字。
- 若无人知晓此节, 未来某时点可能直接引用 6.5KB 文件得出"全量提点结论"。

**建议处置（归协调窗）**
1. 立即: 将现 pkl 改名加 `.smoke` 后缀或移入 `runs/smoke/`, 防 downstream 误用;
2. Q3a 先行确认产出 dog-pose **24 点**权重后, 再放行 Q3b;
3. Q3b 重跑完成后通知 W28 —— C4 参考源已参数化, 一行命令刷新保真度数字:
   `python -m psd.data.synth_stgcn_v2 --reference-pkl <新pkl> --output-json reports/w28-c4-synth-v2-fidelity-vnext.json`
4. ⚠️ 若全量产物为 24 点拓扑, W28 模块会 fail-fast(设计行为), 需先做拓扑适配再重跑。

## P2【门禁缺陷】relay_executor.ps1 VerifyPaths 只查存在性, 不查内容

**现象**
- `Invoke-Step -VerifyPaths` 全部走 `Test-Path`——文件在即成功并自动 `git commit`。
- 本次 P1 的冒烟残留文件恰好能骗过这道门禁: 若 relay 曾以此状态跑过 Q3b 步, 会留下"172 视频提点完成"的虚假成功记录。

**风险**
残缺产物被盖章提交 → 下游(Q3c 微调 / 论文回填)拿坏数据继续跑, 错误传染到最终交付才暴露。

**建议处置（归协调窗）**
- VerifyPaths 支持内容断言: 最小样本数 / 数组维数(拓扑) / 必备元数据字段三选一即可拦截本次全部案例;
- 或每步产物强制附 `{n_samples, shape, sha256}` 元数据 JSON 并纳入校验。

## P3【教训固化】管线脚本默认参数与自身契约矛盾

**现象**
`run_p05_public_real_pipeline.py`: `EXPECTED_KPTS=24` fail-fast 写得很硬, 但 `--weights` 默认值是 COCO 17 点权重——脚本自己跟自己打架。fail-fast 只护住了 extract 循环内部分支, 冒烟期仍产出了 17 点文件落盘。

**普适教训**
fail-fast 校验若不覆盖**入口默认值**, 保护是漏的。

**建议**
`--weights` 改 `required=True`(强制显式传参), 或 main() 入口做"权重拓扑 vs EXPECTED_KPTS"预检。归 W20/pipeline owner。

## P4【制度提案】上游产物异常发现即上报（用户裁决 2026-08-25）

用户已指示: **以后发现类似问题都必须上报**。固化为跨窗口纪律:

1. 任一窗口开工勘察时, 发现上游产物与 truth/manifest 不一致(数量/拓扑/schema/时间戳矛盾), 必须:
   - ① 本窗报告设"上报事项"节登记;
   - ② 共享记忆广播(shared 命名空间);
   - ③ 报告中显式指向协调窗处置, 不自行越权修改他窗领地;
2. 冒烟产物一律带 `.smoke` 后缀或入 `runs/smoke/` 子目录, 与正式产物物理隔离;
3. 修复责任归 owner 窗口, 上报窗口只举证不动手——除非用户明示接管。

## 附: W28 已自愈部分（无需协调窗处理）

- C4 方法论全链已在 n=1 参考下打通并验证(TDD 23 绿, 全仓 319 绿), 参考源参数化;
- 旧行为冻结保护(md5)、非 coco17 fail-fast 门禁已常驻测试;
- 完整技术细节见 `reports/w28-c4-synth-v2-fidelity-2026-08-25.md` §2。
