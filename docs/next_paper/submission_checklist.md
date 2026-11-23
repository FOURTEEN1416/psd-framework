# 投稿清单与期刊选择 — Anchor Recycling（2026-11-24）

> 状态：稿件终态 = `docs/next_paper/paper_manuscript_v1.md` + `latex/main.pdf`
> （30 页 elsarticle 单栏、0 overfull、0 `??`、110+ 项数字审计 AUDIT_OK、
> 41 条文献全引用、highlights 5 条合规、cover letter 模板已备）。
> 注意：CVIU 投稿包（e59fd06）是**另一篇论文**（PSD 框架），其三动作
> （作者信息、COI、EM 提交）仍待用户完成，与本清单互不影响。

---

## 一、三个候选期刊对比（页数/周期为经验估计，以官网最新作者指南为准）

| 维度 | IEEE TMM | PRL | CVIU |
|:--|:--|:--|:--|
| 范围契合 | 骨架行为识别+多模态感知，贴近期刊"人类中心多媒体理解" | 机制聚焦+边界研究，Letter 体量合适 | 理解导向视觉系统、负结果地图、非人灵长类 tier |
| 格式 | IEEEtran 双栏（需从 elsarticle 转排） | Elsevier 单栏（现稿直接可用，需压缩） | Elsevier 单栏（现稿直接可用） |
| 篇幅 | 常规稿约 13–14 页双栏（含参考文献，以官网为准） | Letter 约 8–10 页 | 无硬性页限，30 页可投 |
| 需要的加工 | **重排 IEEEtran + 压缩约 50%**（工作量大） | 压缩到 ~10 页（删阴性细节入附录/_supp） | 基本零加工 |
| 首次决定周期（估计） | 3–6 个月 | 2–3 个月 | 2–4 个月 |
| 与本稿证据的匹配 | 好 | 好 | 好 |
| 主要风险 | 范围偏"多媒体"会被问为何不是视频理解主流设置；体量被砍 | novelty 是否足够"Letter 级" | 影响因子较低，但命中率高 |

**梯队建议**：TMM（冲高）→ PRL（快速）→ CVIU/Neurocomputing（稳）。若想先稳后冲，可倒序。
无论顺序如何，**只能同时投一家**；被拒后携带评审意见转投下一家。

## 二、提交前统一清单（任一期刊都要）

- [ ] `latex/main.pdf` 与 `main.tex`（或转换后的目标格式）最新编译，0 error/0 overfull/0 `??`
- [ ] `highlights.tex`（Elsevier 刊需要；IEEE 不需要）
- [ ] `cover_letter.md` 填入期刊名/主编/文章类型，删除另两段定制段
- [ ] 作者信息、单位、通讯作者邮箱、ORCID（**用户动作**）
- [ ] COI 声明、数据/代码可用性声明（**用户确认口径**）
- [ ] 图源文件（`docs/next_paper/figures/*.pdf`）按投稿系统要求上传
- [ ] 证据包：`reports/p1_sprint_evidence/`（审稿人要求补料时秒回）

## 三、若投 TMM：转排工作清单（估 0.5–1 天）

1. `main.tex` 迁移到 IEEEtran（`\documentclass[journal]{IEEEtran}`），单栏→双栏；
2. 表格改 `table*`/`figure*` 通栏或收窄；附录 A–E 转 IEEE 的 supplementary；
3. 参考文献从内联 `thebibliography` 转 BibTeX（IEEEtran bst）；
4. 全文压缩 ~40%（阴性套件细节、敏感性网格移 supplementary）；
5. 重新编译 + 数字审计（audit 脚本对 md 仍有效，tex 检查需按新模板调整）。

## 四、若投 PRL：压缩工作清单（估 0.5 天）

1. 保留：Abstract、Intro（压半页）、Method（3.x 合并）、Results 主表+主图、
   Discussion（合并四段）、Conclusion；
2. 移除/压缩：阴性套件逐臂细节（保留 Finding 1 一段+指向附录）、
   预算曲线与敏感性网格（保留结论一句话）、附录 A–E 全部转 supplementary 或删除；
3. 目标 9–10 页。

## 五、硬性红线（沿用既定纪律）

- 不同时投多家；被拒后带评审意见转投。
- 不做 arXiv 预印本（既定纪律）。
- 每次投稿前重跑 `audit_next_paper.py` + `check_latex_next_paper.py`，全绿才交。
