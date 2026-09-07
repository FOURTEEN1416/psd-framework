# NTU RGB+D 60 许可留档（投稿就绪门 R7 / 许可 B）

> 取证: 2026-09-03 | 取证方式: 官方发布页公开声明（本数据经由官方渠道获取, 论文 DA 声明引用如下口径）
> **⚠️ 域名勘误（2026-09-07）**: 本文件原记 "rosetta.robu.org" 有误（该域名实为德国 STEREON 玻璃仪器公司, 与 NTU 无关）；官方站 = **rose1.ntu.edu.sg**（NTU ROSE Lab, Prof. Jun Liu 组）。以下口径已按官方页 rose1.ntu.edu.sg/dataset/actionRecognition/ §6-§7 原文核对一致, 全页截图存证 `ntu-license-tou-screenshot-2026-09-07.png`（2026-09-07）。

## 官方许可口径（NTU RGB+D / NTU RGB+D 120 Action Recognition Dataset, ROSE Lab）

NTU RGB+D 由南洋理工大学 ROSE Lab 发布。官方 §6 Terms & Conditions of Use 原文要点：
1. 数据仅限**学术研究用途**（academic research only），教育/研究机构非商业免费；
2. 未经 ROSE Lab 明示许可，**redistribution / derivation（从本数据集生成新数据集）/ commercial usage 均属 illegal**；
3. 隐私条款：数据集人物图像仅可用于学术出版物与报告的展示；
4. §7 强制要求：使用数据集的出版物**必须包含官方 acknowledgement 语录**并引用 Shahroudy et al. CVPR 2016 与 Liu et al. TPAMI 2019。

## 本仓使用合规性

- 获取渠道: 官方申请口径（consortium 版本经标准 research-use 条款获取）
- 论文用途: 实现等价性验证（NTU60 xsub 三流线性评估）+ 跨域保留率复现（E9/E9b/TRANS-001），非商业
- 再分发: 论文 DA 声明明示派生骨架不随包分发，仅提供管线脚本（合规于 §6 禁再分发条款）
- 引用: Shahroudy 2016 + Liu 2020 均在引用池（**2026-09-07 R23a 勘误并修复**: liu2020ntu 此前实际缺失,已补入 refs.bib + E9b 段 \citep——NTU120 数据被使用故 TPAMI 条目为许可 §7 强制项）
- **官方 §7 ack 语录已补入正文**（main.tex Acknowledgements 节,官方原文逐字,2026-09-07）——该合规缺口由本次截图行动发现并闭合。官方 §7 逐字模板（留档为未来验证基准）: "(Portions of) the research in this paper used the NTU RGB+D (or NTU RGB+D 120) Action Recognition Dataset made available by the ROSE Lab at the Nanyang Technological University, Singapore."（正文因两数据集均使用而取 "(and)" 适配）

## 遗留动作

- [x] 投稿打包时: 截取下载页 Terms of Use 页面归档本文件旁 ✅ **2026-09-07 已补**——`ntu-license-tou-screenshot-2026-09-07.png`（官方页全页截图，1920×1080）
- [x] 官方 §7 acknowledgement 语录补入正文 ✅ 2026-09-07
