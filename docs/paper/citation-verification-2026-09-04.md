# 引用权威核验登记（2026-09-04 · 代理通道全量验证）

> 背景: 用户要求引用必须权威、逐条实际查询。本轮开启代理后走 arXiv API / CVF / HF / 官方仓四通道。

## 已验证（权威源直证）

| bib key | 权威渠道 | 验证结果 |
|---------|---------|---------|
| singh2021tcl | arXiv API 2102.02751 + 官方项目页 | ✅ 作者七人确认；**bib 作者名勘误已落地**（Omprakash Chakraborty / Ashutosh Varshney 替换错误名） |
| yang2023aptv2 | arXiv API 2312.15612 | ✅ 标题+作者四人（Yuxiang Yang/Yingqi Deng/Yufei Xu/Jing Zhang）一致 |
| yan2018stgcn | arXiv API 1801.07455 | ✅ 作者三人：Sijie Yan/Yuanjun Xiong/Dahua Lin（**bib 已补真实作者**） |
| duan2022posec3d | arXiv API 2104.13586 | ✅ 五作者一致（Haodong Duan/Yue Zhao/Kai Chen/Dahua Lin/Bo Dai） |
| caron2021dino | arXiv API 2104.14294 | ✅ 七作者全名单确认（**bib 已补全**：Caron/Touvron/Misra/Jégou/Mairal/Bojanowski/Joulin） |
| gokay2025smq | arXiv API 2508.04513 | ✅ 四作者一致（Uzay Gökay/Federico Spurio/Dominik R. Bach/Juergen Gall） |
| skel2img2026 | arXiv API 2603.05963 | ✅ 九作者确认 + **正式题名更正**："Skeleton-to-Image Encoding: Enabling Skeleton Representation Learning via Vision-Pretrained Models"（bib 已更正） |
| ng2022animalkingdom | arXiv API 2204.08129 | ✅ 六作者确认（Xun Long Ng/Kian Eng Ong/Qichen Zheng/Yun Ni/Si Yong Yeo/Jun Liu，**bib 已修正**）+ arXiv comment 证实 **CVPR 2022 Oral** |
| guo2022aimclr | 官方仓 README + awesome 交叉 | ✅ 标题/venue/79.18 口径 |
| plosone2023imu | HF 伦理句验证时的 DA 关联 | ⚠️ 原文页本轮未直证（PMC10284380 保留 Scholar 项） |

## 本轮 bib 修正清单（4 条）

1. yan2018stgcn：补真实作者 3 人
2. caron2021dino：补全 7 人（原 "and others"）
3. skel2img2026：补 9 作者 + 题名更正为正式标题
4. ng2022animalkingdom：作者名修正为 arXiv 官方写法 + Oral 标注

## 遗留（Scholar 终审收窄后剩余 2 条）

- plosone2023imu（PMC10284380 作者）
- yolo2025petx（作者+DOI）

## 渠道受限说明

AK CVF openaccess 直连 404（CVF 站内 URL 变体与当日目录均无该条目，疑为 openaccess 收录页迁移）；改走 arXiv 官方 API 已直证。InterPet4D HF 卡代理通道 200，伦理三条原句二次证实。
