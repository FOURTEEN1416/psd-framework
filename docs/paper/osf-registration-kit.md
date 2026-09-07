# OSF 注册材料包（R22b#7：预注册外部时间戳）

> 目的：把 6 份预注册协议放到第三方时间戳平台（OSF),堵"repo 自证预注册"攻击（R22b#7 MAJOR）。
> **你的手工步骤约 30 分钟**：注册 → 建 project → 上传下述文件 → 把 OSF DOI 填回这里。

## 你要做的（osf.io）

1. 注册账号（osf.io，GitHub 账号可直接登录）
2. New project → Title: `PSD Framework — Pre-registered Protocols (Pattern Recognition submission)` → 勾选 **Private→Public**（投稿时转 Public，或直接 Public）
3. 上传下表"上传文件"列的 6 份文件（仓库根 `docs/paper/` 下）
4. 项目页右侧 Copy DOI（形如 `10.17605/OSF.IO/XXXXX`）
5. 把 DOI 填到下表"回填"列 → 通知任一 AI 窗口，正文/cover letter 的引用句会自动接线

## 上传清单（全部为冻结态,可带 git SHA 作证）

| # | 上传文件 | 协议 ID | 回填（OSF DOI） |
|---|---|---|---|
| 1 | `k9-pilot-preregistration.md` | PSD-K9-PREREG-001 | __________ |
| 2 | `ak-v2-expansion-preregistration.md` | PSD-AKV2-PREREG-001 | __________ |
| 3 | `ntu-lowres-preregistration.md` | PSD-NTU-PREREG-001 | __________ |
| 4 | `ntu120-preregistration.md` | PSD-NTU120-PREREG-001 | __________ |
| 5 | `ucf101-preregistration.md` | PSD-UCF101-PREREG-001 | __________ |
| 6 | `panaf-preregistration.md` | PSD-PANAF-PREREG-001 | __________ |
| 7 | `ntu-transition-preregistration.md` | PSD-NTU-TRANS-001 | __________ |

> 建议：同时记录上传时的 commit SHA（当前 `d7908c9` 之后的最新 master），双证据链（OSF 时间戳 + git SHA）。

## 注册后接线（AI 窗口执行,勿手工改稿）

- 正文 §4.2 统计协议段加一句：*"All pre-registered protocols are additionally timestamped on the Open Science Framework (DOI: <DOI>)."*
- Cover letter 第 1 点（pre-registration discipline）同步加 OSF DOI
- `submission-package-draft.md` 刷新清单勾选本项
- R22b#7 关闭，review-log 登记

## 若不注册（Cover letter 说明口径，备选）

> "Protocol freeze is evidenced by the public repository's commit history: each protocol document was committed with a dated pre-registration note before its first experimental run (see `docs/paper/*preregistration*.md`, tag `review-snapshot`)."

说服力弱一档（R22b 判定 MAJOR 的原话：*"without any verifiable third-party timestamp"*）。
