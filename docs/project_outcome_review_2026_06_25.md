# 项目成果深度复盘 - 2026-06-25

本文回答三个问题：

1. 这个项目到目前到底产出了什么？
2. 这些产出有什么实际用途？
3. 现在能不能合理期待收益，或者应该止损收束？

结论先放前面：

当前项目产出了一个历史上有边际、结构完整、可继续 forward 验证的多头组合候选；没有产出可以上真钱的稳定策略；空头没有找到合格的自动 alpha，只留下了多头风险层；大部分新信息源目前只能作为诊断，不应进入交易决策。

## 1. 最终沉淀的核心资产

### 1.1 主多头候选

当前最有价值的交易结构是：

```text
Core:
    P2 CIC1+CIC2 max8

Management:
    O6 late-burst additive overflow
    CP60 weak-position pruning
    Protect_A cap2 false-exit protection

Diagnostics:
    low-coimpulse risk
    failure/risk-off overlay
    token/on-chain attention context
```

含义：

- `P2 max8` 是核心候选，不是 `MIR1 raw`。
- `CIC1+CIC2` 说明 alpha 更像 basket / burst continuation，而不是 top3/top5 精选。
- `O6` 是后段 burst 小仓 overflow，不是主信号。
- `CP60` 是 1 小时 no-follow-through 的弱仓清理器，不是入场 alpha。
- `Protect_A cap2` 是避免 CP60 在 beta-high context 误杀的保护层。

当前状态：

| 模块 | 当前定位 | 能否决策交易 |
|---|---|---|
| P2 max8 | core forward baseline | paper/shadow only |
| O6 | additive overflow shadow | no real-live |
| CP60 | weak-position pruning shadow | no real-live |
| Protect_A cap2 | research-improved shadow | no real-live |
| low-coimpulse router | risk diagnostic | no action |
| token attention | forward context | no action |
| failure risk-off | long risk-layer shadow/counterfactual | no active gate |

### 1.2 方法论和验证框架

项目真正有价值的产出不只是信号，还包括一套研究纪律：

- as-of gate 审计
- 1m execution 对齐
- selected vs skipped counterfactual
- portfolio capacity simulation
- checkpoint / overflow / protection ledger
- month-cap / leave-one-month / leave-one-symbol
- random / shuffled / density-matched controls
- cost stress: 20bp / 30bp / 50bp
- forward sample sufficiency gates
- global candidate verdict schema

这套东西的作用是：以后不会再因为一个漂亮的历史收益数字就误升级。

## 2. 历史上到底有没有 edge

有，但只在历史验证里成立；forward 还没证明。

历史 benchmark 摘要来自 `reports/v2_0_graph_motif_search/benchmarks.csv`：

| 结构 | full net20 | validation net20 | holdout net20 | full month_cap35 | full max drawdown proxy |
|---|---:|---:|---:|---:|---:|
| B0 P2 max8 | +10.93% | +1.68% | -2.46% | +10.14% | -4.08% |
| B1 P2 max8 + O6 | +12.25% | +1.68% | -2.46% | +12.18% | -4.28% |
| B2 P2 max8 + CP60 | +12.59% | +2.23% | -1.48% | +12.59% | -2.41% |
| B3 P2 max8 + CP60 + O6 | +13.77% | +2.23% | -1.48% | +13.77% | -2.41% |
| B4 P2 max8 + Protect_A cap2 + O6 | +14.76% | +2.31% | -1.48% | +14.76% | -2.41% |

解释：

- full sample 看起来不错。
- validation 仍为正。
- holdout 明确为负。
- CP60 把 holdout 亏损从约 -2.46% 降到 -1.48%，说明它有减亏作用。
- O6 和 Protect_A 在 forward/holdout 当前窗口里没有足够触发样本，因此不能证明。

所以，历史结果支持继续观察，但不支持真钱上线。

## 3. 当前 forward 证据

当前 forward 样本来自 `reports/v2_4_long_stack_promotion_audit/`：

| 结构 | trades | net20 | CP exits | protected exits | overflow trades | 结论 |
|---|---:|---:|---:|---:|---:|---|
| S0 P2 max8 | 12 | -2.46% | 0 | 0 | 0 | insufficient |
| S3 P2 max8 + CP60 + O6 | 12 | -1.48% | 7 | 0 | 0 | insufficient |
| S5 P2 max8 + Protect_A cap2 + O6 | 12 | -1.48% | 7 | 0 | 0 | insufficient |

解释：

- 当前 forward 不是正收益。
- CP60 在小样本里继续表现为减亏器。
- O6 没有 forward 触发样本。
- Protect_A 没有 forward protected-exit 样本。
- 当前样本远低于评价门槛。

当前不能说策略已能赚钱。

最准确表述是：

```text
历史上有结构性 edge。
forward 尚未证明。
当前只能 paper/shadow logging。
real-live 和 canary-live 都应继续禁用。
```

## 4. 空头线产出了什么

空头没有找到合格的自动 alpha。

已经关闭或拒绝：

- standalone short motifs
- failed-reclaim breakdown shorts
- CIC-failure shorts
- crowded-stall automated short
- relative-value beta short
- A1 cross-exchange downside lead-lag short as strategy

几个曾经看起来有希望的结果：

| 方向 | 看起来强的地方 | 为什么不能用 |
|---|---|---|
| Path C crowded-stall short | 约 +1.54%, win 62% | 85.06% alpha 来自单月，且 long worst month 也亏 |
| A1 downside lead-lag | N=56, net20 +1.18%, win 75% | 2025-10 贡献 63.7%, bootstrap CI 跨 0 |
| D2 / RV variants | 个别 3 个月窗口好看 | 6 个月复测转负或不稳 |
| CIC-failure short | 逻辑上直观 | CIC longs 多数恢复，short 反而亏 |

空头真正留下的资产是多头风险层：

- `F5`: CIC2-only same-symbol no-long, 48 bars
- `F3`: no-overflow only, 48 bars

它们的用途不是开空赚钱，而是在 failure context 下少开多头或不启用 overflow。

结论：

```text
没有可部署自动空头 alpha。
有 long risk-off context。
空头研究对收益的贡献方式是减少多头错误暴露，而不是生成独立 short PnL。
```

## 5. 其他研究线的结论

### 5.1 Orderbook / orderflow

结论：

- static orderbook ask-thin / upside-vacuum 假设失败。
- 15m aggregate orderflow ranking 不能解决 selected vs skipped。
- orderflow/orderbook 继续作为数据层和诊断层，不作为 selector。

实际价值：

- 证明了静态盘口和 15m 聚合主动买盘不足以解释 CIC capacity problem。
- 避免继续在低胜算 ranking 上烧成本。

### 5.2 Cross-exchange

结论：

- same-symbol Binance -> Bybit 15m lead-lag 不成立。
- 1m A7 lag pocket 表面很强，但月度集中严重，不能升级。
- Binance taker-buy 对 P2 有一点 context，但 random/shuffled 几乎打平。

实际价值：

- cross-exchange 当前是 diagnostic only。
- 不能作为 selector / gate / shadow portfolio。

### 5.3 Perp crowding / funding / OI

结论：

- high funding + high OI 是 no-long diagnostic，不是 short alpha。
- funding_not_hot + moderate OI + price impulse 不能成为 standalone long。
- funding_extreme + OI_low RV 过不了 strict pair cost。

实际价值：

- 作为状态描述字段保留。
- 不开空，不配对交易，不过滤 CIC。

### 5.4 On-chain / DEX attention

结论：

- market-level on-chain attention 有传播/环境信息。
- token-level attention 覆盖改善，但目前只能 forward context。
- CIC1 token-prior 24h 是最有希望的 context。

关键数字：

- CIC1 prior 24h: net20 约 +1.70%
- CIC1 without prior: net20 约 +1.04%
- lift 约 +0.66pp
- 但 live_action_allowed = false

实际价值：

- 放进 forward ledger。
- 未来可观察 CIC1 是否真的受 token attention 增强。
- 当前不能 skip / size / enable O6 / protect CP60。

### 5.5 Narrative / listing / catalyst

结论：

- real sector label 没打赢 random sector。
- listing 24h chase 差。
- 7d digestion 有右尾味道，但不是 catalyst alpha。

实际价值：

- atlas / diagnostic。
- 暂不继续投入主线资源。

## 6. 项目到底有没有收益潜力

有潜力，但没有证明。

收益潜力来自：

1. P2/CIC basket historical edge。
2. CP60 对弱仓的减亏能力。
3. O6 对 late-burst 的历史增量。
4. Protect_A 对 CP60 false-exit 的历史修正。
5. failure risk-off 对多头暴露的风险控制。

不能确认收益的原因：

1. forward trades 只有 12 笔，远不够。
2. holdout 是负的。
3. O6 forward 触发为 0。
4. Protect_A forward protected exits 为 0。
5. v2.5 execution realism 未通过。
6. v2.6 risk envelope 未最终通过。
7. 30bp / 50bp 成本压力下还不能证明可承受。

结论：

```text
可以继续低成本 paper/shadow。
不能上真钱。
不能扩大为更多历史挖掘。
```

## 7. 投入成本是否值得

如果目标是短期实盘盈利：目前不能证明值得。

如果目标是建立可审计的 crypto alpha 研究体系：已经产出明显价值。

具体价值：

1. 找到了一个可继续 forward 的核心候选：P2/CIC max8。
2. 识别出 alpha 形态：basket/burst continuation，而不是精选 top-k。
3. 建立了风险管理组件：CP60, O6, Protect_A, F3/F5。
4. 关掉了大量高诱惑但不稳的方向。
5. 建立了统一验证框架，降低未来误升级概率。

但投入必须收束。

继续投入的合理形式：

- 维护 forward ledger。
- 保证数据连续性。
- 做执行真实性审计。
- 周期性 promotion/demotion。

不合理的形式：

- 再手工挖一堆新 gate。
- 救已经失败的 orderbook/orderflow/crowding/cross-exchange。
- 为了找好看的历史结果继续扩大搜索。

## 8. 下一步决策标准

### 8.1 继续观察的最低条件

继续跑到这些样本：

| 组件 | 最低评价样本 |
|---|---:|
| P2 core trades | 100 |
| CP60 exits | 50 |
| Protect_A protected exits | 30 |
| O6 overflow trades | 30 |
| failure risk-off suppressions | 50 |
| token-prior P2/CIC trades | 100 |

### 8.2 升级条件

某个结构想升级，至少要满足：

- forward net20 > baseline
- net30 仍为正
- month_cap35 net20 > 0
- worst month / worst burst 不恶化
- 30bp 成本后不崩
- execution realism 通过
- risk envelope 通过
- 不是单 symbol / 单月 / 单 burst 支撑

### 8.3 降级条件

到样本门槛后：

- P2 core trades >= 100 且 net20 <= 0: 降级为 research-only。
- CP60 exits >= 50 且 delta <= 0: 移除 CP60 shadow。
- O6 trades >= 30 且 incremental net20 <= 0: 移除 O6 shadow。
- Protect_A exits >= 30 且 delta <= 0: 移除 Protect_A。
- failure risk-off suppressions >= 50 且伤害 net/drawdown: 降级为日志字段。

## 9. 对收益问题的最终回答

当前不能说“已经能赚钱”。

更准确的投资级结论：

```text
这是一个历史有边际、工程和验证框架较完整、但 forward 未证明的研究候选。
它距离真钱交易至少还差：
    1. 足够 forward 样本；
    2. 通过执行成本压力；
    3. 完成风险 envelope；
    4. 证明 O6/Protect_A 在未来样本里仍有贡献。
```

如果继续投入，应该把它当成一个受控实验，而不是一个即将上线的策略。

## 10. 当前建议

建议做三件事：

1. 停止广泛新 alpha 挖掘。
2. 继续低成本跑 S0/S3/S5 + diagnostics。
3. 达到样本门槛后做一次硬决策：升级、继续观察，或降级。

当前最重要的不是再聪明一点，而是让未来样本决定这套系统是不是值得继续。
