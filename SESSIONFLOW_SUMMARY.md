# Sessionflow Summary

整理时间：2026-06-12
工作目录：E:\graph_quant

## 1. 本轮对话目标

用户希望理解当前项目 `graph_quant` 的整体思路，并确认它是否属于“基于图”的量化策略。

当前结论：

- 项目名包含 `graph` / `pressure_graph`，但现有代码并不是严格意义上的图策略。
- 更准确的定位是：永续合约压力路径研究框架。
- 它通过价格、成交量、资金费率、持仓量 OI、BTC 市场状态等变量，构造规则型路径信号，再进行标签验证、回测和执行现实检验。

## 2. 项目核心理解

项目研究问题：

能否用永续合约的价格、成交量、已结算 funding、open interest 等压力变量，识别未来 4h/12h 有更高上行命中率、且回撤风险可控的多头路径。

主流程：

1. 从交易所采集数据。
2. 构建 15m 级别特征。
3. 生成未来标签。
4. 按规则识别路径信号。
5. 做 bar/event 层统计。
6. 做 15m、1m、tick/public trades 级别执行验证。
7. 冻结候选策略并做成本、成交、baseline、集中度压力测试。

## 3. 关键代码地图

- CLI 入口：[src/pressure_graph/cli.py](E:/graph_quant/src/pressure_graph/cli.py)
- 总管线：[src/pressure_graph/pipeline.py](E:/graph_quant/src/pressure_graph/pipeline.py)
- 特征构建：[src/pressure_graph/features/build.py](E:/graph_quant/src/pressure_graph/features/build.py)
- 未来标签：[src/pressure_graph/labels/future.py](E:/graph_quant/src/pressure_graph/labels/future.py)
- 路径信号：[src/pressure_graph/paths/signals.py](E:/graph_quant/src/pressure_graph/paths/signals.py)
- 15m 回测：[src/pressure_graph/backtest/simulator.py](E:/graph_quant/src/pressure_graph/backtest/simulator.py)
- 入场策略：[src/pressure_graph/backtest/entry_policies.py](E:/graph_quant/src/pressure_graph/backtest/entry_policies.py)
- 1m 执行验证：[src/pressure_graph/backtest/minute_execution.py](E:/graph_quant/src/pressure_graph/backtest/minute_execution.py)
- tick 执行验证：[src/pressure_graph/backtest/trade_sequence.py](E:/graph_quant/src/pressure_graph/backtest/trade_sequence.py)
- v0 报告：[src/pressure_graph/reports/stats.py](E:/graph_quant/src/pressure_graph/reports/stats.py)
- v0.1 报告：[src/pressure_graph/reports/v01.py](E:/graph_quant/src/pressure_graph/reports/v01.py)
- v0.2 报告：[src/pressure_graph/reports/v02.py](E:/graph_quant/src/pressure_graph/reports/v02.py)

## 4. 主要路径信号

当前不是图节点/图边模型，而是三类规则路径：

- `short_squeeze`：OI 上升、funding 不过热、价格有韧性、成交量确认、BTC 未明显崩盘。
- `momentum_ignition`：价格 4h 分位强、成交量扩张、OI 温和上升、funding 仍安全。
- `crowded_long_risk`：funding 很热、OI 高、价格不强，用作多头 veto。

这里的 `path` 是“市场压力演化路径”，不是图论里的 path。

## 5. 防泄漏设计

项目比较重视时间对齐：

- `feature_time` 是当前 15m K 线收盘时间。
- `entry_time` 是下一根 K 线开盘。
- funding 和 OI 使用 backward as-of join，只使用已经发生的数据。
- future labels 从下一根 bar 开始计算，不使用当前 bar 的未来高低点。

这个设计说明项目目标是贴近可交易信号，而不是离线预测指标好看。

## 6. 当前实验状态

本地已存在完整生成物，包括：

- `data/processed/v0/perp_pressure_features.parquet`
- `reports/v0/*`
- `reports/v0_1/*`
- `reports/v0_2/*`

此前读取到的主特征表状态：

- 交易所：Bybit
- 标的数量：30 个 USDT 永续
- 行数：约 105 万条 15m bar
- 时间跨度：2025-06-03 到 2026-06-03 UTC

v0 holdout 结果倾向：

- `short_squeeze` 和 `momentum_ignition` 都显示 3%/4h 上行命中率提升。
- 但 v0 候选评级仍偏弱，主要因为回撤、集中度、执行质量还不够稳。

v0.2 执行层结果倾向：

- 若干冻结候选在 tick/public-trade 执行层有正净期望。
- 但 fill rate、成本假设、止损率、月份集中度仍是主要风险。
- 多个候选对 2026-05 单月贡献很高，说明还不能把它看成稳定策略。

## 7. 是否基于图策略

结论：不是。

缺失典型图策略组件：

- 没有资产节点和边。
- 没有 adjacency matrix。
- 没有 NetworkX / PyG / DGL。
- 没有中心性、社区发现、传播、GNN embedding。
- BTC 只是全市场状态变量，不是图中的中心节点传播模型。

更准确的描述：

这是一个“合约压力因子 + 规则路径信号 + 执行验证”的研究框架。

## 8. 如果要演化成真正图策略

可以考虑：

- 节点：每个币种/合约。
- 边：相关性、beta、资金费率同步性、OI 共振、成交量联动、orderflow 相似性。
- 图特征：中心性、邻居压力扩散、社区内压力同步、BTC/ETH 对山寨的传播延迟。
- 策略问题：压力是否会从中心资产传导到高相关资产，或从强势社区扩散到滞后币种。

## 9. 当前待办建议

优先级较高：

1. 降低月份集中度，特别是 2026-05 单月贡献。
2. 增加跨时间段、跨交易所或滚动窗口检验。
3. 对 v0.2 候选做更严格成本和滑点压力测试。
4. 进一步区分“信号有效”与“入场条件带来的选择偏差”。
5. 如果继续沿图方向发展，先定义资产关系边，再把现有 pressure path 变成节点/邻域特征。

## 10. Sessionflow 可用性说明

本次环境中未发现可直接调用的 `sessionflow`：

- 当前插件列表无 `sessionflow`。
- 本地 PATH 无 `sessionflow` 命令。
- Python 包索引未找到 `sessionflow`。
- npm 上仅发现 `sessionflow@0.0.0`，未返回可用 CLI 入口。

因此本文件采用 sessionflow 风格整理当前对话和项目理解，作为后续上下文接续材料。
