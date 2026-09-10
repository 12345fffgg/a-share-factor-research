# A 股量价因子研究框架

一个面向 A 股截面因子研究的可复现示例项目，覆盖复权收益构造、因子分组、RankIC、分层收益、组合换手和因子综合评分。项目使用合成数据演示完整流程，可直接替换为经授权的行情与因子数据。

> 本仓库是公开展示版，不包含实习公司的代码、研究结论、聚宽原始数据或任何账号凭证。

## 研究流程

```text
行情与因子数据
      ↓
复权价格及多周期未来收益
      ↓
逐日截面排序与十分组划分
      ↓
RankIC / ICIR / 分层收益 / 换手率
      ↓
固定阈值评分 / 稳定性筛选 / 结果输出
```

## 功能特点

- 按股票独立计算 1、5、10、20 日未来复权收益，避免跨股票错位。
- 支持任意数量的因子批量评估，并设置最小截面样本门槛。
- 输出日度 Spearman RankIC、IC 均值、ICIR、正 IC 比例和多空收益差。
- 使用确定性的等频分组逻辑，处理因子值并列时仍可复现实验结果。
- 基于相邻交易日持仓集合交集计算最高分组换手率。
- 构建“基础有效性 70 分 + 近期表现 20 分 + 衰减保持度 10 分”的固定绝对评分，并加入方向、近期稳定性和跨股票池一致性筛选。
- 示例数据固定随机种子，克隆仓库后可复现全部演示结果。

## 因子评分 Notebook

[`notebooks/factor_scoring.ipynb`](notebooks/factor_scoring.ipynb) 直接读取演示结果，完成评分输入整理、创业板/中证 800 差异化阈值打分及跨股票池综合筛选。代码保持短小，评分细节集中在 `src/factor_scoring.py`，方便替换为真实研究数据。

## 快速运行

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run_demo.py
python -m unittest discover -s tests -v
```

运行完成后，`outputs/` 将包含：

- `factor_summary.csv`：因子在不同预测周期上的汇总指标；
- `daily_factor_results.csv`：日度 RankIC 与分组收益；
- `turnover_summary.csv`：最高分组换手统计。

## 接入真实数据

输入表采用长表结构，每行代表某只股票在某个交易日的一条观测。核心字段如下：

| 字段 | 含义 |
|---|---|
| `date` | 交易日期 |
| `code` | 股票代码 |
| `close` | 收盘价 |
| `adjustment_factor` | 复权因子 |
| 因子列 | 例如动量、波动率或经授权导出的 Alpha191 等因子 |

实际研究中还应加入历史股票池、停牌与涨跌停过滤，并根据可成交时点构造收益。多日收益存在重叠时，显著性检验应考虑序列相关。

## 项目结构

```text
.
├── src/factor_research.py       # 因子评价核心函数
├── src/factor_scoring.py        # 固定绝对评分与双股票池筛选
├── notebooks/factor_scoring.ipynb
├── tests/test_factor_research.py
├── run_demo.py                  # 可复现实验入口
├── outputs/                     # 示例输出
├── requirements.txt
└── README.md
```

## 后续方向

- 因子相关性、冗余控制与多因子合成；
- 滚动训练和样本外检验；
- 交易成本、滑点及成交约束；
- AI 辅助生成候选因子表达式，并接入统一评估流水线。

## 说明

本项目仅用于研究与技术展示，不构成投资建议。示例结果来自合成数据，不代表真实市场收益。
