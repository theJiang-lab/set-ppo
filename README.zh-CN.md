# Set-PPO：论文对应代码

对应论文 **Task-Constrained Secure Alignment of Heterogeneous Semantic Knowledge for Web News Delivery**。

本仓库根据作者提供的论文重新实现语义环境、残差修复、分层对抗PPO和评测流程。原始训练代码、权重、具体数据划分与完整基线参数未随文件提供，因此不能宣称逐项复现论文结果。`data/reported/` 保存的是作者提供的实验汇总，和新代码的输出分开。按作者要求，本次不运行训练或实验。

## 使用

```bash
python -m pip install -e ".[text,test]"
setppo prepare --news data/raw/train/news.tsv data/raw/dev/news.tsv --config configs/paper.json --output data/prepared/mind.npz
setppo train --config configs/paper.json --dataset data/prepared/mind.npz --output runs/full/seed_0 --seed 0
setppo evaluate --checkpoint runs/full/seed_0/best.pt --dataset data/prepared/mind.npz --output runs/full/seed_0/test --episodes 500
```

这些是后续使用命令，本次没有执行。MIND数据需从[官方页面](https://msnews.github.io/)按其条款取得。编码器为冻结的all-MiniLM-L6-v2，按文章唯一ID划分80:10:10，PCA与K-means仅拟合训练集。选择模型只用验证集；测试集不参与阈值和checkpoint选择。

## 论文一致性

- 参考配置：4名读者、80项任务、160轮动作、96个原子、16维表示、10维残差、3个修复项、0.85服务门槛、20次观察预算。
- PPO：2,000轮、每轮64个episode、学习率0.0003、minibatch 256、4次更新、gamma 0.99、GAE 0.95、clip 0.2；每种子500项测试，5个种子。具体种子编号是实现选择。
- DATA完成率使用学习更新前的知识状态；KFB消耗动作但不直接完成任务；排队和阻塞任务不会被丢弃。
- DATA不发现未知原子，KFB可以；增长与新发现原子分别统计。
- 合法actor只使用任务描述、历史和知识信念；真实状态只用于集中式critic和明确标注的Oracle诊断。
- E0不能看当前元数据；E1只能额外看类型和目标读者；E0/E1不能看未选择消息的内容或KREQ。E2独立列示。
- 同种子内各方法共享初始任务和知识案例。运行结果包含配对指纹。

[公式对应表](docs/paper_alignment.md)见 `docs/paper_alignment.md`；[未报告参数与实现选择](docs/implementation_choices.md)单独列出。部分阈值、学习信号、网络宽度和参考观察者策略没有在论文给出，代码采用明确标注的可修改默认值，不冒充原实验配置。

## 基线和数据

可运行随机、knowledge-greedy、KFB-first及模块移除/选择器变体。CPO、RCPO、sTRPO-A/B/C原始训练配置及混合策略权重缺失，仓库保留它们的实测汇总和外部策略接入接口，不伪造训练实现或把PPO改名为这些方法。

```bash
setppo audit-reported --root .
```

该命令仅核对47条原结果、32项QA任务汇总和降幅计算，不运行实验。QA评分工具只统计已有作答记录；原始题目、作答模型与文章清单未提供，不能恢复原问答生成流程。

发布前已有16项单元测试通过；本次继续进行了语法、参数和汇总数字核对，没有运行训练或实验。详细范围见[验证记录](docs/validation.md)。

