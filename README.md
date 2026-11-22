# 多视角共享注意力

samples/scene.json 是本地合成的视角和监督交接。可检查资料的形状、来源关系或交换文件边界。时间序列条目表示调用方送达或发起读取的顺序，数值为输入。

完整交换约定见 [交换格式](exchange.md)。运行环境为 Python 3.11 和 NumPy 2.2.6，使用本地文件即可运行资料工具。

```sh
python3 -m pip install -r requirements.txt
python3 inspect_exchange.py samples/scene.json
python3 -m unittest discover -s tests -v
```

## 共享注意力算子

`shared_attention.py` 直接消费导出器的分组提示，与完整逻辑序列同语义：

- `plan_from_exchange(data, visibility=None)` 解析一次请求；`visibility=` 可应用下一次请求的视角隔离范围，共享关系按新计划重新判定。只读样本之外的视角组合用 `build_plan` 以同样的结构接入。
- `forward(plan, observe=..., budget_bytes=...)` 返回各查询输出、归一化对数值（空查询为 -inf）、按逻辑成员标明归属的稀疏概率，以及账目 stats。
- `backward(plan, heads, budget_bytes=...)` 把监督位置的梯度按 Q/K/V 拥有者归并；同一查询被多个监督位置引用时先累加其全部有效贡献。

打分 q·k 按 (Q 拥有者, group, K 拥有者) 去重，同组成员不合并概率，数值相同的不同拥有者各自独立计分与更新；输出与梯度按拥有者代数归并。掩码成员不占概率也不产梯度。中间临时区逐查询计费（只含实际构造的分数、概率与待归并梯度），绝不铺出完整 查询×成员 分数矩阵，超出 `budget_bytes` 抛 `BudgetExceeded`；stats 中的 `score_dots`/`logical_scores`/`peak_temp_bytes` 供核对共享确实省去了乘加。
