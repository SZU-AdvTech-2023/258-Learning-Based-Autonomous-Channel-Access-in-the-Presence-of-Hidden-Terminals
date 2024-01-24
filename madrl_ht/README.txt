# MADRL_HT
- `[DIR] agent` 包含各种无线节点的接入协议（部署在节点端的actor网络）
    - `agent.py` Actor模板类
    - `csma.py` CSMA/CA协议
    - `ppo.py` PPO智能接入协议（Actor部分）实现
- `[DIR] critic` 包含AP上部署的critic网络
    - `critic.py` Critic模板类
    - `ppo.py` PPO智能接入协议（Critic部分）实现
- `[DIR] models` 主要用于探究不同神经网络模型的影响
    - `bilstm.py` 即论文中使用的BiLSTM神经网络架构
- `[DIR] some_plot_func` 一些用于数据可视化函数的模板
- `buffer.py` PPO算法的经验池逻辑实现
- `configs.py` 包含一些无线网络拓扑结构
- `main.py` 主函数入口
- `multi_agent_env.py` 无线网络的仿真环境
- `multi_agent_manager.py` 多个无线节点的管理器
    - `__init__` 函数中可手动配置节点所部署的接入协议（默认都为PPO协议，可配置为与传统协议共存的情形，但需注意观测空间配置）
    - `correct_exp_after` 函数实现了观测回顾机制的逻辑
- `utils.py` 工程中可能用到的一些函数

# 运行

`main.py` 是主函数入口
1. `get_args` 函数中配置默认的实验配置，或可通过命令行参数的方式配置实验配置
    - `--logdir` 实验目录（有相同目录默认删除覆盖）
    - `--task` 节点个数
    - `--topo` 参照 `configs.py` 中拓扑结构
    - `--plot-smooth-window` 运行时实时的吞吐量指标的滑动窗口大小
2. `__main__` 处开始，`range` 调整多次实验的次数

