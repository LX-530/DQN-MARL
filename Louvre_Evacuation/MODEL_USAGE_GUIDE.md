# 🤖 DQN模型导入和使用指南

## 📋 概述

本指南详细介绍如何加载和使用训练好的DQN疏散模型。

## 🗂️ 模型文件说明

训练完成后，在 `dqn_results/` 目录下会生成以下模型文件：

| 文件名 | 说明 | 使用场景 |
|--------|------|----------|
| `best_model.pth` | 总体最佳模型 | 综合性能最好的模型 |
| `best_evacuation_model.pth` | 疏散效果最佳模型 | 专门针对疏散率优化的模型 |
| `model_episode_X.pth` | 特定回合模型 | 训练过程中保存的检查点 |

## 🚀 快速开始

### 方法1：使用完整的模型加载器

```bash
# 进入项目目录
cd CA-dqn1/Louvre_Evacuation

# 运行模型加载器
python load_and_use_model.py
```

### 方法2：使用现有的评估脚本

```bash
# 运行模型评估
python evaluate_model.py
```

## 💻 代码示例

### 基本模型加载

```python
import torch
from agents.dqn_agent import DQNAgent
from envs.evacuation_env import EvacuationEnv

# 1. 创建环境
env = EvacuationEnv(
    width=36,
    height=30,
    num_people=150,
    fire_zones=[[18, 14], [19, 15], [20, 16]],
    exit_location=[36, 15]
)

# 2. 创建智能体
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = DQNAgent(
    state_size=env.state_size,
    action_size=env.action_size,
    device=device,
    config={
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon': 0.0,  # 推理时不使用探索
        'hidden_size': 256
    }
)

# 3. 加载模型
model_path = 'dqn_results/best_model.pth'
agent.load(model_path)
agent.epsilon = 0.0  # 确保推理时不探索

print("✅ 模型加载成功!")
```

### 运行单个回合

```python
# 重置环境
state = env.reset()
total_reward = 0
step_count = 0
done = False

print("🏃 开始疏散模拟...")

while not done:
    # 智能体选择动作
    action = agent.act(state)
    
    # 执行动作
    next_state, reward, done, info = env.step(action)
    
    # 更新状态
    state = next_state
    total_reward += reward
    step_count += 1
    
    # 打印进度
    if step_count % 10 == 0:
        evacuation_rate = info.get('evacuation_rate', 0.0)
        death_rate = info.get('death_rate', 0.0)
        print(f"步骤 {step_count}: 疏散率={evacuation_rate:.1%}, 死亡率={death_rate:.1%}")

# 输出最终结果
print(f"\n📊 疏散完成!")
print(f"总步数: {step_count}")
print(f"总奖励: {total_reward:.2f}")
print(f"疏散率: {info.get('evacuation_rate', 0.0):.1%}")
print(f"死亡率: {info.get('death_rate', 0.0):.1%}")
```

### 批量评估

```python
def evaluate_model(agent, env, num_episodes=10):
    """评估模型性能"""
    results = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += reward
        
        results.append({
            'episode': episode + 1,
            'reward': total_reward,
            'evacuation_rate': info.get('evacuation_rate', 0.0),
            'death_rate': info.get('death_rate', 0.0)
        })
        
        print(f"回合 {episode + 1}: 奖励={total_reward:.2f}, "
              f"疏散率={info['evacuation_rate']:.1%}")
    
    return results

# 运行评估
results = evaluate_model(agent, env, num_episodes=10)

# 计算平均性能
avg_reward = sum(r['reward'] for r in results) / len(results)
avg_evacuation = sum(r['evacuation_rate'] for r in results) / len(results)
avg_death = sum(r['death_rate'] for r in results) / len(results)

print(f"\n📈 平均性能:")
print(f"平均奖励: {avg_reward:.2f}")
print(f"平均疏散率: {avg_evacuation:.1%}")
print(f"平均死亡率: {avg_death:.1%}")
```

## 🔧 高级用法

### 自定义环境参数

```python
# 创建自定义环境
custom_env = EvacuationEnv(
    width=40,           # 自定义地图宽度
    height=35,          # 自定义地图高度
    num_people=200,     # 自定义人员数量
    fire_zones=[[20, 15], [21, 16]],  # 自定义火源位置
    exit_location=[40, 17]  # 自定义出口位置
)

# 注意：如果环境参数与训练时不同，可能影响模型性能
```

### 模型性能分析

```python
import matplotlib.pyplot as plt
import numpy as np

def analyze_performance(results):
    """分析模型性能"""
    rewards = [r['reward'] for r in results]
    evacuation_rates = [r['evacuation_rate'] for r in results]
    death_rates = [r['death_rate'] for r in results]
    
    # 创建图表
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 奖励分布
    axes[0].hist(rewards, bins=10, alpha=0.7, color='blue')
    axes[0].set_title('奖励分布')
    axes[0].set_xlabel('奖励')
    axes[0].set_ylabel('频次')
    
    # 疏散率分布
    axes[1].hist([r*100 for r in evacuation_rates], bins=10, alpha=0.7, color='green')
    axes[1].set_title('疏散率分布')
    axes[1].set_xlabel('疏散率 (%)')
    axes[1].set_ylabel('频次')
    
    # 死亡率分布
    axes[2].hist([r*100 for r in death_rates], bins=10, alpha=0.7, color='red')
    axes[2].set_title('死亡率分布')
    axes[2].set_xlabel('死亡率 (%)')
    axes[2].set_ylabel('频次')
    
    plt.tight_layout()
    plt.savefig('model_performance_analysis.png', dpi=300)
    plt.show()

# 使用示例
analyze_performance(results)
```

## 🎯 使用场景

### 1. 单次疏散模拟
适用于：
- 验证模型效果
- 观察疏散过程
- 生成可视化结果

### 2. 批量性能评估
适用于：
- 模型性能统计
- 不同模型对比
- 稳定性测试

### 3. 参数敏感性分析
适用于：
- 测试不同环境参数
- 评估模型泛化能力
- 寻找最优配置

## ⚠️ 注意事项

### 1. 模型兼容性
- 确保环境参数与训练时一致
- 检查模型文件是否完整
- 验证PyTorch版本兼容性

### 2. 性能考虑
- GPU加速推理速度更快
- 批量评估时注意内存使用
- 大规模环境可能需要更多计算资源

### 3. 结果解释
- 模型性能可能因随机性而有波动
- 多次运行取平均值更可靠
- 关注疏散率和死亡率的平衡

## 🐛 常见问题

### Q: 模型加载失败
**A:** 检查以下几点：
- 模型文件路径是否正确
- 模型文件是否完整（未损坏）
- PyTorch版本是否兼容
- 设备（CPU/GPU）是否匹配

### Q: 推理结果不理想
**A:** 可能的原因：
- 环境参数与训练时不匹配
- 模型训练不充分
- 测试环境过于复杂
- 随机种子影响

### Q: 内存不足
**A:** 解决方案：
- 减少批量评估的回合数
- 使用CPU而非GPU
- 减少环境中的人员数量
- 关闭不必要的可视化

## 📞 技术支持

如果遇到问题，请检查：
1. 模型文件是否存在且完整
2. 依赖库是否正确安装
3. 环境配置是否匹配
4. 日志输出中的错误信息

## 🔗 相关文件

- `load_and_use_model.py` - 完整的模型加载器
- `evaluate_model.py` - 模型评估脚本
- `agents/dqn_agent.py` - DQN智能体实现
- `envs/evacuation_env.py` - 疏散环境实现
- `configs/dqn.yaml` - 配置文件

---

✅ **祝你使用愉快！** 如有问题，请参考代码注释或联系开发者。 