# DQN疏散系统使用指南

## 📁 项目结构

```
CA-dqn1/Louvre_Evacuation/
├── agents/                    # 智能体模块
│   ├── __init__.py
│   └── dqn_agent.py          # DQN智能体实现
├── envs/                     # 环境模块
│   ├── __init__.py
│   ├── evacuation_env.py     # 疏散环境
│   ├── fire_model.py         # 火灾模型
│   ├── map.py               # 地图系统
│   └── people.py            # 人员模型
├── utils/                    # 工具模块
│   ├── __init__.py
│   ├── visualization.py     # 可视化工具
│   └── reward_visualizer.py # 奖励追踪器
├── runners/                  # 运行脚本
│   └── train_dqn.py         # 标准训练脚本
├── tests/                    # 测试脚本
│   └── test_robot_static_analysis_corrected.py
├── dqn_results/             # 训练结果目录
│   ├── reward_logs/         # 奖励日志
│   └── *.pth               # 模型文件
├── configs/                 # 配置文件
│   └── dqn.yaml            # DQN配置
├── train_dqn.py            # 简化训练脚本
├── evaluate_model.py       # 模型评估脚本
├── visualize_progress.py   # 训练进度可视化
└── monitor_training.py     # 训练监控脚本
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install torch numpy matplotlib pandas pyyaml

# 进入项目目录
cd CA-dqn1/Louvre_Evacuation
```

### 2. 基础训练（推荐新手）

```bash
# 运行简化版训练脚本
python train_dqn.py
```

**说明：**
- 使用默认配置进行训练
- 自动创建结果目录
- 实时显示训练进度
- 自动保存最佳模型

### 3. 标准训练（推荐进阶用户）

```bash
# 运行完整版训练脚本
python runners/train_dqn.py
```

**说明：**
- 需要配置文件 `configs/dqn.yaml`
- 更完整的训练功能
- 详细的性能记录

## 📋 各文件详细使用说明

### 🎯 训练相关文件

#### 1. `train_dqn.py` - 简化训练脚本
**用途：** 快速开始训练，适合新手

**使用方法：**
```bash
python train_dqn.py
```

**特点：**
- ✅ 无需配置文件，使用内置默认配置
- ✅ 自动创建目录结构
- ✅ 实时显示训练进度
- ✅ 自动保存训练数据和模型

**输出文件：**
- `dqn_results/best_model.pth` - 最佳模型
- `dqn_results/reward_logs/` - 训练数据

#### 2. `runners/train_dqn.py` - 标准训练脚本
**用途：** 完整功能训练，适合进阶用户

**使用方法：**
```bash
# 确保配置文件存在
python runners/train_dqn.py
```

**配置文件：** 需要 `configs/dqn.yaml`

**特点：**
- 🔧 完全可配置的训练参数
- 📊 详细的性能记录
- 🎨 自动生成可视化图表

### 📊 监控和可视化文件

#### 3. `monitor_training.py` - 训练监控脚本
**用途：** 实时监控训练进度

**使用方法：**
```bash
# 实时监控模式
python monitor_training.py

# 生成当前进度图
python monitor_training.py plot
```

**功能：**
- 📈 实时显示训练统计
- 📊 生成进度图表
- ⏱️ 自定义更新间隔

#### 4. `visualize_progress.py` - 训练进度可视化
**用途：** 生成训练进度分析图

**使用方法：**
```bash
python visualize_progress.py
```

**输出：**
- `training_progress.png` - 训练进度四合一图表
- 控制台统计信息

**图表内容：**
- 奖励学习曲线
- 每回合步数变化
- 疏散率统计
- 奖励分布直方图

### 🎯 评估和测试文件

#### 5. `evaluate_model.py` - 模型评估脚本
**用途：** 评估训练好的模型性能

**使用方法：**
```bash
# 评估默认最佳模型
python evaluate_model.py

# 评估指定模型
python evaluate_model.py --model_path dqn_results/model_episode_100.pth
```

**功能：**
- 🎯 无探索评估（epsilon=0）
- 📊 生成评估报告
- 🎨 可视化最佳轨迹
- 📈 性能统计分析

**输出文件：**
- `evaluation_results.png` - 评估结果图
- `best_trajectory.png` - 最佳轨迹图

#### 6. `tests/test_robot_static_analysis_corrected.py` - 静态分析测试
**用途：** 测试机器人在固定位置的影响

**使用方法：**
```bash
cd tests/
python test_robot_static_analysis_corrected.py
```

**功能：**
- 🤖 机器人固定位置测试
- 👥 人员轨迹分析
- 💊 健康值变化监控
- 📊 疏散效果统计

### ⚙️ 核心模块文件

#### 7. `agents/dqn_agent.py` - DQN智能体
**说明：** 核心DQN算法实现

**主要类：**
- `DQNNetwork` - 神经网络结构
- `DQNAgent` - 智能体逻辑

**关键参数：**
```python
config = {
    'learning_rate': 0.0001,
    'gamma': 0.99,
    'epsilon': 1.0,
    'batch_size': 32,
    'memory_size': 50000
}
```

#### 8. `envs/evacuation_env.py` - 疏散环境
**说明：** 疏散模拟环境

**主要功能：**
- 🗺️ 地图管理
- 👥 人员模拟
- 🔥 火灾传播
- 🎯 奖励计算

#### 9. `utils/visualization.py` - 可视化工具
**说明：** 训练和评估可视化

**主要类：**
- `PerformanceRecorder` - 性能记录器
- `DQNTrainingVisualizer` - 训练可视化器

#### 10. `utils/reward_visualizer.py` - 奖励追踪器
**说明：** 奖励数据管理和可视化

**主要功能：**
- 📊 奖励数据记录
- 📈 学习曲线生成
- 📋 统计分析

## 🎮 使用流程

### 新手推荐流程

1. **开始训练**
   ```bash
   python train_dqn.py
   ```

2. **监控进度**
   ```bash
   # 新开终端
   python monitor_training.py
   ```

3. **查看结果**
   ```bash
   python visualize_progress.py
   ```

4. **评估模型**
   ```bash
   python evaluate_model.py
   ```

### 进阶用户流程

1. **配置参数**
   编辑 `configs/dqn.yaml`

2. **标准训练**
   ```bash
   python runners/train_dqn.py
   ```

3. **深度分析**
   ```bash
   python tests/test_robot_static_analysis_corrected.py
   ```

## 📊 输出文件说明

### 训练结果目录 `dqn_results/`

```
dqn_results/
├── reward_logs/
│   ├── episode_data.csv      # 回合数据
│   └── reward_data.json      # 详细奖励数据
├── best_model.pth           # 最佳模型（按总奖励）
├── best_evacuation_model.pth # 最佳疏散模型
├── model_episode_X.pth      # 定期保存的模型
└── *.png                    # 各种分析图表
```

### 可视化图表

- `training_progress.png` - 训练进度分析
- `evaluation_results.png` - 模型评估结果
- `best_trajectory.png` - 最佳轨迹可视化
- `robot_static_analysis_corrected.png` - 静态分析结果

## ⚙️ 配置参数说明

### DQN参数 (`configs/dqn.yaml`)

```yaml
# 环境配置
env:
  width: 36          # 地图宽度
  height: 30         # 地图高度
  num_people: 150    # 人员数量

# 智能体配置
agent:
  learning_rate: 0.0001    # 学习率
  gamma: 0.99              # 折扣因子
  epsilon: 1.0             # 初始探索率
  epsilon_min: 0.02        # 最小探索率
  epsilon_decay: 0.9995    # 探索率衰减
  batch_size: 32           # 批次大小
  memory_size: 50000       # 经验回放缓冲区大小

# 训练配置
episodes: 1000            # 训练回合数
save_frequency: 50        # 模型保存频率
```

## 🔧 常见问题解决

### 1. 内存不足
```bash
# 减少批次大小和缓冲区大小
# 在configs/dqn.yaml中修改：
agent:
  batch_size: 16
  memory_size: 10000
```

### 2. 训练速度慢
```bash
# 检查CUDA是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 减少人员数量进行快速测试
env:
  num_people: 50
```

### 3. 可视化显示问题
```bash
# 设置环境变量
export DISPLAY=:0

# 或者使用保存模式（不显示窗口）
python visualize_progress.py --no-display
```

## 📈 性能优化建议

### 训练优化
1. **GPU加速：** 确保CUDA可用
2. **批次大小：** 根据显存调整
3. **学习率：** 从0.001开始，逐步调整
4. **探索策略：** 平衡探索与利用

### 环境优化
1. **人员数量：** 测试时可减少到50-100人
2. **地图大小：** 保持36x30的标准大小
3. **最大步数：** 根据场景复杂度调整

## 🎯 进阶使用技巧

### 1. 自定义奖励函数
编辑 `envs/evacuation_env.py` 中的 `_calculate_reward` 方法

### 2. 修改网络结构
编辑 `agents/dqn_agent.py` 中的 `DQNNetwork` 类

### 3. 添加新的可视化
参考 `utils/visualization.py` 创建自定义图表

### 4. 实验记录
使用 `utils/reward_visualizer.py` 进行详细的实验分析

## 📞 技术支持

如遇到问题，请检查：
1. 依赖包是否正确安装
2. 配置文件格式是否正确
3. 文件路径是否存在
4. 显存是否足够

祝您训练顺利！🎉 