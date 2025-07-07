# 🚀 DQN疏散系统快速参考

## 📋 一分钟快速开始

```bash
# 1. 进入目录
cd CA-dqn1/Louvre_Evacuation

# 2. 开始训练（新手推荐）
python train_dqn.py

# 3. 查看进度（新开终端）
python visualize_progress.py

# 4. 评估模型
python evaluate_model.py
```

## 📁 核心文件速查

| 文件名 | 用途 | 命令 |
|--------|------|------|
| `train_dqn.py` | 🎯 **新手训练** | `python train_dqn.py` |
| `runners/train_dqn.py` | 🔧 **进阶训练** | `python runners/train_dqn.py` |
| `visualize_progress.py` | 📊 **查看进度** | `python visualize_progress.py` |
| `monitor_training.py` | 👁️ **实时监控** | `python monitor_training.py` |
| `evaluate_model.py` | 🎯 **模型评估** | `python evaluate_model.py` |

## 🎮 使用场景

### 🆕 我是新手，想快速体验
```bash
python train_dqn.py
```
**特点：** 零配置，自动运行，适合快速上手

### 🔧 我想自定义参数
1. 编辑 `configs/dqn.yaml`
2. 运行 `python runners/train_dqn.py`

### 📊 我想看训练效果
```bash
python visualize_progress.py  # 静态图表
python monitor_training.py    # 实时监控
```

### 🎯 我想测试模型
```bash
python evaluate_model.py     # 性能评估
cd tests/ && python test_robot_static_analysis_corrected.py  # 深度分析
```

## 📊 输出文件说明

```
dqn_results/
├── best_model.pth              # ⭐ 最佳模型
├── reward_logs/
│   ├── episode_data.csv        # 📈 训练数据
│   └── reward_data.json        # 📊 详细奖励
└── *.png                       # 🎨 可视化图表
```

## ⚡ 常用命令组合

### 完整训练流程
```bash
# 终端1: 开始训练
python train_dqn.py

# 终端2: 监控进度
python monitor_training.py

# 训练完成后: 查看结果
python visualize_progress.py
python evaluate_model.py
```

### 快速测试流程
```bash
# 修改人员数量为50（快速测试）
# 编辑 train_dqn.py 中的 num_people=50

python train_dqn.py
python evaluate_model.py
```

## 🔧 问题速查

| 问题 | 解决方案 |
|------|----------|
| 💾 **内存不足** | 减少 `batch_size` 和 `memory_size` |
| 🐌 **训练太慢** | 减少 `num_people` 到 50-100 |
| 🖼️ **图表不显示** | 检查 matplotlib 后端设置 |
| 📁 **找不到文件** | 确保在 `CA-dqn1/Louvre_Evacuation/` 目录下 |

## 📈 性能参考

| 配置 | 人员数 | 训练时间 | 内存占用 |
|------|--------|----------|----------|
| 🚀 **快速测试** | 50人 | ~10分钟 | ~2GB |
| ⚖️ **标准配置** | 150人 | ~30分钟 | ~4GB |
| 🎯 **完整训练** | 150人 | ~2小时 | ~8GB |

## 🎯 成功指标

✅ **训练成功标志：**
- 疏散率 > 90%
- 死亡率 < 10%
- 平均奖励 > 3000
- 平均步数 < 200

✅ **文件生成确认：**
- `dqn_results/best_model.pth` 存在
- `training_progress.png` 生成
- `episode_data.csv` 有数据

---

💡 **提示：** 详细说明请查看 `README.md` 