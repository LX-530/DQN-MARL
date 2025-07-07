#!/usr/bin/env python3
# python CA-dqn1/Louvre_Evacuation/runners/train_dqn.py
# -*- coding: utf-8 -*-
"""
DQN疏散系统训练脚本
"""

import sys
import os

# 将项目根目录添加到Python路径，以解决模块导入问题
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import yaml
import torch
import numpy as np
from Louvre_Evacuation.agents.dqn_agent import DQNAgent
from Louvre_Evacuation.envs.evacuation_env import EvacuationEnv
from Louvre_Evacuation.utils.visualization import PerformanceRecorder, visualize_trajectories
from Louvre_Evacuation.utils.reward_visualizer import RewardTracker

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def train_dqn():
    """训练DQN智能体"""
    print("开始DQN疏散系统训练...")
    print("=" * 50)
    
    # 设置环境变量
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    # --- 最终路径修复：configs目录与Louvre_Evacuation模块同级 ---
    config_path = os.path.join(project_root, 'configs', 'dqn.yaml')
    config_path = os.path.normpath(config_path)
    
    # 加载配置
    config = load_config(config_path)
    
    # 创建保存目录 - 基于项目根目录构建路径
    save_dir = os.path.join(project_root, config.get('save_path', 'dqn_results'))
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建奖励跟踪器
    reward_tracker = RewardTracker(save_dir=os.path.join(save_dir, 'reward_logs'))
    
    # 创建环境
    env_config = config['env']
    env = EvacuationEnv(
        width=env_config['width'],
        height=env_config['height'],
        fire_zones=env_config['fire_zones'],
        exit_location=env_config['exit_location'],
        num_people=env_config['num_people']
    )
    
    print(f"环境创建成功:")
    print(f"  - 地图尺寸: {env.width}×{env.height}")
    print(f"  - 人员数量: {env.num_people}")
    print(f"  - 出口位置: {env.exit_location}")
    print(f"  - 状态空间: {env.state_size}")
    print(f"  - 动作空间: {env.action_size}")
    
    # 创建智能体
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent_config = config['agent']
    agent = DQNAgent(env.state_size, env.action_size, device, agent_config)
    
    print(f"\n智能体创建成功:")
    print(f"  - 使用设备: {device}")
    print(f"  - 网络参数: {sum(p.numel() for p in agent.q_network.parameters()):,}")
    print(f"  - 学习率: {agent_config['learning_rate']}")
    print(f"  - 批次大小: {agent_config['batch_size']}")
    
    # 训练参数
    episodes = config['episodes']
    update_target_freq = config.get('update_target_freq', 50)
    
    # 性能记录器
    performance_recorder = PerformanceRecorder()
    
    # 训练循环
    print(f"\n开始训练 {episodes} 个回合...")
    print("-" * 50)
    
    best_reward = float('-inf')
    recent_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < env.max_steps:
            # 启用探索
            action = agent.act(state, training=True)
            
            # 执行动作 - 修正以接收4个返回值
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            agent.remember(state, action, reward, next_state, done)
            
            # 记录奖励
            total_reward += reward
            reward_tracker.record_step(reward)
            
            # 更新状态
            state = next_state
            steps += 1
            
            # 训练智能体
            if len(agent.memory) > agent.batch_size:
                agent.learn()
            
            if done:
                break
        
        # 更新目标网络
        if episode % update_target_freq == 0:
            agent.update_target_network()
        
        # 获取性能指标
        metrics = env.get_performance_metrics()
        evacuation_rate = metrics['evacuation_rate']
        death_rate = metrics['death_rate']
        
        # 记录回合数据
        reward_tracker.record_episode(
            episode=episode,
            total_reward=total_reward,
            steps=steps,
            evacuation_rate=evacuation_rate,
            death_rate=death_rate
        )
        
        performance_recorder.record_episode(env, episode, total_reward)
        
        # 更新最佳奖励
        if total_reward > best_reward:
            best_reward = total_reward
            # 保存最佳模型
            agent.save(os.path.join(save_dir, 'best_model.pth'))
        
        # 记录最近奖励
        recent_rewards.append(total_reward)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)
        
        # 简化打印进度 - 每50回合打印一次
        if episode % 50 == 0 or episode == episodes - 1:
            avg_recent_reward = np.mean(recent_rewards)
            print(f"Episode {episode:4d}: "
                  f"Reward={total_reward:7.2f}, "
                  f"Avg100={avg_recent_reward:7.2f}, "
                  f"Steps={steps:3d}, "
                  f"Evac={evacuation_rate:.2%}, "
                  f"Death={death_rate:.2%}, "
                  f"ε={agent.epsilon:.4f}")
    
    # 训练完成
    print("\n" + "="*50)
    print("训练完成！")
    print("="*50)
    
    # 保存最终模型
    agent.save(os.path.join(save_dir, 'dqn_model.pth'))
    print(f"模型已保存到: {os.path.join(save_dir, 'dqn_model.pth')}")
    
    # 保存奖励数据
    reward_tracker.save_data()
    
    # 生成最终的奖励分析图
    try:
        print("\n生成最终奖励分析图...")
        
        # 基础奖励曲线
        reward_tracker.plot_reward_curves(
            save_path=os.path.join(save_dir, 'final_reward_curves.png'),
            show=False
        )
        
        # 详细分析图
        reward_tracker.plot_detailed_analysis(
            save_path=os.path.join(save_dir, 'detailed_analysis.png'),
            show=False
        )
        
        print("奖励分析图生成完成！")
        
    except Exception as e:
        print(f"生成奖励分析图失败: {e}")
    
    # 最终统计
    print("\n=== 最终训练统计 ===")
    reward_tracker.print_statistics()
    
    print(f"\n最佳奖励: {best_reward:.2f}")
    print(f"最终探索率: {agent.epsilon:.4f}")
    
    # 保存性能数据
    df = performance_recorder.get_dataframe()
    df.to_csv(os.path.join(save_dir, 'training_performance.csv'), index=False)
    print(f"性能数据已保存到: {os.path.join(save_dir, 'training_performance.csv')}")
    
    print("\n训练完成！所有结果已保存到:", save_dir)
    
    return agent, reward_tracker

if __name__ == "__main__":
    try:
        agent, tracker = train_dqn()
        print("\n🎉 训练成功完成！")
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc() 