#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DQN模型导入和使用指南
演示如何加载训练好的模型并进行推理
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.dqn_agent import DQNAgent
from envs.evacuation_env import EvacuationEnv

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ModelLoader:
    """模型加载器"""
    
    def __init__(self, config_path='configs/dqn.yaml'):
        """初始化模型加载器"""
        self.config_path = config_path
        self.config = self.load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 使用设备: {self.device}")
        
        # 创建环境
        self.env = self.create_environment()
        
        # 创建智能体
        self.agent = self.create_agent()
        
    def load_config(self):
        """加载配置文件"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            print(f"⚠️  配置文件不存在: {self.config_path}")
            # 返回默认配置
            return {
                'environment': {
                    'width': 36,
                    'height': 30,
                    'num_people': 150,
                    'fire_zones': [[18, 14], [19, 15], [20, 16]],
                    'exit_location': [36, 15]
                },
                'agent': {
                    'learning_rate': 0.001,
                    'gamma': 0.99,
                    'epsilon': 0.0,  # 推理时不使用探索
                    'epsilon_min': 0.0,
                    'epsilon_decay': 1.0,
                    'memory_size': 10000,
                    'batch_size': 32,
                    'hidden_size': 256
                }
            }
    
    def create_environment(self):
        """创建环境"""
        env_config = self.config['environment']
        env = EvacuationEnv(
            width=env_config['width'],
            height=env_config['height'],
            num_people=env_config['num_people'],
            fire_zones=env_config['fire_zones'],
            exit_location=env_config['exit_location']
        )
        print(f"🌍 环境创建成功: {env.width}×{env.height}, {env.num_people}人")
        return env
    
    def create_agent(self):
        """创建智能体"""
        agent_config = self.config['agent']
        agent = DQNAgent(
            state_size=self.env.state_size,
            action_size=self.env.action_size,
            device=self.device,
            config=agent_config
        )
        print(f"🤖 智能体创建成功: 状态空间{self.env.state_size}, 动作空间{self.env.action_size}")
        return agent
    
    def load_model(self, model_path):
        """加载训练好的模型"""
        print(f"📥 正在加载模型: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            return False
        
        try:
            # 加载模型
            self.agent.load(model_path)
            self.agent.epsilon = 0.0  # 推理时不使用探索
            print(f"✅ 模型加载成功!")
            return True
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def run_single_episode(self, visualize=True):
        """运行单个回合"""
        print(f"\n🏃 开始运行单个回合...")
        
        # 重置环境
        state = self.env.reset()
        total_reward = 0
        step_count = 0
        done = False
        
        # 记录轨迹
        trajectory = []
        
        while not done:
            # 智能体选择动作
            action = self.agent.act(state)
            
            # 执行动作
            next_state, reward, done, info = self.env.step(action)
            
            # 记录数据
            trajectory.append({
                'step': step_count,
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'info': info
            })
            
            total_reward += reward
            state = next_state
            step_count += 1
            
            # 打印进度
            if step_count % 10 == 0:
                evacuation_rate = info.get('evacuation_rate', 0.0)
                death_rate = info.get('death_rate', 0.0)
                print(f"  步骤 {step_count}: 奖励={reward:.2f}, 疏散率={evacuation_rate:.1%}, 死亡率={death_rate:.1%}")
        
        # 最终结果
        final_info = trajectory[-1]['info']
        evacuation_rate = final_info.get('evacuation_rate', 0.0)
        death_rate = final_info.get('death_rate', 0.0)
        
        print(f"\n📊 回合结束:")
        print(f"  总步数: {step_count}")
        print(f"  总奖励: {total_reward:.2f}")
        print(f"  疏散率: {evacuation_rate:.1%}")
        print(f"  死亡率: {death_rate:.1%}")
        
        if visualize:
            self.visualize_episode(trajectory)
        
        return trajectory, total_reward, evacuation_rate, death_rate
    
    def run_multiple_episodes(self, num_episodes=10):
        """运行多个回合进行评估"""
        print(f"\n🎯 开始评估模型 ({num_episodes} 回合)...")
        
        results = {
            'rewards': [],
            'evacuation_rates': [],
            'death_rates': [],
            'steps': []
        }
        
        for episode in range(num_episodes):
            print(f"\n--- 回合 {episode + 1}/{num_episodes} ---")
            
            trajectory, reward, evacuation_rate, death_rate = self.run_single_episode(visualize=False)
            
            results['rewards'].append(reward)
            results['evacuation_rates'].append(evacuation_rate)
            results['death_rates'].append(death_rate)
            results['steps'].append(len(trajectory))
            
            print(f"回合 {episode + 1}: 奖励={reward:.2f}, 疏散率={evacuation_rate:.1%}, 死亡率={death_rate:.1%}")
        
        # 统计结果
        self.print_evaluation_results(results)
        self.plot_evaluation_results(results)
        
        return results
    
    def print_evaluation_results(self, results):
        """打印评估结果"""
        print(f"\n" + "="*50)
        print(f"📈 评估结果统计:")
        print(f"="*50)
        
        rewards = results['rewards']
        evacuation_rates = results['evacuation_rates']
        death_rates = results['death_rates']
        steps = results['steps']
        
        print(f"🎯 奖励统计:")
        print(f"  平均奖励: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"  最高奖励: {np.max(rewards):.2f}")
        print(f"  最低奖励: {np.min(rewards):.2f}")
        
        print(f"\n👥 疏散统计:")
        print(f"  平均疏散率: {np.mean(evacuation_rates):.1%} ± {np.std(evacuation_rates):.1%}")
        print(f"  最高疏散率: {np.max(evacuation_rates):.1%}")
        print(f"  最低疏散率: {np.min(evacuation_rates):.1%}")
        
        print(f"\n💀 死亡统计:")
        print(f"  平均死亡率: {np.mean(death_rates):.1%} ± {np.std(death_rates):.1%}")
        print(f"  最高死亡率: {np.max(death_rates):.1%}")
        print(f"  最低死亡率: {np.min(death_rates):.1%}")
        
        print(f"\n⏱️  步数统计:")
        print(f"  平均步数: {np.mean(steps):.1f} ± {np.std(steps):.1f}")
        print(f"  最多步数: {np.max(steps)}")
        print(f"  最少步数: {np.min(steps)}")
    
    def plot_evaluation_results(self, results):
        """绘制评估结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 奖励分布
        axes[0, 0].hist(results['rewards'], bins=10, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('奖励分布')
        axes[0, 0].set_xlabel('奖励值')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 疏散率分布
        axes[0, 1].hist([r*100 for r in results['evacuation_rates']], bins=10, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_title('疏散率分布')
        axes[0, 1].set_xlabel('疏散率 (%)')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 死亡率分布
        axes[1, 0].hist([r*100 for r in results['death_rates']], bins=10, alpha=0.7, color='red', edgecolor='black')
        axes[1, 0].set_title('死亡率分布')
        axes[1, 0].set_xlabel('死亡率 (%)')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 步数分布
        axes[1, 1].hist(results['steps'], bins=10, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].set_title('步数分布')
        axes[1, 1].set_xlabel('步数')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'model_evaluation_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 评估结果图表已保存: {filename}")
        plt.show()
    
    def visualize_episode(self, trajectory):
        """可视化单个回合"""
        print(f"📊 生成可视化图表...")
        
        # 提取数据
        steps = [t['step'] for t in trajectory]
        rewards = [t['reward'] for t in trajectory]
        evacuation_rates = [t['info'].get('evacuation_rate', 0.0) for t in trajectory]
        death_rates = [t['info'].get('death_rate', 0.0) for t in trajectory]
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 奖励曲线
        axes[0, 0].plot(steps, rewards, 'b-', linewidth=2)
        axes[0, 0].set_title('奖励变化')
        axes[0, 0].set_xlabel('步数')
        axes[0, 0].set_ylabel('奖励')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 疏散率曲线
        axes[0, 1].plot(steps, [r*100 for r in evacuation_rates], 'g-', linewidth=2)
        axes[0, 1].set_title('疏散率变化')
        axes[0, 1].set_xlabel('步数')
        axes[0, 1].set_ylabel('疏散率 (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 死亡率曲线
        axes[1, 0].plot(steps, [r*100 for r in death_rates], 'r-', linewidth=2)
        axes[1, 0].set_title('死亡率变化')
        axes[1, 0].set_xlabel('步数')
        axes[1, 0].set_ylabel('死亡率 (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 累积奖励
        cumulative_rewards = np.cumsum(rewards)
        axes[1, 1].plot(steps, cumulative_rewards, 'purple', linewidth=2)
        axes[1, 1].set_title('累积奖励')
        axes[1, 1].set_xlabel('步数')
        axes[1, 1].set_ylabel('累积奖励')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'episode_visualization_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 回合可视化图表已保存: {filename}")
        plt.show()


def main():
    """主函数 - 演示如何使用训练好的模型"""
    print("🚀 DQN模型导入和使用演示")
    print("="*50)
    
    # 1. 创建模型加载器
    loader = ModelLoader()
    
    # 2. 加载训练好的模型
    # 可以选择不同的模型文件
    model_paths = [
        'dqn_results/best_evacuation_model.pth', # 最佳疏散模型 
    ]
    
    # 尝试加载第一个可用的模型
    model_loaded = False
    for model_path in model_paths:
        if loader.load_model(model_path):
            model_loaded = True
            break
    
    if not model_loaded:
        print("❌ 没有找到可用的模型文件!")
        print("请确保已经完成训练并生成了模型文件。")
        return
    
    # 3. 选择运行模式
    print(f"\n🎮 选择运行模式:")
    print("1. 运行单个回合 (详细可视化)")
    print("2. 运行多个回合 (性能评估)")
    print("3. 两者都运行")
    
    try:
        choice = input("请输入选择 (1/2/3): ").strip()
        
        if choice == '1':
            # 运行单个回合
            loader.run_single_episode(visualize=True)
            
        elif choice == '2':
            # 运行多个回合
            num_episodes = int(input("请输入评估回合数 (默认10): ") or "10")
            loader.run_multiple_episodes(num_episodes)
            
        elif choice == '3':
            # 两者都运行
            print("\n🎯 首先运行单个回合...")
            loader.run_single_episode(visualize=True)
            
            print("\n🎯 然后运行多个回合评估...")
            num_episodes = int(input("请输入评估回合数 (默认5): ") or "5")
            loader.run_multiple_episodes(num_episodes)
            
        else:
            print("❌ 无效选择，运行单个回合...")
            loader.run_single_episode(visualize=True)
            
    except KeyboardInterrupt:
        print("\n⏹️  用户中断操作")
    except Exception as e:
        print(f"❌ 运行出错: {e}")
    
    print(f"\n✅ 程序运行完成!")


if __name__ == "__main__":
    main() 