#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
奖励跟踪和可视化工具
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import deque
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class RewardTracker:
    """奖励跟踪器"""
    
    def __init__(self, save_dir='reward_logs'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 数据存储
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_evacuation_rates = []
        self.episode_death_rates = []
        self.step_rewards = []
        
        # 滑动窗口统计
        self.window_size = 100
        self.recent_rewards = deque(maxlen=self.window_size)
        
        # 训练统计
        self.total_episodes = 0
        self.total_steps = 0
        
    def record_episode(self, episode, total_reward, steps, evacuation_rate, death_rate):
        """记录回合数据"""
        self.episode_rewards.append(total_reward)
        self.episode_steps.append(steps)
        self.episode_evacuation_rates.append(evacuation_rate)
        self.episode_death_rates.append(death_rate)
        
        self.recent_rewards.append(total_reward)
        self.total_episodes = episode + 1
        self.total_steps += steps
    
    def record_step(self, reward):
        """记录步骤奖励"""
        self.step_rewards.append(reward)
    
    def get_recent_average(self):
        """获取最近的平均奖励"""
        if len(self.recent_rewards) == 0:
            return 0
        return np.mean(self.recent_rewards)
    
    def get_statistics(self):
        """获取统计信息"""
        if len(self.episode_rewards) == 0:
            return {}
        
        return {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'avg_reward': np.mean(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'recent_avg_reward': self.get_recent_average(),
            'avg_evacuation_rate': np.mean(self.episode_evacuation_rates),
            'avg_death_rate': np.mean(self.episode_death_rates),
            'avg_steps_per_episode': np.mean(self.episode_steps)
        }
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        if not stats:
            print("暂无统计数据")
            return
        
        print(f"总回合数: {stats['total_episodes']}")
        print(f"总步数: {stats['total_steps']}")
        print(f"平均奖励: {stats['avg_reward']:.2f}")
        print(f"最高奖励: {stats['max_reward']:.2f}")
        print(f"最低奖励: {stats['min_reward']:.2f}")
        print(f"奖励标准差: {stats['std_reward']:.2f}")
        print(f"最近{self.window_size}回合平均奖励: {stats['recent_avg_reward']:.2f}")
        print(f"平均疏散率: {stats['avg_evacuation_rate']:.2%}")
        print(f"平均死亡率: {stats['avg_death_rate']:.2%}")
        print(f"平均每回合步数: {stats['avg_steps_per_episode']:.1f}")
    
    def save_data(self):
        """保存数据到文件"""
        data = {
            'episode_rewards': self.episode_rewards,
            'episode_steps': self.episode_steps,
            'episode_evacuation_rates': self.episode_evacuation_rates,
            'episode_death_rates': self.episode_death_rates,
            'step_rewards': self.step_rewards,
            'statistics': self.get_statistics()
        }
        
        # 保存为JSON
        with open(os.path.join(self.save_dir, 'reward_data.json'), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # 保存为CSV
        df = pd.DataFrame({
            'episode': range(len(self.episode_rewards)),
            'reward': self.episode_rewards,
            'steps': self.episode_steps,
            'evacuation_rate': self.episode_evacuation_rates,
            'death_rate': self.episode_death_rates
        })
        df.to_csv(os.path.join(self.save_dir, 'episode_data.csv'), index=False)
        
        print(f"奖励数据已保存到: {self.save_dir}")
    
    def load_data(self, filepath):
        """从文件加载数据"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.episode_rewards = data['episode_rewards']
        self.episode_steps = data['episode_steps']
        self.episode_evacuation_rates = data['episode_evacuation_rates']
        self.episode_death_rates = data['episode_death_rates']
        self.step_rewards = data['step_rewards']
        
        # 重建recent_rewards
        self.recent_rewards = deque(
            self.episode_rewards[-self.window_size:], 
            maxlen=self.window_size
        )
        
        self.total_episodes = len(self.episode_rewards)
        self.total_steps = sum(self.episode_steps)
    
    def plot_reward_curves(self, save_path=None, show=True):
        """绘制奖励曲线"""
        if len(self.episode_rewards) == 0:
            print("暂无数据可绘制")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        episodes = range(len(self.episode_rewards))
        
        # 子图1: 奖励曲线
        ax1.plot(episodes, self.episode_rewards, alpha=0.6, color='blue', label='每回合奖励')
        
        # 计算滑动平均
        if len(self.episode_rewards) >= self.window_size:
            moving_avg = []
            for i in range(self.window_size-1, len(self.episode_rewards)):
                avg = np.mean(self.episode_rewards[i-self.window_size+1:i+1])
                moving_avg.append(avg)
            
            ax1.plot(range(self.window_size-1, len(self.episode_rewards)), 
                    moving_avg, color='red', linewidth=2, 
                    label=f'{self.window_size}回合滑动平均')
        
        ax1.set_title('奖励学习曲线')
        ax1.set_xlabel('回合')
        ax1.set_ylabel('奖励')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 疏散率和死亡率
        ax2.plot(episodes, self.episode_evacuation_rates, color='green', 
                label='疏散率', linewidth=2)
        ax2.plot(episodes, self.episode_death_rates, color='red', 
                label='死亡率', linewidth=2)
        ax2.set_title('疏散效果')
        ax2.set_xlabel('回合')
        ax2.set_ylabel('比例')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 子图3: 每回合步数
        ax3.plot(episodes, self.episode_steps, color='purple', alpha=0.7)
        ax3.set_title('每回合步数')
        ax3.set_xlabel('回合')
        ax3.set_ylabel('步数')
        ax3.grid(True, alpha=0.3)
        
        # 子图4: 奖励分布直方图
        ax4.hist(self.episode_rewards, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax4.axvline(np.mean(self.episode_rewards), color='red', linestyle='--', 
                   label=f'平均: {np.mean(self.episode_rewards):.2f}')
        ax4.set_title('奖励分布')
        ax4.set_xlabel('奖励')
        ax4.set_ylabel('频次')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"奖励曲线图已保存: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_detailed_analysis(self, save_path=None, show=True):
        """绘制详细分析图"""
        if len(self.episode_rewards) == 0:
            print("暂无数据可绘制")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        episodes = range(len(self.episode_rewards))
        
        # 子图1: 奖励vs疏散率散点图
        scatter = ax1.scatter(self.episode_evacuation_rates, self.episode_rewards, 
                             c=episodes, cmap='viridis', alpha=0.6)
        ax1.set_title('奖励 vs 疏散率')
        ax1.set_xlabel('疏散率')
        ax1.set_ylabel('奖励')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='回合')
        
        # 子图2: 奖励vs死亡率散点图
        scatter2 = ax2.scatter(self.episode_death_rates, self.episode_rewards, 
                              c=episodes, cmap='plasma', alpha=0.6)
        ax2.set_title('奖励 vs 死亡率')
        ax2.set_xlabel('死亡率')
        ax2.set_ylabel('奖励')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='回合')
        
        # 子图3: 学习进度（分段平均）
        if len(episodes) >= 100:
            segment_size = len(episodes) // 10
            segment_rewards = []
            segment_evacuation = []
            segment_death = []
            
            for i in range(0, len(episodes), segment_size):
                end_idx = min(i + segment_size, len(episodes))
                segment_rewards.append(np.mean(self.episode_rewards[i:end_idx]))
                segment_evacuation.append(np.mean(self.episode_evacuation_rates[i:end_idx]))
                segment_death.append(np.mean(self.episode_death_rates[i:end_idx]))
            
            segment_episodes = range(len(segment_rewards))
            
            ax3_twin = ax3.twinx()
            
            line1 = ax3.plot(segment_episodes, segment_rewards, 'b-', 
                           linewidth=3, label='平均奖励')
            line2 = ax3_twin.plot(segment_episodes, segment_evacuation, 'g-', 
                                linewidth=3, label='平均疏散率')
            line3 = ax3_twin.plot(segment_episodes, segment_death, 'r-', 
                                linewidth=3, label='平均死亡率')
            
            ax3.set_title('学习进度（分段平均）')
            ax3.set_xlabel('训练阶段')
            ax3.set_ylabel('平均奖励', color='b')
            ax3_twin.set_ylabel('平均比例', color='g')
            
            # 合并图例
            lines = line1 + line2 + line3
            labels = [l.get_label() for l in lines]
            ax3.legend(lines, labels, loc='upper left')
        
        ax3.grid(True, alpha=0.3)
        
        # 子图4: 收敛性分析
        if len(self.episode_rewards) >= 200:
            # 计算不同窗口大小的滑动平均
            windows = [50, 100, 200]
            colors = ['red', 'green', 'blue']
            
            for window, color in zip(windows, colors):
                if len(self.episode_rewards) >= window:
                    moving_avg = []
                    for i in range(window-1, len(self.episode_rewards)):
                        avg = np.mean(self.episode_rewards[i-window+1:i+1])
                        moving_avg.append(avg)
                    
                    ax4.plot(range(window-1, len(self.episode_rewards)), 
                            moving_avg, color=color, linewidth=2, 
                            label=f'{window}回合滑动平均')
            
            ax4.set_title('收敛性分析')
            ax4.set_xlabel('回合')
            ax4.set_ylabel('滑动平均奖励')
            ax4.legend()
        
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"详细分析图已保存: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close() 