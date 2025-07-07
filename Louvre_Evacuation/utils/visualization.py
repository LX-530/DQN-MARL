#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化工具模块
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from collections import defaultdict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PerformanceRecorder:
    """性能记录器"""
    
    def __init__(self):
        self.episode_data = []
        self.step_data = []
    
    def record_episode(self, env, episode, total_reward):
        """记录回合数据"""
        metrics = env.get_performance_metrics()
        
        self.episode_data.append({
            'episode': episode,
            'total_reward': total_reward,
            'evacuated': metrics['evacuated'],
            'dead': metrics['dead'],
            'remaining': metrics['remaining'],
            'evacuation_rate': metrics['evacuation_rate'],
            'death_rate': metrics['death_rate'],
            'avg_health': metrics['avg_health'],
            'min_health': metrics['min_health'],
            'total_steps': metrics['total_steps']
        })
    
    def record_step(self, env, step, action, reward):
        """记录步骤数据"""
        metrics = env.get_performance_metrics()
        
        self.step_data.append({
            'step': step,
            'action': action,
            'reward': reward,
            'evacuated': metrics['evacuated'],
            'dead': metrics['dead'],
            'remaining': metrics['remaining'],
            'avg_health': metrics['avg_health']
        })
    
    def get_dataframe(self):
        """获取数据框"""
        return pd.DataFrame(self.episode_data)


def visualize_trajectories(env, save_path=None):
    """可视化轨迹"""
    plt.figure(figsize=(12, 10))
    
    # 绘制地图
    plt.xlim(0, env.width)
    plt.ylim(0, env.height)
    plt.gca().set_aspect('equal')
    
    # 绘制火源区域
    for pos in env.fire_zones:
        fire_rect = patches.Rectangle(pos, 1, 1, 
                                    linewidth=1, edgecolor='red', 
                                    facecolor='red', alpha=0.3)
        plt.gca().add_patch(fire_rect)
    
    # 绘制出口
    exit_rect = patches.Rectangle((env.exit_location[0]-1, env.exit_location[1]-1), 2, 2, 
                                linewidth=2, edgecolor='green', 
                                facecolor='green', alpha=0.5)
    plt.gca().add_patch(exit_rect)
    
    # 绘制机器人轨迹
    if hasattr(env, 'robot_trajectory') and env.robot_trajectory:
        robot_x = [pos[0] for pos, _ in env.robot_trajectory]
        robot_y = [pos[1] for pos, _ in env.robot_trajectory]
        plt.plot(robot_x, robot_y, 'b-', linewidth=3, label='机器人轨迹', alpha=0.8)
        
        # 标记起始和结束位置
        plt.scatter(robot_x[0], robot_y[0], color='blue', s=100, marker='o', 
                   edgecolor='black', linewidth=2, label='机器人起始')
        plt.scatter(robot_x[-1], robot_y[-1], color='blue', s=100, marker='s', 
                   edgecolor='black', linewidth=2, label='机器人结束')
    
    plt.title('疏散轨迹图', fontsize=14, fontweight='bold')
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


class DQNTrainingVisualizer:
    """DQN训练可视化器"""
    
    def __init__(self, env, agent, save_dir='dqn_results'):
        self.env = env
        self.agent = agent
        self.save_dir = save_dir
        
        # 数据存储
        self.trajectories = {}
        self.health_data = {}
        self.evacuation_times = {}
        self.death_times = {}
        self.simulation_data = []
        self.selected_person_ids = []
        
    def run_evaluation_episode(self, num_selected=20):
        """运行评估回合并收集数据"""
        # 重置环境
        state = self.env.reset()
        
        # 随机选择要观察的人员
        all_person_ids = [p.id for p in self.env.people.list]
        self.selected_person_ids = np.random.choice(
            all_person_ids, 
            min(num_selected, len(all_person_ids)), 
            replace=False
        ).tolist()
        
        # 初始化数据存储
        for person_id in self.selected_person_ids:
            self.trajectories[person_id] = []
            self.health_data[person_id] = []
            self.evacuation_times[person_id] = None
            self.death_times[person_id] = None
        
        step = 0
        max_steps = 1000
        
        while step < max_steps:
            # 记录选定人员的数据
            self._record_step_data(step)
            
            # 智能体选择动作（不探索）
            action = self.agent.act(state, training=False)
            
            # 执行动作
            next_state, reward, done = self.env.step(action)
            
            step += 1
            state = next_state
            
            # 记录整体模拟数据
            metrics = self.env.get_performance_metrics()
            self.simulation_data.append({
                'step': step,
                'evacuated': metrics['evacuated'],
                'dead': metrics['dead'],
                'remaining': metrics['remaining'],
                'avg_health': metrics['avg_health']
            })
            
            if done:
                break
        
        # 计算疏散时间
        self._calculate_evacuation_times()
        
        return step
    
    def _record_step_data(self, step):
        """记录每步的数据"""
        for person in self.env.people.list:
            if person.id in self.selected_person_ids:
                # 记录轨迹
                self.trajectories[person.id].append({
                    'step': step,
                    'pos': person.pos,
                    'health': person.health,
                    'savety': person.savety,
                    'dead': person.dead
                })
                
                # 记录健康数据
                self.health_data[person.id].append({
                    'step': step,
                    'health': person.health
                })
    
    def _calculate_evacuation_times(self):
        """计算每个人的疏散时间和死亡时间"""
        for person_id in self.selected_person_ids:
            trajectory = self.trajectories[person_id]
            for point in trajectory:
                # 记录疏散时间
                if point['savety'] and self.evacuation_times[person_id] is None:
                    self.evacuation_times[person_id] = point['step']
                    break
                # 记录死亡时间
                if point['dead'] and self.death_times[person_id] is None:
                    self.death_times[person_id] = point['step']
                    break
    
    def create_four_panel_visualization(self, save_path=None):
        """创建四面板可视化图表（与静态分析相同）"""
        plt.figure(figsize=(20, 16))
        
        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 子图1: 轨迹图
        self._plot_trajectories(ax1)
        
        # 子图2: 健康值变化曲线
        self._plot_health_curves(ax2)
        
        # 子图3: 疏散时间分布
        self._plot_evacuation_times(ax3)
        
        # 子图4: 整体模拟统计
        self._plot_simulation_stats(ax4)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 DQN训练结果可视化图已保存: {save_path}")
            plt.close()
        else:
            plt.show()
    
    def _plot_trajectories(self, ax):
        """绘制轨迹图"""
        # 设置地图范围
        ax.set_xlim(0, 36)
        ax.set_ylim(0, 30)
        ax.set_aspect('equal')
        
        # 绘制人员初始分布区域
        init_area = patches.Rectangle((0, 0), 15, 30, 
                                    linewidth=2, edgecolor='lightblue', 
                                    facecolor='lightblue', alpha=0.2)
        ax.add_patch(init_area)
        
        # 绘制火源区域
        fire_rect = patches.Rectangle((18, 14), 3, 3, 
                                    linewidth=2, edgecolor='red', 
                                    facecolor='red', alpha=0.3)
        ax.add_patch(fire_rect)
        
        # 绘制出口
        exit_rect = patches.Rectangle((35, 14), 2, 2, 
                                    linewidth=2, edgecolor='green', 
                                    facecolor='green', alpha=0.5)
        ax.add_patch(exit_rect)
        
        # 绘制机器人轨迹
        if hasattr(self.env, 'robot_trajectory') and self.env.robot_trajectory:
            robot_x = [pos[0] for pos, _ in self.env.robot_trajectory]
            robot_y = [pos[1] for pos, _ in self.env.robot_trajectory]
            ax.plot(robot_x, robot_y, 'b-', linewidth=3, label='机器人轨迹', alpha=0.8)
            
            # 标记起始和结束位置
            ax.scatter(robot_x[0], robot_y[0], color='blue', s=100, marker='o', 
                      edgecolor='black', linewidth=2)
            ax.scatter(robot_x[-1], robot_y[-1], color='blue', s=100, marker='s', 
                      edgecolor='black', linewidth=2)
        
        # 绘制选定人员的轨迹
        colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(self.selected_person_ids)))
        
        for i, person_id in enumerate(self.selected_person_ids):
            if person_id in self.trajectories:
                trajectory = self.trajectories[person_id]
                if trajectory:
                    x_coords = [point['pos'][0] for point in trajectory]
                    y_coords = [point['pos'][1] for point in trajectory]
                    
                    # 绘制轨迹线
                    ax.plot(x_coords, y_coords, color=colors[i], 
                           alpha=0.7, linewidth=2, label=f'Person {person_id}')
                    
                    # 标记起始点
                    ax.scatter(x_coords[0], y_coords[0], color=colors[i], 
                              s=80, marker='o', edgecolor='black', linewidth=2)
                    
                    # 标记结束点
                    if len(x_coords) > 1:
                        ax.scatter(x_coords[-1], y_coords[-1], color=colors[i], 
                                  s=80, marker='s', edgecolor='black', linewidth=2)
        
        ax.set_title(f'DQN智能体疏散轨迹图\n选定人员轨迹 ({len(self.selected_person_ids)}人)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    def _plot_health_curves(self, ax):
        """绘制健康值变化曲线"""
        colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(self.selected_person_ids)))
        
        # 计算平均健康值
        all_steps = set()
        for person_id in self.selected_person_ids:
            if person_id in self.health_data:
                for point in self.health_data[person_id]:
                    all_steps.add(point['step'])
        
        all_steps = sorted(all_steps)
        avg_health = []
        
        for step in all_steps:
            step_healths = []
            for person_id in self.selected_person_ids:
                if person_id in self.health_data:
                    for point in self.health_data[person_id]:
                        if point['step'] == step:
                            step_healths.append(point['health'])
                            break
            if step_healths:
                avg_health.append(np.mean(step_healths))
            else:
                avg_health.append(100)
        
        # 绘制平均健康值曲线（粗线）
        if all_steps and avg_health:
            ax.plot(all_steps, avg_health, color='red', linewidth=4, 
                   label='平均健康值', alpha=0.8)
        
        # 绘制个体健康值曲线（细线）
        for i, person_id in enumerate(self.selected_person_ids):
            if person_id in self.health_data:
                health_data = self.health_data[person_id]
                if health_data:
                    steps = [point['step'] for point in health_data]
                    healths = [point['health'] for point in health_data]
                    ax.plot(steps, healths, color=colors[i], alpha=0.4, 
                           linewidth=1, label=f'Person {person_id}')
        
        ax.set_title('健康值变化曲线', fontsize=14, fontweight='bold')
        ax.set_xlabel('时间步')
        ax.set_ylabel('健康值')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_evacuation_times(self, ax):
        """绘制疏散时间分布 - 显示所有150人的数据"""
        # 基于模拟数据重新计算所有人的时间分布
        evacuation_times_by_step = {}
        death_times_by_step = {}
        
        prev_evacuated = 0
        prev_dead = 0
        
        for data_point in self.simulation_data:
            step = data_point['step']
            current_evacuated = data_point['evacuated']
            current_dead = data_point['dead']
            
            # 计算这一步新增的疏散和死亡人数
            new_evacuated = current_evacuated - prev_evacuated
            new_dead = current_dead - prev_dead
            
            # 为新疏散的人员分配时间
            for _ in range(new_evacuated):
                if step not in evacuation_times_by_step:
                    evacuation_times_by_step[step] = 0
                evacuation_times_by_step[step] += 1
            
            # 为新死亡的人员分配时间
            for _ in range(new_dead):
                if step not in death_times_by_step:
                    death_times_by_step[step] = 0
                death_times_by_step[step] += 1
            
            prev_evacuated = current_evacuated
            prev_dead = current_dead
        
        # 重建时间列表
        all_evacuation_times = []
        all_death_times = []
        
        for step, count in evacuation_times_by_step.items():
            all_evacuation_times.extend([step] * count)
        
        for step, count in death_times_by_step.items():
            all_death_times.extend([step] * count)
        
        # 统计未疏散且未死亡的人数
        total_people = self.env.num_people
        not_evacuated = total_people - len(all_evacuation_times) - len(all_death_times)
        
        # 绘制疏散时间分布
        if all_evacuation_times:
            ax.hist(all_evacuation_times, bins=min(25, len(set(all_evacuation_times))), 
                   alpha=0.7, color='green', edgecolor='black', label='成功疏散')
            
            # 添加统计信息
            mean_time = np.mean(all_evacuation_times)
            median_time = np.median(all_evacuation_times)
            ax.axvline(mean_time, color='red', linestyle='--', 
                      label=f'平均疏散时间: {mean_time:.1f}')
            ax.axvline(median_time, color='orange', linestyle='--', 
                      label=f'中位数: {median_time:.1f}')
        
        # 绘制死亡时间分布
        if all_death_times:
            ax.hist(all_death_times, bins=min(15, len(set(all_death_times))), 
                   alpha=0.7, color='red', edgecolor='black', label='死亡时间')
        
        ax.set_title(f'全部{total_people}人时间分布\n成功疏散: {len(all_evacuation_times)}人, 死亡: {len(all_death_times)}人, 未疏散: {not_evacuated}人', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('时间（步数）')
        ax.set_ylabel('人数')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_simulation_stats(self, ax):
        """绘制整体模拟统计"""
        if not self.simulation_data:
            return
        
        df = pd.DataFrame(self.simulation_data)
        
        # 绘制疏散进度
        ax.plot(df['step'], df['evacuated'], label='疏散人数', color='green', linewidth=2)
        ax.plot(df['step'], df['dead'], label='死亡', color='red', linewidth=2)
        ax.plot(df['step'], df['remaining'], label='剩余', color='blue', linewidth=2)
        
        ax.set_title('整体疏散进度', fontsize=14, fontweight='bold')
        ax.set_xlabel('时间步')
        ax.set_ylabel('人数')
        ax.grid(True, alpha=0.3)
        ax.legend() 