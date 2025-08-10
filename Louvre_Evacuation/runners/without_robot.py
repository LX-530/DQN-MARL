#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器人静态位置分析测试 - 修正版
直接导入map.py和people.py，人员分布在0-7.5范围内
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import pandas as pd

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 直接导入map和people模块
from envs.map import Map, Init_Exit, Init_Barrier
from envs.people import People, Person

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class CorrectedRobotStaticAnalysis:
    """修正版机器人静态位置分析类"""
    
    # def __init__(self, robot_position=(15, 15), num_people=150, selected_people=20):
    def __init__(self, num_people=150, selected_people=20):
        """
        初始化分析器
        
        Args:
            num_people: 总人员数量
            selected_people: 选择观察的人员数量
        """
        self.num_people = num_people
        self.selected_people = selected_people
        
        # 创建地图 - 使用新的凸字形走廊布局
        self.map = Map(
            L=36,  # 长度
            W=30,  # 宽度
            E=Init_Exit((36, 15)),  # 主出口位置
            B=[]  # 移除额外障碍物，使用内建的走廊布局
        )
        
        # 数据存储
        self.selected_person_ids = []
        self.trajectories = {}
        self.health_data = {}
        self.evacuation_times = {}
        self.death_times = {}  # 添加死亡时间记录
        self.simulation_data = []
        
        # 创建人员（修正版位置生成）
        self._create_corrected_people()
        
    def _create_corrected_people(self):
        """创建人员，确保分布在0-7.5范围内"""
        print(" 创建人员分布（0-15范围）...")
        
        # 创建人员列表
        people_list = []
        
        # 在0-7.5范围内生成人员位置
        for i in range(self.num_people):
            # X坐标：1-7范围内（避免边界）
            pos_x = random.uniform(0, 14.5)
            # Y坐标：1-7范围内（避免边界）  
            pos_y = random.uniform(0, 29.5)
            
            # 确保位置有效
            while not self.map.Check_Valid(int(pos_x), int(pos_y)):
                pos_x = random.uniform(0, 14.5)
                pos_y = random.uniform(0, 29.5)
            
            person = Person(i+1, pos_x, pos_y)
            people_list.append(person)
        
        # 创建People对象并手动设置人员列表
        self.people = People(0, self.map)  # 先创建空的People对象
        self.people.list = people_list  # 手动设置人员列表
        self.people.tot = self.num_people
        
        # 更新密度图
        self.people.rmap = np.zeros((self.map.Length+2, self.map.Width+2))
        self.people.thmap = np.zeros((self.map.Length+2, self.map.Width+2))
        
        for person in self.people.list:
            x, y = int(person.pos[0]), int(person.pos[1])
            self.people.rmap[x][y] = 1
            self.people.thmap[x][y] = 1
        
        # 打印位置分布统计
        x_coords = [p.pos[0] for p in self.people.list]
        y_coords = [p.pos[1] for p in self.people.list]
        print(f"📊 人员位置分布:")
        print(f"   X坐标范围: {min(x_coords):.1f} - {max(x_coords):.1f}")
        print(f"   Y坐标范围: {min(y_coords):.1f} - {max(y_coords):.1f}")
        print(f"   总人数: {len(self.people.list)}")
    
    def run_simulation(self):
        """运行模拟并收集数据"""
        print(f"🏃 开始人员疏散分析...")
        print(f"👥 总人员数量: {self.num_people}")
        print(f"🎯 选择观察人员: {self.selected_people}")
        print(f"📍 人员分布范围: 0-15")
        print("="*50)
        
        # 随机选择要观察的人员
        all_person_ids = [p.id for p in self.people.list]
        self.selected_person_ids = random.sample(all_person_ids, 
                                               min(self.selected_people, len(all_person_ids)))
        
        print(f"🎲 随机选择的人员ID: {sorted(self.selected_person_ids)}")
        
        # 初始化数据存储
        for person_id in self.selected_person_ids:
            self.trajectories[person_id] = []
            self.health_data[person_id] = []
            self.evacuation_times[person_id] = None
            self.death_times[person_id] = None  # 添加死亡时间初始化
        
        step = 0
        max_steps = 200  # 设置为150步
        
        # 打印初始健康值
        initial_healths = [p.health for p in self.people.list if p.id in self.selected_person_ids]
        print(f"💊 初始健康值: 平均={np.mean(initial_healths):.1f}, 范围={min(initial_healths):.1f}-{max(initial_healths):.1f}")
        
        while step < max_steps:
            # 记录选定人员的数据
            self._record_step_data(step)
            
            # 人员移动和状态更新 - 修复：传入时间步长
            evacuated_count = self.people.run(time_step=0.5)
            
            step += 1
            
            # 检查疏散完成情况
            evacuated = sum(1 for p in self.people.list if p.savety)
            dead = sum(1 for p in self.people.list if p.dead)
            remaining = self.num_people - evacuated - dead
            
            # 移除火灾模型更新
            # if hasattr(self.map.fire_model, 'update'):
            #     self.map.fire_model.update()
            
            if step % 10 == 0:
                selected_alive = [p for p in self.people.list 
                                if p.id in self.selected_person_ids and not p.dead]
                if selected_alive:
                    avg_health = np.mean([p.health for p in selected_alive])
                    print(f"步骤 {step:3d}: 疏散 {evacuated:2d}人 | 死亡 {dead:2d}人 | 剩余 {remaining:2d}人 | 平均健康 {avg_health:.1f}")
                else:
                    print(f"步骤 {step:3d}: 疏散 {evacuated:2d}人 | 死亡 {dead:2d}人 | 剩余 {remaining:2d}人 | 选定人员已全部处理")
            
            # 记录整体模拟数据
            all_people = self.people.list
            if all_people:
                total_health = sum(p.health for p in all_people)
                avg_health = total_health / len(all_people)
            else:
                avg_health = 0
                
            self.simulation_data.append({
                'step': step,
                'evacuated': evacuated,
                'dead': dead,
                'remaining': remaining,
                'avg_health': avg_health
            })
            
            if remaining == 0:
                break
        
        print(f"\n✅ 模拟完成！总步数: {step}")
        print(f"📊 最终结果: 疏散 {evacuated}人, 死亡 {dead}人")
        
        # 打印最终健康值
        final_selected = [p for p in self.people.list if p.id in self.selected_person_ids]
        if final_selected:
            final_healths = [p.health for p in final_selected]
            print(f"💊 最终健康值: 平均={np.mean(final_healths):.1f}, 范围={min(final_healths):.1f}-{max(final_healths):.1f}")
        
        # 计算疏散时间
        self._calculate_evacuation_times()
        
        return step
    
    def _record_step_data(self, step):
        """记录每步的数据"""
        for person in self.people.list:
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
    
    def visualize_trajectories(self, save_path='robot_static_analysis_corrected.png'):
        """可视化人员轨迹"""
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
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 修正版轨迹分析图已保存: {save_path}")
        plt.show()
    
    def _plot_trajectories(self, ax):
        """绘制轨迹图"""
        # 设置地图范围
        ax.set_xlim(0, 36)
        ax.set_ylim(0, 30)
        ax.set_aspect('equal')
        
        # 绘制人员初始分布区域（0-15）
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
        
        # 绘制选定人员的轨迹
        colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(self.selected_person_ids)))
        
        for i, person_id in enumerate(self.selected_person_ids):
            trajectory = self.trajectories[person_id]
            if trajectory:
                x_coords = [point['pos'][0] for point in trajectory]
                y_coords = [point['pos'][1] for point in trajectory]
                
                # 绘制轨迹线
                ax.plot(x_coords, y_coords, color=colors[i], 
                       alpha=0.7, linewidth=2, label=f'Person {person_id}')
                
                # 标记起始点
                ax.scatter(x_coords[0], y_coords[0], color=colors[i], 
                          s=80, marker='.', edgecolor='black', linewidth=2)
                
                # 标记结束点
                if len(x_coords) > 1:
                    ax.scatter(x_coords[-1], y_coords[-1], color=colors[i], 
                              s=80, marker='s', edgecolor='black', linewidth=2)
        
        ax.set_title(f'人员轨迹图\n人员分布: 0-15范围', 
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
            for point in self.health_data[person_id]:
                all_steps.add(point['step'])
        
        all_steps = sorted(all_steps)
        avg_health = []
        
        for step in all_steps:
            step_healths = []
            for person_id in self.selected_person_ids:
                for point in self.health_data[person_id]:
                    if point['step'] == step:
                        step_healths.append(point['health'])
                        break
            if step_healths:
                # 修复：确保包含所有选定人员的健康值，死亡人员健康值为0
                avg_health.append(np.mean(step_healths))
            else:
                avg_health.append(100)
        
        # 绘制平均健康值曲线（粗线）
        ax.plot(all_steps, avg_health, color='red', linewidth=4, 
               label='平均健康值', alpha=0.8)
        
        # 绘制个体健康值曲线（细线）
        for i, person_id in enumerate(self.selected_person_ids):
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
        # 获取所有150人的疏散时间数据
        all_evacuation_times = []
        all_death_times = []
        
        # 遍历所有150人，计算疏散和死亡时间
        for person in self.people.list:
            person_id = person.id
            
            # 检查是否成功疏散
            if person.savety:
                # 需要从person的trajectory中找到疏散时间
                # 由于只有选定的20人有详细轨迹记录，对于其他人员需要估算
                if person_id in self.trajectories:
                    # 选定人员：从轨迹中找到确切的疏散时间
                    for point in self.trajectories[person_id]:
                        if point['savety']:
                            all_evacuation_times.append(point['step'])
                            break
                else:
                    # 非选定人员：根据模拟数据估算疏散时间
                    # 查找该人员最可能的疏散时间
                    for data_point in self.simulation_data:
                        if data_point['evacuated'] > 0:
                            # 简单估算：在疏散过程中的某个时间点疏散
                            estimated_time = data_point['step']
                            all_evacuation_times.append(estimated_time)
                            break
            
            # 检查是否死亡
            elif person.dead:
                if person_id in self.trajectories:
                    # 选定人员：从轨迹中找到确切的死亡时间
                    for point in self.trajectories[person_id]:
                        if point['dead']:
                            all_death_times.append(point['step'])
                            break
                else:
                    # 非选定人员：根据模拟数据估算死亡时间
                    for data_point in self.simulation_data:
                        if data_point['dead'] > 0:
                            estimated_time = data_point['step']
                            all_death_times.append(estimated_time)
                            break
        
        # 更精确的方法：基于模拟数据重新计算所有人的时间分布
        # 从simulation_data中提取更准确的时间分布
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
        not_evacuated = self.num_people - len(all_evacuation_times) - len(all_death_times)
        
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
        
        ax.set_title(f'全部150人时间分布\n成功疏散: {len(all_evacuation_times)}人, 死亡: {len(all_death_times)}人, 未疏散: {not_evacuated}人', 
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

def main():
    """主函数"""
    print("🏃 人员疏散分析测试")
    print("📍 人员分布范围: 0-15")
    print("="*50)
    
    # 创建分析器
    analyzer = CorrectedRobotStaticAnalysis(
        num_people=150,
        selected_people=20
    )
    
    # 运行模拟
    total_steps = analyzer.run_simulation()
    
    # 注释掉图像生成，改为数据输出
    # analyzer.visualize_trajectories('people_evacuation_analysis.png')
    
    # === 输出详细数据 ===
    print("\n" + "="*50)
    print("📊 详细数据分析")
    print("="*50)
    
    # 1. 输出每个观察人员的轨迹数据
    print("\n🛤️  人员轨迹数据:")
    for person_id in sorted(analyzer.selected_person_ids)[:5]:  # 只显示前5个人的详细轨迹
        trajectory = analyzer.trajectories[person_id]
        print(f"\n人员 {person_id} 轨迹:")
        print(f"  起始位置: ({trajectory[0]['pos'][0]:.2f}, {trajectory[0]['pos'][1]:.2f})")
        print(f"  最终位置: ({trajectory[-1]['pos'][0]:.2f}, {trajectory[-1]['pos'][1]:.2f})")
        print(f"  轨迹长度: {len(trajectory)} 步")
        print(f"  最终状态: {'已疏散' if trajectory[-1]['savety'] else '死亡' if trajectory[-1]['dead'] else '未完成'}")
        
        # 显示轨迹的前几步和最后几步
        print("  前5步轨迹:")
        for i, point in enumerate(trajectory[:5]):
            print(f"    步骤{point['step']:3d}: 位置({point['pos'][0]:6.2f}, {point['pos'][1]:6.2f}) 健康{point['health']:6.1f}")
        
        if len(trajectory) > 10:
            print("  ...")
            print("  最后5步轨迹:")
            for point in trajectory[-5:]:
                print(f"    步骤{point['step']:3d}: 位置({point['pos'][0]:6.2f}, {point['pos'][1]:6.2f}) 健康{point['health']:6.1f}")
    
    # 2. 输出健康值统计
    print(f"\n💊 健康值统计:")
    all_initial_health = []
    all_final_health = []
    for person_id in analyzer.selected_person_ids:
        health_data = analyzer.health_data[person_id]
        if health_data:
            all_initial_health.append(health_data[0]['health'])
            all_final_health.append(health_data[-1]['health'])
    
    print(f"  初始健康值: 平均={np.mean(all_initial_health):.1f}, 范围={min(all_initial_health):.1f}-{max(all_initial_health):.1f}")
    print(f"  最终健康值: 平均={np.mean(all_final_health):.1f}, 范围={min(all_final_health):.1f}-{max(all_final_health):.1f}")
    print(f"  健康值下降: 平均={np.mean(all_initial_health) - np.mean(all_final_health):.1f}")
    
    # 3. 输出疏散时间统计
    print(f"\n⏰ 疏散时间统计:")
    evacuation_times = [t for t in analyzer.evacuation_times.values() if t is not None]
    death_times = [t for t in analyzer.death_times.values() if t is not None]
    
    if evacuation_times:
        print(f"  成功疏散人数: {len(evacuation_times)}")
        print(f"  平均疏散时间: {np.mean(evacuation_times):.1f} 步")
        print(f"  疏散时间范围: {min(evacuation_times)}-{max(evacuation_times)} 步")
    
    if death_times:
        print(f"  死亡人数: {len(death_times)}")
        print(f"  平均死亡时间: {np.mean(death_times):.1f} 步")
        print(f"  死亡时间范围: {min(death_times)}-{max(death_times)} 步")
    
    # 4. 检查位置边界问题
    print(f"\n🗺️  位置边界检查:")
    out_of_bounds_count = 0
    max_x, max_y = 36, 30
    
    for person_id in analyzer.selected_person_ids:
        trajectory = analyzer.trajectories[person_id]
        for point in trajectory:
            x, y = point['pos']
            if x < 0 or x > max_x or y < 0 or y > max_y:
                out_of_bounds_count += 1
                if out_of_bounds_count <= 5:  # 只显示前5个越界位置
                    print(f"  ⚠️  人员{person_id} 在步骤{point['step']} 越界: ({x:.2f}, {y:.2f})")
    
    print(f"  总越界次数: {out_of_bounds_count}")
    
    # 5. 移动模式分析
    print(f"\n🚶 移动模式分析:")
    total_distance = 0
    straight_line_distance = 0
    
    for person_id in analyzer.selected_person_ids[:5]:  # 分析前5个人的移动模式
        trajectory = analyzer.trajectories[person_id]
        if len(trajectory) > 1:
            # 计算实际移动距离
            actual_distance = 0
            for i in range(1, len(trajectory)):
                prev_pos = trajectory[i-1]['pos']
                curr_pos = trajectory[i]['pos']
                dist = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
                actual_distance += dist
            
            # 计算直线距离
            start_pos = trajectory[0]['pos']
            end_pos = trajectory[-1]['pos']
            straight_dist = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
            
            efficiency = straight_dist / actual_distance if actual_distance > 0 else 0
            print(f"  人员{person_id}: 实际距离={actual_distance:.1f}, 直线距离={straight_dist:.1f}, 效率={efficiency:.3f}")
            
            total_distance += actual_distance
            straight_line_distance += straight_dist
    
    overall_efficiency = straight_line_distance / total_distance if total_distance > 0 else 0
    print(f"  整体移动效率: {overall_efficiency:.3f} (1.0为完全直线移动)")
    
    print("\n" + "="*50)
    print("✅ 数据分析完成！")
    print("🔧 主要发现:")
    print("  - ✅ 已输出详细轨迹数据")
    print("  - ✅ 已输出健康值变化统计")
    print("  - ✅ 已输出疏散时间分析")
    print("  - ✅ 已检查位置边界问题")
    print("  - ✅ 已分析移动模式效率")

if __name__ == "__main__":
    main() 