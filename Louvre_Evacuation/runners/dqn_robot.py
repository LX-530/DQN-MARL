#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器人静态位置分析测试 - 修正版
直接导入map.py和people.py，人员分布在0-7.5范围内
"""
# 避免 OpenMP 多次初始化警告（Intel OpenMP 与 MKL 冲突）
import os as _os
_os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import pandas as pd
import torch
import yaml



# 添加项目路径 (必须在导入项目内模块之前)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.dqn_agent import DQNAgent  # 新增

# 直接导入map和people模块
from envs.map import Map, Init_Exit, Init_Barrier
from envs.people import People, Person

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# === 读取统一配置 ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
config_path = os.path.join(project_root, 'configs', 'dqn.yaml')
with open(config_path, 'r', encoding='utf-8') as _cf:
    _cfg = yaml.safe_load(_cf)
env_cfg = _cfg.get('env', {})

class CorrectedRobotStaticAnalysis:
    """修正版机器人静态位置分析类"""
    
    # def __init__(self, robot_position=(15, 15), num_people=150, selected_people=20):
    def __init__(self, num_people=None, selected_people=20, model_path='../../dqn_results/best_model.pth'):
        """
        初始化分析器
        
        Args:
            num_people: 总人员数量
            selected_people: 选择观察的人员数量
            model_path: DQN模型路径
        """
        # 若调用方未显式传人数，则使用配置人数
        self.num_people = num_people if num_people is not None else env_cfg.get('num_people', 150)
        self.selected_people = selected_people
        
        # 地图尺寸与出口、火源障碍与训练环境一致
        width = env_cfg.get('width', 36)
        height = env_cfg.get('height', 30)
        exit_loc = tuple(env_cfg.get('exit_location', [36, 15]))
        self.map = Map(
            L=width,
            W=height,
            E=Init_Exit(exit_loc),
            B=[]  # 使用内建的凸字形走廊布局
        )
        
        # 数据存储
        self.selected_person_ids = []
        self.trajectories = {}
        self.health_data = {}
        self.evacuation_times = {}
        self.death_times = {}  # 添加死亡时间记录
        self.simulation_data = []
        # 新增: 机器人轨迹记录
        self.robot_trajectory = []
        
        # === 人员初始化：与训练环境保持一致 ===
        # 直接使用 People 类的内部分布逻辑，避免手动偏移导致结果不一致
        self.people = People(self.num_people, self.map)
        print(f"✅ 已创建 {self.num_people} 名人员 (内部随机/风险分布)")
        
        # 打印简单分布统计
        xs = [p.pos[0] for p in self.people.list]
        ys = [p.pos[1] for p in self.people.list]
        print(f"📊 人员位置分布 X 范围: {min(xs):.1f}~{max(xs):.1f} | Y 范围: {min(ys):.1f}~{max(ys):.1f}")
        
        # === 新增 DQN 智能体加载 ===
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dummy_config = {
            'gamma': 0.99,
            'epsilon': 0.0,  # 推理不探索
            'epsilon_min': 0.0,
            'epsilon_decay': 1.0,
            'learning_rate': 1e-4,
            'batch_size': 32,
            'target_update_freq': 200,
            'warmup_steps': 1000,
            'memory_size': 10000
        }
        self.agent = DQNAgent(state_size=(11,11,6), action_size=5, device=self.device, config=dummy_config)
        
        # 修正模型路径
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        model_path = os.path.join(project_root, 'CA-dqn1', 'dqn_results', 'best_model.pth')
        
        try:
            self.agent.load(model_path)
            print(f'✅ 已加载DQN模型: {model_path}')
        except Exception as e:
            print(f'⚠️ 无法加载模型 {model_path}: {e}\n   将使用随机策略。')
    
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
        
        # 记录机器人初始位置
        self.robot_trajectory.append(tuple(self.map.robot_position))
        
        step = 0
        max_steps = 200  # 设置为150步
        
        # 打印初始健康值
        initial_healths = [p.health for p in self.people.list if p.id in self.selected_person_ids]
        print(f"💊 初始健康值: 平均={np.mean(initial_healths):.1f}, 范围={min(initial_healths):.1f}-{max(initial_healths):.1f}")
        
        while step < max_steps:
            # 移动机器人并记录轨迹
            state = self._get_state()  # 计算当前状态
            try:
                action = self.agent.act(state, training=False)
            except Exception:
                action = random.randint(0,4)
            self.map.move_robot(action)
            self.robot_trajectory.append(tuple(self.map.robot_position))
            
            # 更新阶段 (0~1)，影响人员排斥系数
            self.map.current_phase = step / max_steps  # type: ignore
            
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
                # 计算所有人的健康值（死亡人员健康值为0）
                total_health = sum(p.health for p in all_people)
                avg_health = total_health / len(all_people)  # 除以总人数，而不是只除以存活人数
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
                
                
        
        ax.set_title(f'人员轨迹图\n人员分布: 0-15范围', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # 绘制机器人轨迹 (应放在人员轨迹之后，确保可见)
        if self.robot_trajectory:
            r_x = [pos[0] for pos in self.robot_trajectory]
            r_y = [pos[1] for pos in self.robot_trajectory]
            ax.plot(r_x, r_y, color='black', linestyle='--', linewidth=3, label='Robot')
            ax.scatter(r_x[0], r_y[0], color='black', s=120, marker='*', edgecolor='yellow', linewidth=2)
            ax.scatter(r_x[-1], r_y[-1], color='black', s=120, marker='X', edgecolor='yellow', linewidth=2)
            # 更新图例包含机器人
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

    def _get_state(self):
        """构造与训练时一致的 11×11×6 状态张量"""
        robot_x, robot_y = self.map.robot_position
        state = np.zeros((11,11,6), dtype=np.float32)
        max_potential = np.max(self.map.space) if np.max(self.map.space)>0 else 1.0
        exit_pos = (36,15)
        for i in range(11):
            for j in range(11):
                map_x = robot_x + (i - 5)
                map_y = robot_y + (j - 5)
                # 特征0: 势能
                if self.map.Check_Valid(map_x, map_y):
                    state[i,j,0] = self.map.space[int(map_x)][int(map_y)] / max_potential
                # 特征1: 人员密度
                if self.map.Check_Valid(map_x, map_y):
                    state[i,j,1] = self.people.rmap[int(map_x)][int(map_y)]
                # 特征2: 火灾危险度（移除火源模型后设为0）
                state[i,j,2] = 0.0  # 简化：无火源危险度
                # 特征3: 障碍物/边界
                if (not self.map.Check_Valid(map_x, map_y)) or ((map_x,map_y) in self.map.barrier_list):
                    state[i,j,3] = 1.0
                # 特征4: 出口
                if (map_x, map_y) == exit_pos:
                    state[i,j,4] = 1.0
                # 特征5: 机器人
                if i==5 and j==5:
                    state[i,j,5] = 1.0
        return state

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
    print("📊 DQN机器人疏散效果分析")
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
    
    # 2. 机器人轨迹分析
    print(f"\n🤖 机器人轨迹分析:")
    if analyzer.robot_trajectory:
        print(f"  起始位置: {analyzer.robot_trajectory[0]}")
        print(f"  最终位置: {analyzer.robot_trajectory[-1]}")
        print(f"  移动步数: {len(analyzer.robot_trajectory)-1}")
        
        # 计算机器人移动距离
        total_distance = 0
        for i in range(1, len(analyzer.robot_trajectory)):
            prev_pos = analyzer.robot_trajectory[i-1]
            curr_pos = analyzer.robot_trajectory[i]
            dist = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
            total_distance += dist
        print(f"  总移动距离: {total_distance:.2f}")
        print(f"  平均每步移动: {total_distance/(len(analyzer.robot_trajectory)-1):.3f}")
    
    # 3. 输出健康值统计
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
    
    # 4. 输出疏散时间统计
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
    
    # 5. 整体疏散效果对比
    print(f"\n📈 整体疏散效果:")
    final_data = analyzer.simulation_data[-1] if analyzer.simulation_data else {}
    total_evacuated = final_data.get('evacuated', 0)
    total_dead = final_data.get('dead', 0)
    total_remaining = final_data.get('remaining', 0)
    
    print(f"  ✅ 成功疏散: {total_evacuated}人 ({total_evacuated/analyzer.num_people:.1%})")
    print(f"  ❌ 死亡人数: {total_dead}人 ({total_dead/analyzer.num_people:.1%})")
    print(f"  ⏳ 未完成疏散: {total_remaining}人 ({total_remaining/analyzer.num_people:.1%})")
    print(f"  ⏱️ 总疏散时间: {total_steps} 步")
    
    print("\n" + "="*50)
    print("✅ DQN机器人疏散效果分析完成！")
    print("🔧 主要发现:")
    print("  - ✅ 已输出详细轨迹数据")
    print("  - ✅ 已输出机器人行为分析")
    print("  - ✅ 已输出健康值变化统计")
    print("  - ✅ 已输出疏散时间分析")
    print("  - ✅ 已输出整体疏散效果对比")
    

if __name__ == "__main__":
    main() 