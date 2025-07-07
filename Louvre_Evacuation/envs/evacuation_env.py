# 迁移EvacuationEnv类，接口保持不变
# ...（此处省略，实际迁移您的EvacuationEnv实现） 

import numpy as np
from .map import Map, Init_Barrier
from .fire_model import FireSpreadModel, FireSource
from .people import People

# ============================================
# 疏散环境封装
# ============================================

class EvacuationEnv:
    """疏散环境封装"""
    # === 可调奖励系数（供自动调参脚本覆盖） ===
    EVAC_REWARD: float = 50.0         # 新疏散一人奖励
    # === 奖励权重重新平衡 ===
    # 将死亡惩罚大幅提高，确保智能体将"避免死亡"作为最高优先级；
    # 同时显著降低每步的存活激励，避免出现"什么都不做但每步拿到大量正奖励"的策略。
    DEATH_PENALTY: float = 500.0       # 新死亡惩罚/人（从200→500）
    DEATH_ACC_PENALTY: float = 5.0     # 已死亡持续惩罚/人·步（从0.5→5.0）
    ALIVE_BONUS: float = 0.1           # 存活激励/人·步（从1.0→0.1）

    def __init__(self, width=36, height=30, fire_zones=None, exit_location=None, num_people=150):
        self.width = width
        self.height = height
        self.num_people = num_people
        
        # 引入真实时间模拟参数
        self.time_per_step = 0.5  # 每个step代表0.5秒
        self.max_simulation_time = 600 # 最大模拟时间（秒），例如10分钟
        
        self.max_steps = int(self.max_simulation_time / self.time_per_step) # 根据时间计算最大步数
        
        # 修复出口位置设置
        if exit_location is None:
            exit_location = [36, 15]  # 修正出口
        if fire_zones is None:
            fire_zones = {(18, 14), (19, 14), (20, 14), (18, 15), (19, 15), (20, 15), (18, 16), (19, 16), (20, 16)}
        
        self.exit_location = exit_location
        self.fire_zones = fire_zones
        
        # 初始化地图 - 确保出口位置正确传递
        self.map = Map(width, height, [exit_location], 
                      [Init_Barrier(A=(18, 14), B=(20, 16))])
        
        # 优化火灾模型 - 缩小面积但提高危险系数
        self.fire_model = FireSpreadModel([
            FireSource(
                center=(19, 15),  # 障碍物中心
                size=(2, 2),      # 缩小火源面积 (从6x4减少到2x2)
                temp_max=900,     # 提高温度 (从600提高到900)
                co_max=1800       # 提高CO浓度 (从1200提高到1800)
            )
        ])
        
        # 状态空间定义
        self.state_size = (11, 11, 6)  # 11x11网格，6个特征通道
        self.action_size = 5  # 上、下、左、右、停留
        
        self.reset()

    def reset(self):
        """重置环境状态"""
        # 重置机器人位置到固定位置 [15, 15]
        self.map.robot_position = [15, 15]
        self.robot_direction = 1
        
        # 初始化人员（在左侧区域生成）
        self.people = People(self.num_people, self.map)
        
        self.time = 0
        self.current_step = 0
        self.prev_evacuated = 0
        self.prev_dead = 0
        
        # 重置轨迹记录
        self.robot_trajectory = [(tuple(self.map.robot_position), 0)]
        
        # 记录人员初始轨迹
        for p in self.people.list:
            p.trajectory = [{'pos': p.pos, 'step': 0}]
        
        return self._get_state()

    def _get_state(self):
        """获取状态表示（11x11x6）"""
        robot_x, robot_y = self.map.robot_position
        state = np.zeros(self.state_size)
        
        # 遍历11x11网格区域（以机器人为中心）
        for i in range(11):
            for j in range(11):
                map_x = robot_x + (i - 5)
                map_y = robot_y + (j - 5)
                
                # 特征0: 静态场势能（归一化）
                if self.map.Check_Valid(map_x, map_y):
                    max_potential = np.max(self.map.space)
                    if max_potential > 0:
                        state[i, j, 0] = self.map.space[map_x][map_y] / max_potential
                
                # 特征1: 人员密度
                if self.map.Check_Valid(map_x, map_y):
                    state[i, j, 1] = self.people.rmap[map_x][map_y]
                
                # 特征2: 火灾危险度（归一化）
                state[i, j, 2] = self.fire_model.get_max_danger((map_x, map_y))
                
                # 特征3: 障碍物/边界
                if not self.map.Check_Valid(map_x, map_y) or (map_x, map_y) in self.map.barrier_list:
                    state[i, j, 3] = 1.0
                
                # 特征4: 出口位置
                if (map_x, map_y) == tuple(self.exit_location):
                    state[i, j, 4] = 1.0
                
                # 特征5: 机器人位置（中心标记）
                if i == 5 and j == 5:
                    state[i, j, 5] = 1.0
        
        return state

    def step(self, action):
        """执行一个时间步"""
        # 机器人移动
        self.map.move_robot(action, self.people.list)
        
        # 记录机器人轨迹
        self.robot_trajectory.append((tuple(self.map.robot_position), self.current_step))

        # 人群移动 - 传入时间步长
        evac_count = self.people.run(self.time_per_step)
        
        # 记录人员轨迹
        for p in self.people.list:
            p.trajectory.append({'pos': p.pos, 'step': self.current_step})
        
        # 更新火灾模型（渐进式火势蔓延）
        if hasattr(self.fire_model, 'update'):
            self.fire_model.update()
        # 同步地图中的火灾模型（人员查询的是 map.fire_model）
        if hasattr(self.map, 'fire_model') and hasattr(self.map.fire_model, 'update'):
            self.map.fire_model.update()
        
        # 计算奖励
        reward = self._calculate_reward()
        
        self.time += self.time_per_step
        self.current_step += 1
        
        # 更新计数
        current_evacuated = sum(1 for p in self.people.list if p.savety)
        current_dead = sum(1 for p in self.people.list if p.dead)
        
        # 结束条件 - 增加真实时间判断
        all_accounted_for = (current_evacuated + current_dead == self.num_people)
        time_is_up = (self.time >= self.max_simulation_time)
        done = all_accounted_for or time_is_up
        
        # 创建info字典
        info = {
            'robot_position': tuple(self.map.robot_position),
            'people_positions': [p.pos for p in self.people.list],
            'health_values': [p.health for p in self.people.list],
            'evacuation_status': [p.savety for p in self.people.list],
            'fire_spread': [],  # 可以添加火灾扩散信息
            'evacuation_rate': current_evacuated / self.num_people,
            'death_rate': current_dead / self.num_people,
            'current_step': self.current_step,
            'simulation_time': self.time # 添加模拟时间
        }
        
        return self._get_state(), reward, done, info

    def _calculate_reward(self):
        """优化的奖励函数 - 鼓励智能机器人行为"""
        reward = 0
        robot_pos = np.array(self.map.robot_position)
        
        # 当前状态统计
        current_evacuated = sum(1 for p in self.people.list if p.savety)
        current_dead = sum(1 for p in self.people.list if p.dead)
        remaining_people = self.num_people - current_evacuated - current_dead
        
        # === 1. 疏散进度奖励 ===
        # 新疏散人员奖励（相比上一步）
        if not hasattr(self, 'prev_evacuated'):
            self.prev_evacuated = 0
        new_evacuations = current_evacuated - self.prev_evacuated
        reward += new_evacuations * EvacuationEnv.EVAC_REWARD  # 每新疏散一人奖励
        
        # === 2. 机器人引导效果奖励 ===
        people_influenced = 0
        guidance_quality = 0
        exit_pos = np.array(self.exit_location)
        
        for p in self.people.list:
            if not p.savety and not p.dead:  # 只考虑未疏散且存活的人员
                person_pos = np.array(p.pos)
                dist_to_robot = np.linalg.norm(person_pos - robot_pos)
                
                # 机器人影响范围内的人员
                if dist_to_robot <= 5:  # 影响范围5单位
                    people_influenced += 1
        
                    # 根据人员到出口的距离评估引导质量
                    dist_to_exit = np.linalg.norm(person_pos - exit_pos)
                    if dist_to_exit > 20:  # 远离出口的人员更需要引导
                        guidance_quality += 2.0
                    elif dist_to_exit > 10:
                        guidance_quality += 1.5
                    else:
                        guidance_quality += 1.0
                    
                    # 健康状况差的人员更需要帮助
                    if p.health < 80:
                        guidance_quality += 1.0
                    elif p.health < 60:
                        guidance_quality += 2.0
        
        reward += guidance_quality  # 引导质量奖励
        
        # === 3. 机器人位置优化奖励 ===
        if remaining_people > 0:
            # 计算机器人到未疏散人员的平均距离
            remaining_positions = [np.array(p.pos) for p in self.people.list 
                                 if not p.savety and not p.dead]
            if remaining_positions:
                avg_dist_to_people = np.mean([np.linalg.norm(robot_pos - pos) 
                                            for pos in remaining_positions])
                # 距离人群适中时给予奖励（不要太远也不要太近）
                optimal_distance = 8.0  # 最优距离
                distance_reward = max(0, 2.0 - abs(avg_dist_to_people - optimal_distance) * 0.2)
                reward += distance_reward
        
        # === 4. 动态时间惩罚 ===
        # 根据剩余人员数量调整时间惩罚
        if remaining_people > 0:
            urgency_factor = remaining_people / self.num_people
            # 加强时间惩罚：基准 -0.2，线性系数 0.3
            time_penalty = -0.2 - (urgency_factor * 0.3)  # 剩余人员越多惩罚越重
            reward += time_penalty
        else:
            reward -= 0.02  # 全部疏散后的轻微时间惩罚
        
        # === 5. 健康保护奖励 ===
        total_health = sum(p.health for p in self.people.list if not p.dead)
        if self.num_people - current_dead > 0:
            avg_health = total_health / (self.num_people - current_dead)
            # 提高健康奖励系数，鼓励保持更高健康水平
            health_bonus = (avg_health - 90) * 0.05  # 每超过 90 分 1 分，奖励 0.05
            reward += health_bonus
        
        # === 6. 疏散完成大奖励 ===
        if current_evacuated == self.num_people:
            # 基础完成奖励
            completion_reward = 100
            # 时间奖励（越快完成奖励越高）
            time_bonus = max(0, 300 - self.current_step) * 0.2
            # 健康奖励（平均健康越高奖励越高）
            if self.num_people > 0:
                final_avg_health = sum(p.health for p in self.people.list) / self.num_people
                # 强化完成时的健康奖励
                health_bonus = (final_avg_health - 80) * 1.0
                reward += completion_reward + time_bonus + health_bonus
            else:
                reward += completion_reward + time_bonus
        
        # === 7. 死亡严重惩罚（加强） ===
        new_deaths = current_dead - getattr(self, 'prev_dead', 0)
        reward -= new_deaths * EvacuationEnv.DEATH_PENALTY
        # 对当前死亡人数持续惩罚，防止一次性惩罚后不再关心
        reward -= current_dead * EvacuationEnv.DEATH_ACC_PENALTY
        
        # === 7b. 存活激励 ===
        survivors = self.num_people - current_dead
        reward += survivors * EvacuationEnv.ALIVE_BONUS  # 每存活一人激励
        
        # === 8. 效率奖励 ===
        # 奖励在合理时间内的高效疏散
        if self.current_step > 0:
            efficiency = current_evacuated / self.current_step
            if efficiency > 0.1:  # 效率阈值
                reward += efficiency * 5
        
        # 更新状态记录
        self.prev_evacuated = current_evacuated
        self.prev_dead = current_dead
        
        return reward

    def get_performance_metrics(self):
        """获取性能指标"""
        evacuated = sum(1 for p in self.people.list if p.savety)
        dead = sum(1 for p in self.people.list if p.dead)
        remaining = self.num_people - evacuated - dead
        
        avg_health = np.mean([p.health for p in self.people.list if not p.dead])
        min_health = min([p.health for p in self.people.list if not p.dead], default=100)
        
        return {
            'evacuated': evacuated,
            'dead': dead,
            'remaining': remaining,
            'evacuation_rate': evacuated / self.num_people,
            'death_rate': dead / self.num_people,
            'avg_health': avg_health,
            'min_health': min_health,
            'total_steps': self.current_step,
            'total_time': self.time
        } 
    pass 