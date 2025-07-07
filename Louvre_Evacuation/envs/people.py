import numpy as np
import random
from .map import MoveTO
import time
import math


class Person:
	
	# 速度常量
	NORMAL_SPEED = 1.0   # m/s，整体行进速度降低
	CRAWLING_SPEED = 0.4  # m/s，当健康值极低时的爬行速度
 
	def __init__(self, id, pos_x, pos_y):
		self.id = id
		self.pos = (pos_x, pos_y)
		self.speed = Person.NORMAL_SPEED
		self.savety = False
		self.health = 100.0  # 健康值
		self.dead = False    # 死亡状态
		self.trajectory = [] # 轨迹记录
		self.move_accumulator = 0.0 # 移动距离累积器
		self.record_position()  # 记录初始位置
  
	def update_state(self, robot_pos, danger_level):
		"""
		更新人员的健康和速度状态。
		- 健康值低于20时进入爬行状态。
		- 不再受机器人距离影响而加速，机器人改为纯粹的排斥引导。
		"""
		# 1. 更新健康
		self.update_health(danger_level)
		if self.dead:
			return

		# 2. 根据健康更新速度
		# 健康越差速度越慢，并在极低健康值时跌至爬行速度
		if self.health < 20:
			# 极低健康值：只能以爬行速度移动
			self.speed = Person.CRAWLING_SPEED
		else:
			# 健康 20~100 线性映射到 0.3~1.0 的速度系数
			speed_factor = 0.3 + 0.7 * (self.health / 100.0)
			self.speed = Person.NORMAL_SPEED * speed_factor

	def name(self):
		return "ID_"+str(self.id)

	def __str__(self):
		return  self.name() + " (%d, %d)" % (self.pos[0], self.pos[1])
	
	def record_position(self):
		"""记录当前位置和时间步"""
		self.trajectory.append({
			'pos': self.pos,
			'health': self.health,
			'savety': self.savety,
			'dead': self.dead
		})
	
	def update_health(self, danger_level):
		"""根据危险度更新健康值 - 降低损失版"""
		if danger_level <= 0:
			# 安全区域：健康值保持不变，不允许恢复
			return
		
		# 提高健康损失系数，增加死亡概率
		if danger_level >= 0.8:
			health_loss = danger_level * 50.0 + np.random.uniform(1.0, 3.0)
		elif danger_level >= 0.5:
			health_loss = danger_level * 40.0 + np.random.uniform(0.8, 2.0)
		elif danger_level >= 0.2:
			health_loss = danger_level * 30.0 + np.random.uniform(0.5, 1.5)
		else:
			health_loss = danger_level * 20.0 + np.random.uniform(0.2, 1.0)
		if self.health < 50:
			health_loss *= 1.2  # 从1.5降低到1.2
		elif self.health < 25:
			health_loss *= 1.4  # 从2.0降低到1.4
		self.health -= health_loss
		# 健康值不为负
		if self.health <= 0:
			self.health = 0
			self.dead = True
		elif self.health <= 8.0:  # 健康值为2.0死亡阈值
			self.dead = True
		# 健康值只能减少，不能增加
		self.health = max(0, min(self.health, 100))


class People:
    
    # === 可调机器人排斥参数 ===
    ROBOT_REPEL_K: float = -20.0   # 进一步减弱排斥强度
    ROBOT_REPEL_RANGE: float = 5.0 # 减小作用半径
    
    def create_risky_initial_positions(self, cnt):
        """创建包含风险的初始位置分布"""
        positions = []
        
        # 30%的人员放在相对安全的区域
        safe_count = int(cnt * 0.3)
        safe_areas = [
            (5, 25), (8, 24), (12, 22), (15, 20), (10, 28),
            (6, 26), (9, 25), (13, 23), (16, 21), (11, 27)
        ]
        
        for i in range(safe_count):
            if i < len(safe_areas):
                pos = safe_areas[i]
            else:
                # 随机安全位置
                pos = (
                    np.random.randint(3, 15),
                    np.random.randint(20, 29)
                )
            positions.append(pos)
        
        # 40%的人员放在中等风险区域
        medium_count = int(cnt * 0.4)
        medium_areas = [
            (17, 18), (16, 16), (18, 14), (20, 12), (22, 14),
            (15, 17), (19, 13), (21, 15), (23, 17), (14, 15),
            (24, 16), (13, 14), (25, 18), (12, 13), (26, 19)
        ]
        
        for i in range(medium_count):
            if i < len(medium_areas):
                pos = medium_areas[i]
            else:
                # 随机中风险位置
                pos = (
                    np.random.randint(12, 27),
                    np.random.randint(10, 20)
                )
            positions.append(pos)
        
        # 30%的人员放在高风险区域（靠近火源）
        high_count = cnt - safe_count - medium_count
        high_risk_areas = [
            (18, 16), (20, 14), (17, 14), (21, 16), (19, 17),
            (22, 15), (16, 15), (23, 16), (18, 13), (20, 17)
        ]
        
        for i in range(high_count):
            if i < len(high_risk_areas):
                pos = high_risk_areas[i]
            else:
                # 随机高风险位置（火源附近）
                pos = (
                    np.random.randint(16, 24),
                    np.random.randint(12, 18)
                )
            positions.append(pos)
        
        return positions[:cnt]

    def __init__(self, cnt, myMap):
        self.list = []
        self.tot = cnt
        self.map = myMap
        self.rmap = np.zeros((myMap.Length+2, myMap.Width+2))  # 密度图
        self.thmap = np.zeros((myMap.Length+2, myMap.Width+2)) # 热力图
        
        # 修复后的初始化方法
        if hasattr(myMap, 'create_risky_initial_positions'):
            # 使用新的风险位置生成方法
            positions = myMap.create_risky_initial_positions(cnt)
            for i in range(cnt):
                if i < len(positions):
                    pos_x, pos_y = positions[i]
                else:
                    # 备用随机位置生成
                    pos_x = random.randint(1, myMap.Length-2)  # 修复：使用 Length 而不是 width
                    pos_y = random.randint(1, myMap.Width-2)   # 修复：使用 Width 而不是 height
                    while not myMap.Check_Valid(pos_x, pos_y):
                        pos_x = random.randint(1, myMap.Length-2)
                        pos_y = random.randint(1, myMap.Width-2)
                person = Person(i+1, pos_x+0.5, pos_y+0.5)
                self.list.append(person)
                self.rmap[pos_x][pos_y] = 1
                self.thmap[pos_x][pos_y] = 1
        else:
            # 原始初始化方法
            for i in range(cnt):
                pos_x = random.randint(1, myMap.Length-2)  # 修复：使用 Length 而不是 width
                pos_y = random.randint(1, myMap.Width-2)   # 修复：使用 Width 而不是 height
                while not myMap.Check_Valid(pos_x, pos_y):
                    pos_x = random.randint(1, myMap.Length-2)
                    pos_y = random.randint(1, myMap.Width-2)
                person = Person(i+1, pos_x+0.5, pos_y+0.5)
                self.list.append(person)
                self.rmap[pos_x][pos_y] = 1
                self.thmap[pos_x][pos_y] = 1

    def run(self, time_step):
        """
        基于真实时间的模拟运行。
        - 使用固定的时间步长 (time_step)
        - 人员根据速度累积移动距离，决定是否移动
        """
        # 1. 更新所有人员的状态（健康和速度）
        for p in self.list:
            if not p.savety and not p.dead:
                danger = self.map.get_fire_danger(p.pos)
                robot_pos = getattr(self.map, 'robot_position', None)
                p.update_state(robot_pos, danger)

        # 2. 为可以移动的人员规划路径
        move_plan = {}
        for p in self.list:
            if p.savety or p.dead:
                continue
            
            # 累积移动距离
            p.move_accumulator += p.speed * time_step
            
            # 如果累积距离足够移动一格
            if p.move_accumulator >= 1.0:
                p.move_accumulator -= 1.0 # 消耗掉移动一格的距离
                
                x, y = int(p.pos[0]), int(p.pos[1])
                best_dir = self.find_best_direction(x, y)
                
                if best_dir is not None:
                    dx, dy = MoveTO[best_dir]
                    new_x, new_y = x + dx, y + dy
                    if (new_x, new_y) not in move_plan:
                        move_plan[(new_x, new_y)] = []
                    move_plan[(new_x, new_y)].append((p, x, y))

        # 3. 如果没人移动，直接返回
        if not move_plan:
            cnt = sum(1 for p in self.list if p.savety)
            return cnt

        # 4. 执行移动并解决冲突
        for target, movers in move_plan.items():
            random.shuffle(movers)
            if len(movers) == 1:
                p, old_x, old_y = movers[0]
                self.execute_move(p, old_x, old_y, *target)
            else:
                # 有冲突，只移动一人
                p, old_x, old_y = movers[0]
                self.execute_move(p, old_x, old_y, *target)
                # 其他人留在原地，增加热力图值
                for p_stay, x_stay, y_stay in movers[1:]:
                    self.thmap[x_stay][y_stay] += 1
        
        # 5. 在所有移动结束后，统计最终疏散人数
        cnt = sum(1 for p in self.list if p.savety)
        return cnt

    def find_best_direction(self, x, y):
        """
        "短视"寻路决策:
        - 主要目标是沿着势能场最快下降的方向走 (趋向出口)。
        - 会被机器人"排斥"，以避开机器人正在示警的危险区域。
        """
        best_dir = None
        max_score = -float('inf')
        
        for dire in range(8):  # 检查8个方向
            dx, dy = MoveTO[dire]
            nx, ny = x + dx, y + dy
            
            if self.map.Check_Valid(nx, ny) and self.rmap[nx][ny] == 0:
                # 基础势能差（核心驱动力，让人走向出口）
                delta_p = self.map.getDeltaP((x,y), (nx,ny))
                
                # 机器人排斥效应（多机器人距离加权：取最近一台机器人）
                robot_effect = 0
                # 支持 robot_positions 列表
                if hasattr(self.map, 'robot_positions') and self.map.robot_positions:
                    dist_to_robot = min(np.linalg.norm(np.array([nx, ny]) - np.array(rp)) for rp in self.map.robot_positions)
                elif hasattr(self.map, 'robot_position'):
                    dist_to_robot = np.linalg.norm(np.array([nx, ny]) - np.array(self.map.robot_position))
                else:
                    dist_to_robot = None

                if dist_to_robot is not None and dist_to_robot < People.ROBOT_REPEL_RANGE:
                    # 距离越近排斥越强；距离加权 1/d ，系数可调
                    robot_effect = People.ROBOT_REPEL_K / (dist_to_robot + 0.1)

                # 综合评分: 强烈的趋向出口 + 机器人排斥
                score = (
                    delta_p * 5.0 +       # 极大地增强了走向出口的意愿
                    robot_effect +        # 机器人排斥力
                    random.uniform(-0.1, 0.1) # 微小的随机扰动，避免卡死
                )
                
                if score > max_score:
                    max_score = score
                    best_dir = dire
        
        return best_dir
    
    def execute_move(self, p, old_x, old_y, new_x, new_y):
        # 更新位置记录
        self.rmap[old_x][old_y] = 0
        self.rmap[new_x][new_y] = 1
        p.pos = (new_x + 0.5, new_y + 0.5)  # 更新为格子中心坐标
        
        # 记录轨迹
        p.record_position()
        
        # 更新热力图
        self.thmap[new_x][new_y] += 1
        
        # 检查是否到达出口
        if self.map.checkSavefy(p.pos):
            p.savety = True
            self.rmap[new_x][new_y] = 0



# Total_People = 2
# P = People(Total_People, myMap)


# Eva_Number = 0
# while Eva_Number<Total_People:
# 	Eva_Number = P.run()

	# time.sleep(0.5)