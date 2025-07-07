import numpy as np
from queue import Queue
import random
from .fire_model import FireSource, FireSpreadModel
import heapq

Direction = {
    "RIGHT": 0, "UP": 1, "LEFT": 2, "DOWN": 3, "NONE": -1
}

MoveTO = []
MoveTO.append(np.array([1, 0]))     # RIGHT
MoveTO.append(np.array([0, -1]))    # UP
MoveTO.append(np.array([-1, 0]))    # LEFT
MoveTO.append(np.array([0, 1]))     # DOWN
MoveTO.append(np.array([1, -1]))    
MoveTO.append(np.array([-1, -1]))   
MoveTO.append(np.array([-1, 1]))    
MoveTO.append(np.array([1, 1]))     


def Init_Exit(P1):
    return [P1]  # 直接返回单点出口

def Init_Barrier(A, B):
    if A[0] > B[0]:
        A, B = B, A
    x1, y1 = A[0], A[1]
    x2, y2 = B[0], B[1]
    if y1 < y2:
        return ((x1, y1), (x2, y2))
    else:
        return ((x1, y2), (x2, y1))

Outer_Size = 1

class Map:
    def __init__(self, L, W, E, B):
        self.Length = L
        self.Width = W
        self.Exit = E
        self.Barrier = B
        self.barrier_list = []
        self.space = np.zeros((self.Length+Outer_Size*2, self.Width+Outer_Size*2))
        for j in range(0, self.Width+Outer_Size*2):
            self.space[0][j] = self.space[L+1][j] = float("inf")
            self.barrier_list.append((0, j))
            self.barrier_list.append((L+1, j))
        for i in range(0, self.Length+Outer_Size*2):
            self.space[i][0] = self.space[i][W+1] = float("inf")
            self.barrier_list.append((i, 0))
            self.barrier_list.append((i, W+1))
        for (A, B) in self.Barrier:
            for i in range(A[0], B[0]+1):
                for j in range(A[1], B[1]+1):
                    self.space[i][j] = float("inf")
                    self.barrier_list.append((i, j))
        self.fire_model = FireSpreadModel([
            FireSource(
                center=((A[0]+B[0])/2, (A[1]+B[1])/2),
                size=(2, 2),  # 缩小火源面积，统一设为2x2
                temp_max=900,  # 提高温度
                co_max=1800    # 提高CO浓度
            ) for (A,B) in self.Barrier
        ])
        (ex, ey) = self.Exit[0]
        self.space[ex][ey] = 1
        if ex == self.Length:
            self.space[ex+1][ey] = 1
        if ey == self.Width:
            self.space[ex][ey+1] = 1
        if (ex, ey) in self.barrier_list:
            self.barrier_list.remove((ex, ey))
        # 默认单机器人仍然放在 (15,15)
        self.robot_range = (15, 30)
        self.robot_position = [15, 15]
        # 新增：支持多机器人
        self.robot_positions = [self.robot_position.copy()]  # 列表形式，保持向后兼容
        self.Init_Potential()
    def print(self, mat):
        for line in mat:
            for v in line:
                print(v, end=' ')
            print("")
    def Check_Valid(self, x, y):
        x, y = int(x), int(y)
        if x >= self.Length+1 or x <= 0 or y >= self.Width+1 or y <= 0:
            return False
        if self.space[x][y] == float("inf"):
            return False
        else:
            return True
    def checkSavefy(self, pos):
        x, y = int(pos[0]), int(pos[1])
        
        # 修复边界处理逻辑
        # 如果人员到达地图边界的出口位置，认为已安全疏散
        if x >= self.Length+1:  # 右边界
            x = self.Length+1
        elif x <= 0:  # 左边界
            x = 0
        
        if y >= self.Width+1:  # 上边界
            y = self.Width+1
        elif y <= 0:  # 下边界
            y = 0
        
        # 检查是否到达出口位置
        for exit_pos in self.Exit:
            # 出口通常在边界上，允许一定的容差
            if abs(x - exit_pos[0]) <= 1 and abs(y - exit_pos[1]) <= 1:
                return True
        return False
    def _init_fire_sources(self):
        self.fire_model = FireSpreadModel([
            FireSource(
                center=((A[0]+B[0])/2, (A[1]+B[1])/2),
                size=(2, 2),  # 缩小火源面积，统一设为2x2
                temp_max=900,  # 提高温度
                co_max=1800    # 提高CO浓度
            ) for (A,B) in self.Barrier
        ])
    def get_fire_danger(self, pos):
        if hasattr(self, 'fire_model'):
            return self.fire_model.get_max_danger(pos)
        return 0.0
    def Init_Potential(self):
        minDis = np.full((self.Length+2, self.Width+2), float('inf'))
        heap = []
        for (ex, ey) in self.Exit:
            minDis[ex][ey] = 1
            heapq.heappush(heap, (1, ex, ey))
        while heap:
            current_dist, x, y = heapq.heappop(heap)
            for i, move in enumerate(MoveTO):
                nx, ny = x + move[0], y + move[1]
                cost = 1.0 if i < 4 else 1.4
                if self.Check_Valid(nx, ny):
                    new_dist = current_dist + cost
                    if new_dist < minDis[nx][ny]:
                        minDis[nx][ny] = new_dist
                        heapq.heappush(heap, (new_dist, nx, ny))
        for i in range(self.Length+2):
            for j in range(self.Width+2):
                if minDis[i][j] != float('inf'):
                    danger = self.get_fire_danger((i,j))
                    minDis[i][j] += 200 * (danger ** 2)
        self.space = minDis
    def getDeltaP(self, P1, P2):
        x1, y1 = int(P1[0]), int(P1[1])
        x2, y2 = int(P2[0]), int(P2[1])
        return self.space[x1][y1] - self.space[x2][y2]
    def Random_Valid_Point(self):
        x = random.uniform(1, self.Length+2)
        y = random.uniform(1, self.Width+2)
        while not self.Check_Valid(x, y):
            x = random.uniform(1, self.Length+2)
            y = random.uniform(1, self.Width+2)
        return x, y
    def move_robot(self, action=None, people_list=None, robot_id: int = 0):
        """移动指定编号的机器人。

        参数:
            action: 0~4，同单机器人版本
            robot_id: 机器人索引，默认 0 向后兼容
        """
        # 确保 robot_positions 列表足够长
        while len(self.robot_positions) <= robot_id:
            # 初始化在 (15,15)
            self.robot_positions.append([15, 15])

        if action is None:
            x, y = self.robot_positions[robot_id]
            if x >= self.robot_range[1]:
                self.robot_direction = -1
            elif x <= self.robot_range[0]:
                self.robot_direction = 1
            self.robot_positions[robot_id][0] += self.robot_direction
        else:
            if action not in [0, 1, 2, 3, 4]:
                return
            x, y = self.robot_positions[robot_id]
            new_x, new_y = x, y
            if action == 0:
                new_x = x + 1
            elif action == 1:
                new_y = y - 1
            elif action == 2:
                new_x = x - 1
            elif action == 3:
                new_y = y + 1
            elif action == 4:
                pass
            if (self.robot_range[0] <= new_x <= self.robot_range[1] and 
                0 <= new_y <= self.Width and 
                self.Check_Valid(new_x, new_y)):
                self.robot_positions[robot_id] = [new_x, new_y]

        # 更新兼容旧属性
        if robot_id == 0:
            self.robot_position = self.robot_positions[0]
        # 若有人请求 robot_position 但实际上多个机器人，只取第一台。
    def is_robot(self, x, y):
        return int(x) == self.robot_position[0] and int(y) == self.robot_position[1]
