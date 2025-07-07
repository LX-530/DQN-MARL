#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多机器人疏散环境（2 台机器人）

保持与单机器人版 `EvacuationEnv` 接口尽量一致，但 `reset()` 与 `step()`
返回 **列表状态** `[state_robot0, state_robot1]`，而 `step()` 接收动作列表
`[action0, action1]`。

动作编码仍为 0~4（右/上/左/下/停）。
"""

from typing import List, Tuple
import numpy as np
from .evacuation_env import EvacuationEnv

class EvacuationEnvMulti(EvacuationEnv):
    """双机器人版本，机器人初始位置 (10,15) 与 (20,15)"""

    def __init__(self, width=36, height=30, fire_zones=None, exit_location=None, num_people=150):
        # 先定义机器人数量，供父类 reset 使用
        self.num_robots = 2

        # 直接调用父类构造
        super().__init__(width, height, fire_zones, exit_location, num_people)

        # 覆盖机器人初始位置
        self.map.robot_positions = [[10, 15], [20, 15]]
        self.map.robot_position = self.map.robot_positions[0]  # 向后兼容

    # -------------------------- 重载核心方法 -------------------------- #
    def reset(self) -> List[np.ndarray]:
        """返回两个机器人的局部状态列表"""
        state0 = super().reset()
        # 在父类 reset 中机器人位置被重置，现在再覆盖
        self.map.robot_positions = [[10, 15], [20, 15]]
        self.map.robot_position = self.map.robot_positions[0]
        # 记录轨迹
        self.robot_trajectory = [
            (tuple(self.map.robot_positions[0]), 0),
            (tuple(self.map.robot_positions[1]), 0),
        ]
        return self._get_joint_state()

    def _get_joint_state(self):
        """返回 [state_robot0, state_robot1]"""
        states = []
        for idx in range(self.num_robots):
            # 临时切换 map.robot_position 以复用父类 _get_state()
            orig = self.map.robot_position
            self.map.robot_position = self.map.robot_positions[idx]
            states.append(super()._get_state())
            self.map.robot_position = orig
        return states

    def step(self, actions: List[int]):
        """actions = [a0, a1]"""
        assert len(actions) == self.num_robots, "需要为每台机器人提供一个动作"

        # 依次移动机器人
        for ridx, act in enumerate(actions):
            self.map.move_robot(act, self.people.list, robot_id=ridx)
            # 记录轨迹
            self.robot_trajectory.append((tuple(self.map.robot_positions[ridx]), self.current_step))

        # 其余逻辑与父类一致
        evac_count = self.people.run(self.time_per_step)
        # 更新火灾模型
        if hasattr(self.fire_model, 'update'):
            self.fire_model.update()
        if hasattr(self.map, 'fire_model') and hasattr(self.map.fire_model, 'update'):
            self.map.fire_model.update()

        reward = self._calculate_reward()
        self.time += self.time_per_step
        self.current_step += 1

        # 结束条件复用父类
        current_evacuated = sum(1 for p in self.people.list if p.savety)
        current_dead = sum(1 for p in self.people.list if p.dead)
        done = (current_evacuated + current_dead == self.num_people) or (self.time >= self.max_simulation_time)

        info = {
            'robot_positions': [tuple(pos) for pos in self.map.robot_positions],
            'evacuation_rate': current_evacuated / self.num_people,
            'death_rate': current_dead / self.num_people,
            'current_step': self.current_step,
            'simulation_time': self.time,
        }
        return self._get_joint_state(), reward, done, info 