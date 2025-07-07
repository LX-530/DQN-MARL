#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双机器人 Double DQN 训练脚本（简化实验级）"""

import os, sys, yaml, torch, numpy as np
from collections import deque

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Louvre_Evacuation.envs.evacuation_env_multi import EvacuationEnvMulti
from Louvre_Evacuation.agents.dqn_agent import DQNAgent


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def train_double_dqn():
    cfg = load_config(os.path.join(project_root, 'configs', 'dqn.yaml'))

    env = EvacuationEnvMulti(
        width=cfg['env']['width'],
        height=cfg['env']['height'],
        fire_zones=cfg['env']['fire_zones'],
        exit_location=cfg['env']['exit_location'],
        num_people=cfg['env']['num_people'],
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent_cfg = cfg['agent']

    agent1 = DQNAgent(env.state_size, env.action_size, device, agent_cfg)
    agent2 = DQNAgent(env.state_size, env.action_size, device, agent_cfg)

    episodes = 200  # 根据需求调整至200回合

    # 日志列表
    logs = []

    for ep in range(episodes):
        states = env.reset()  # list[2]
        done = False
        total_reward = 0
        while not done:
            action1 = int(agent1.act(states[0], training=True))
            action2 = int(agent2.act(states[1], training=True))
            next_states, reward, done, info = env.step([action1, action2])

            agent1.remember(states[0], action1, reward, next_states[0], done)
            agent2.remember(states[1], action2, reward, next_states[1], done)

            if len(agent1.memory) > agent1.batch_size:
                agent1.learn(); agent2.learn()

            states = next_states
            total_reward += reward

        logs.append({
            'episode': ep,
            'reward': total_reward,
            'evac_rate': info['evacuation_rate'],
            'death_rate': info['death_rate']
        })

        if ep % 10 == 0:
            print(f"Episode {ep}: reward={total_reward:.2f} evac={info['evacuation_rate']:.1%} death={info['death_rate']:.1%}")

    # 保存
    save_dir = os.path.join(project_root, 'dqn_results')
    os.makedirs(save_dir, exist_ok=True)
    agent1.save(os.path.join(save_dir, 'double_dqn_agent1.pth'))
    agent2.save(os.path.join(save_dir, 'double_dqn_agent2.pth'))

    # 保存日志
    import csv
    log_path = os.path.join(save_dir, 'double_dqn_training_log.csv')
    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['episode','reward','evac_rate','death_rate'])
        writer.writeheader(); writer.writerows(logs)

    print("训练完成并已保存模型与日志")

if __name__ == '__main__':
    train_double_dqn() 