#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""简化版 QMIX 训练脚本（实际为 VDN，总 Q=Q1+Q2），用于双机器人实验。
重点：快速得到一个能学到基本策略的模型，满足实验对比需求。"""

import os, sys, yaml, torch, numpy as np, random
from collections import deque
from torch.optim.adam import Adam
import csv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Louvre_Evacuation.envs.evacuation_env_multi import EvacuationEnvMulti
from Louvre_Evacuation.agents.dqn_agent import DQNAgent  # 复用单体网络结构


def load_cfg():
    cfg_path = os.path.join(project_root, 'configs', 'dqn.yaml')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def train_qmix():
    cfg = load_cfg()
    env = EvacuationEnvMulti(**cfg['env'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent_cfg = cfg['agent']
    agent1 = DQNAgent(env.state_size, env.action_size, device, agent_cfg)
    agent2 = DQNAgent(env.state_size, env.action_size, device, agent_cfg)

    episodes = 200  # 需求指定200回合
    gamma = agent_cfg.get('gamma', 0.99)

    replay = deque(maxlen=5000)

    # 定义 mixing 网络
    class MixingNetwork(torch.nn.Module):
        def __init__(self, n_agents: int, embed_dim: int = 32):
            super().__init__()
            self.n_agents = n_agents
            self.embed_dim = embed_dim
            self.fc1_weight = torch.nn.Parameter(torch.randn(n_agents, embed_dim))
            self.fc1_bias = torch.nn.Parameter(torch.zeros(embed_dim))
            self.fc2_weight = torch.nn.Parameter(torch.randn(embed_dim, 1))
            self.fc2_bias = torch.nn.Parameter(torch.zeros(1))

        def forward(self, q_vals: torch.Tensor):  # (batch, n_agents)
            w1 = torch.abs(self.fc1_weight)
            w2 = torch.abs(self.fc2_weight)
            hidden = torch.relu(torch.matmul(q_vals, w1) + self.fc1_bias)
            y = torch.matmul(hidden, w2) + self.fc2_bias
            return y.squeeze(-1)  # (batch,)

    mixing = MixingNetwork(n_agents=2).to(device)
    target_mixing = MixingNetwork(n_agents=2).to(device)
    target_mixing.load_state_dict(mixing.state_dict())

    mix_optimizer = Adam(mixing.parameters(), lr=1e-3)

    logs = []

    for ep in range(episodes):
        states = env.reset()
        done = False
        total_reward = 0
        while not done:
            a1 = int(agent1.act(states[0], training=True))
            a2 = int(agent2.act(states[1], training=True))
            next_states, reward, done, info = env.step([a1, a2])

            replay.append((states, [a1, a2], reward, next_states, done))
            states = next_states
            total_reward += reward

            # 学习
            if len(replay) >= agent1.batch_size:
                batch = random.sample(replay, agent1.batch_size)

                # 批量张量
                s_b1 = torch.FloatTensor(np.array([b[0][0] for b in batch])).to(device)
                s_b2 = torch.FloatTensor(np.array([b[0][1] for b in batch])).to(device)
                a_b1 = torch.LongTensor(np.array([b[1][0] for b in batch])).to(device)
                a_b2 = torch.LongTensor(np.array([b[1][1] for b in batch])).to(device)
                r_b = torch.FloatTensor(np.array([b[2] for b in batch])).to(device)
                ns_b1 = torch.FloatTensor(np.array([b[3][0] for b in batch])).to(device)
                ns_b2 = torch.FloatTensor(np.array([b[3][1] for b in batch])).to(device)
                d_b = torch.BoolTensor(np.array([b[4] for b in batch])).to(device)

                # 当前 Q
                q1 = agent1.q_network(s_b1).gather(1, a_b1.unsqueeze(1)).squeeze()
                q2 = agent2.q_network(s_b2).gather(1, a_b2.unsqueeze(1)).squeeze()
                q_cat = torch.stack([q1, q2], dim=1)
                q_tot = mixing(q_cat)

                # 目标 Q
                with torch.no_grad():
                    max_next_q1 = agent1.target_network(ns_b1).max(1)[0]
                    max_next_q2 = agent2.target_network(ns_b2).max(1)[0]
                    target_q_cat = torch.stack([max_next_q1, max_next_q2], dim=1)
                    target_q_tot = target_mixing(target_q_cat)
                    y_tot = r_b + gamma * target_q_tot * (~d_b)

                # 损失
                loss = torch.nn.functional.mse_loss(q_tot, y_tot)

                mix_optimizer.zero_grad(); agent1.optimizer.zero_grad(); agent2.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent1.q_network.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(agent2.q_network.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(mixing.parameters(), 1.0)
                mix_optimizer.step(); agent1.optimizer.step(); agent2.optimizer.step()

                # 更新 target 网路
                if random.random() < 0.01:
                    agent1.update_target_network(); agent2.update_target_network()
                    target_mixing.load_state_dict(mixing.state_dict())

        logs.append({'episode': ep, 'reward': total_reward, 'evac_rate': info['evacuation_rate'], 'death_rate': info['death_rate']})

        if ep % 10 == 0:
            print(f"QMIX Episode {ep}: reward={total_reward:.1f} evac={info['evacuation_rate']:.1%} death={info['death_rate']:.1%}")

    save_dir = os.path.join(project_root, 'dqn_results')
    os.makedirs(save_dir, exist_ok=True)
    agent1.save(os.path.join(save_dir, 'qmix_agent1.pth'))
    agent2.save(os.path.join(save_dir, 'qmix_agent2.pth'))
    log_path = os.path.join(save_dir, 'qmix_training_log.csv')
    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['episode','reward','evac_rate','death_rate'])
        writer.writeheader(); writer.writerows(logs)

    print('QMIX 训练完成并保存模型与日志')

if __name__ == '__main__':
    train_qmix() 