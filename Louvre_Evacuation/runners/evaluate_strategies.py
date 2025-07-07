#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略性能比较脚本

本脚本评估以下三种情形在疏散环境中的表现（平均健康值、疏散时间）：
1. 无机器人（baseline）
2. 静态机器人（始终停留在 (15,15)）
3. 训练好的 DQN 机器人

使用方法：
python evaluate_strategies.py --model_path ../../dqn_results/best_model.pth --episodes 20

如果未提供 --model_path，将尝试自动在 dqn_results/best_model.pth 查找。
"""

import os
import sys
import argparse
import numpy as np
import yaml
import torch

# 调整路径，确保可以正确导入包
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Louvre_Evacuation.envs.evacuation_env import EvacuationEnv
from Louvre_Evacuation.agents.dqn_agent import DQNAgent


# ------------------------ 辅助函数 ------------------------ #

def build_env_from_config(cfg):
    """根据配置创建 EvacuationEnv。"""
    env_cfg = cfg['env'] if 'env' in cfg else cfg.get('environment', {})
    return EvacuationEnv(
        width=env_cfg.get('width', 36),
        height=env_cfg.get('height', 30),
        fire_zones=env_cfg.get('fire_zones'),
        exit_location=env_cfg.get('exit_location'),
        num_people=env_cfg.get('num_people', 150),
    )


def evaluate(env_builder, policy_fn, episodes):
    """通用评估函数。

    参数:
        env_builder: () -> EvacuationEnv  的可调用对象，每次调用应返回一个 *新* 的环境实例。
        policy_fn: (state, env) -> action  的函数，给定状态和环境返回动作。
        episodes: 评估回合数。
    返回:
        dict，包含平均健康、平均疏散时间以及原始记录。
    """
    metrics_list = []
    for ep in range(episodes):
        env = env_builder()
        state = env.reset()
        done = False
        while not done:
            action = policy_fn(state, env)
            state, _, done, _ = env.step(action)
        metrics = env.get_performance_metrics()
        metrics_list.append(metrics)
    # 统计
    avg_health = np.mean([m['avg_health'] for m in metrics_list])
    avg_time = np.mean([m['total_time'] for m in metrics_list])
    return {
        'records': metrics_list,
        'avg_health': avg_health,
        'avg_time': avg_time,
    }


# ------------------------ 三种策略 ------------------------ #

def no_robot_policy(state, env):
    """无机器人：将机器人放在远离人群的位置并保持静止。"""
    # 仅在第一次调用时移走机器人
    if getattr(env, '_no_robot_shifted', False) is False:
        env.map.robot_position = [1000, 1000]
        env._no_robot_shifted = True
    return 4  # action 4 = 停留


def static_robot_policy(state, env):
    """静态机器人：始终停留在 (15,15)。"""
    return 4  # 停留


def build_dqn_policy(model_path, device, env_sample):
    """根据训练好的模型构造策略函数。"""
    agent = DQNAgent(env_sample.state_size, env_sample.action_size, device, {
        'gamma': 0.99,
        'epsilon': 0.0,
        'epsilon_min': 0.0,
        'epsilon_decay': 1.0,
        'learning_rate': 1e-4,
        'batch_size': 32,
        'memory_size': 10000,
    })
    agent.load(model_path)
    agent.epsilon = 0.0  # 关闭探索

    def _policy(state, _env):
        return agent.act(state)

    return _policy


# ------------------------ 主程序 ------------------------ #

def main():
    parser = argparse.ArgumentParser(description="比较无机器人、静态机器人与 DQN 机器人的性能")
    parser.add_argument('--config', default=os.path.join(project_root, 'configs', 'dqn.yaml'), help='配置文件路径')
    parser.add_argument('--model_path', default=os.path.join(project_root, 'dqn_results', 'best_model.pth'), help='DQN 模型路径')
    parser.add_argument('--episodes', type=int, default=20, help='评估回合数')
    args = parser.parse_args()

    # 加载配置
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    # 构造环境生成器
    env_builder = lambda: build_env_from_config(cfg)
    env_sample = env_builder()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 无机器人
    res_no_robot = evaluate(env_builder, no_robot_policy, args.episodes)
    print("\n=== 无机器人 (baseline) ===")
    print(f"平均健康值: {res_no_robot['avg_health']:.2f}, 平均疏散时间: {res_no_robot['avg_time']:.1f}s")

    # 静态机器人
    res_static = evaluate(env_builder, static_robot_policy, args.episodes)
    print("\n=== 静态机器人 (15,15) ===")
    print(f"平均健康值: {res_static['avg_health']:.2f}, 平均疏散时间: {res_static['avg_time']:.1f}s")

    # DQN 机器人
    if not os.path.exists(args.model_path):
        print(f"⚠️  DQN 模型文件不存在: {args.model_path}，跳过 DQN 评估。")
        return

    dqn_policy = build_dqn_policy(args.model_path, device, env_sample)
    res_dqn = evaluate(env_builder, dqn_policy, args.episodes)
    print("\n=== DQN 机器人 ===")
    print(f"平均健康值: {res_dqn['avg_health']:.2f}, 平均疏散时间: {res_dqn['avg_time']:.1f}s")

    # 总结
    print("\n=== 总结 ===")
    print("策略\t\t平均健康值\t平均疏散时间(s)")
    print(f"无机器人\t{res_no_robot['avg_health']:.2f}\t\t{res_no_robot['avg_time']:.1f}")
    print(f"静态机器人\t{res_static['avg_health']:.2f}\t\t{res_static['avg_time']:.1f}")
    print(f"DQN 机器人\t{res_dqn['avg_health']:.2f}\t\t{res_dqn['avg_time']:.1f}")


if __name__ == "__main__":
    main() 