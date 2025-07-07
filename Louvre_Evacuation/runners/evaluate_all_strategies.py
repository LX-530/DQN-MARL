#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评估五种疏散策略：
1. 无机器人
2. 静态机器人 (15,15)
3. 单机器人 DQN
4. 双机器人 Double DQN
5. 双机器人 QMIX（如提供模型）
"""
import os, sys, argparse, yaml, torch, numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Louvre_Evacuation.envs.evacuation_env import EvacuationEnv
from Louvre_Evacuation.envs.evacuation_env_multi import EvacuationEnvMulti
from Louvre_Evacuation.agents.dqn_agent import DQNAgent


# ------------------------------------------------------------------
# 构建环境
# ------------------------------------------------------------------


def build_env_single(cfg):
    env_cfg = cfg['env']
    return EvacuationEnv(
        width=env_cfg['width'],
        height=env_cfg['height'],
        fire_zones=env_cfg['fire_zones'],
        exit_location=env_cfg['exit_location'],
        num_people=env_cfg['num_people'],
    )


def build_env_multi(cfg):
    env_cfg = cfg['env']
    return EvacuationEnvMulti(
        width=env_cfg['width'],
        height=env_cfg['height'],
        fire_zones=env_cfg['fire_zones'],
        exit_location=env_cfg['exit_location'],
        num_people=env_cfg['num_people'],
    )


# ------------------------------------------------------------------
# 策略定义
# ------------------------------------------------------------------


def policy_no_robot(state, env):
    """无机器人：把机器人搬到远处并保持停留"""
    if hasattr(env, 'map'):
        if hasattr(env.map, 'robot_positions'):
            env.map.robot_positions = [[1000, 1000]] * len(getattr(env.map, 'robot_positions', [[0, 0]]))
        if hasattr(env.map, 'robot_position'):
            env.map.robot_position = [1000, 1000]
    return 4  # 停留


def policy_static_robot(state, env):
    """单机器人静止"""
    return 4


def build_single_dqn_policy(model_path, device, env_sample):
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
    agent.epsilon = 0.0

    def _policy(state, _env):
        return agent.act(state)

    return _policy


def build_double_dqn_policy(paths, device, env_sample):
    agent1 = DQNAgent(env_sample.state_size, env_sample.action_size, device, {})
    agent2 = DQNAgent(env_sample.state_size, env_sample.action_size, device, {})
    agent1.load(paths[0]); agent2.load(paths[1])
    agent1.epsilon = agent2.epsilon = 0.0

    def _policy(state_list, _env):
        a1 = agent1.act(state_list[0])
        a2 = agent2.act(state_list[1])
        return [a1, a2]

    return _policy


def build_qmix_policy(paths, device, env_sample):
    agent1 = DQNAgent(env_sample.state_size, env_sample.action_size, device, {})
    agent2 = DQNAgent(env_sample.state_size, env_sample.action_size, device, {})
    agent1.load(paths[0]); agent2.load(paths[1])
    agent1.epsilon = agent2.epsilon = 0.0

    def _policy(state_list, _env):
        return [agent1.act(state_list[0]), agent2.act(state_list[1])]

    return _policy


# ------------------------------------------------------------------
# 通用评估
# ------------------------------------------------------------------


def evaluate(env_builder, policy_fn, episodes):
    metrics = []
    for _ in range(episodes):
        env = env_builder()
        state = env.reset()
        done = False
        while not done:
            action = policy_fn(state, env)
            state, _, done, _ = env.step(action)
        metrics.append(env.get_performance_metrics())
    # 聚合
    avg_health = np.mean([m['avg_health'] for m in metrics])
    death_rate = np.mean([m['death_rate'] for m in metrics])
    avg_time = np.mean([m['total_time'] for m in metrics])
    return avg_health, death_rate, avg_time


# ------------------------------------------------------------------
# 主程序
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--config', default=os.path.join(project_root, 'configs', 'dqn.yaml'))
    parser.add_argument('--single_dqn_path', default=os.path.join(project_root, 'dqn_results', 'best_model.pth'))
    parser.add_argument('--double_dqn_paths', nargs=2, default=[
        os.path.join(project_root, 'dqn_results', 'double_dqn_agent1.pth'),
        os.path.join(project_root, 'dqn_results', 'double_dqn_agent2.pth')])
    parser.add_argument('--qmix_paths', nargs=2, default=[
        os.path.join(project_root, 'dqn_results', 'qmix_agent1.pth'),
        os.path.join(project_root, 'dqn_results', 'qmix_agent2.pth')])
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 单环境 sample
    env_single_sample = build_env_single(cfg)
    env_multi_sample = build_env_multi(cfg)

    # 策略构建
    policies = {
        '无机器人': policy_no_robot,
        '静态机器人': policy_static_robot,
        '单机器人DQN': build_single_dqn_policy(args.single_dqn_path, device, env_single_sample),
        '双机器人DoubleDQN': build_double_dqn_policy(args.double_dqn_paths, device, env_multi_sample),
        '双机器人QMIX': build_qmix_policy(args.qmix_paths, device, env_multi_sample),
    }

    results = {}
    # 评估单环境策略
    results['无机器人'] = evaluate(lambda: build_env_single(cfg), policies['无机器人'], args.episodes)
    results['静态机器人'] = evaluate(lambda: build_env_single(cfg), policies['静态机器人'], args.episodes)
    results['单机器人DQN'] = evaluate(lambda: build_env_single(cfg), policies['单机器人DQN'], args.episodes)
    # 多机器人环境策略
    results['双机器人DoubleDQN'] = evaluate(lambda: build_env_multi(cfg), policies['双机器人DoubleDQN'], args.episodes)
    results['双机器人QMIX'] = evaluate(lambda: build_env_multi(cfg), policies['双机器人QMIX'], args.episodes)

    # 打印结果
    print("\n=== 五策略比较 (episodes = %d) ===" % args.episodes)
    print("策略\t\t平均健康\t死亡率\t平均时间(s)")
    for k, v in results.items():
        print(f"{k}\t{v[0]:.2f}\t{v[1]*100:.1f}%\t{v[2]:.1f}")


if __name__ == '__main__':
    main() 