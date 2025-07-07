#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""overnight_experiments.py

批量搜索 EvacuationEnv 奖励函数系数，自动记录结果并输出最优组合。

搜索空间（可自行调整）：
    DEATH_PENALTY:   [150, 200, 250, 300]
    ALIVE_BONUS:     [0.3, 0.5, 0.7]

每组参数跑 `episodes_per_run` 回合（默认 100，纯采样，不做学习），
统计平均死亡率 / 疏散率 / 平均健康 / 平均时间。

结果：
  • dqn_results/experiment_summary.csv  保存全部组合指标
  • dqn_results/best_reward_cfg.json    保存最优组合（死亡率最小再看疏散率）
"""
import os, sys, itertools, csv, json, yaml, torch, numpy as np, argparse, random
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Louvre_Evacuation.envs.evacuation_env import EvacuationEnv
from Louvre_Evacuation.agents.dqn_agent import DQNAgent

# --------------------------- 命令行解析 --------------------------- #
parser = argparse.ArgumentParser(description="批量搜索奖励系数")
parser.add_argument('--episodes', type=int, default=200, help='每组参数回合数')
parser.add_argument('--death_min', type=int, default=100)
parser.add_argument('--death_max', type=int, default=400)
parser.add_argument('--death_step', type=int, default=50)
parser.add_argument('--alive_min', type=float, default=0.2)
parser.add_argument('--alive_max', type=float, default=1.0)
parser.add_argument('--alive_step', type=float, default=0.1)
parser.add_argument('--random_sample', type=int, default=0, help='若 >0，则随机抽样该数量组合而非遍历')
args_cli = parser.parse_args()

# 生成搜索列表
DEATH_PENALTY_LIST = list(range(args_cli.death_min, args_cli.death_max + 1, args_cli.death_step))
ALIVE_BONUS_LIST = [round(x,2) for x in np.arange(args_cli.alive_min, args_cli.alive_max + 1e-6, args_cli.alive_step)]
episodes_per_run = args_cli.episodes

# --------------------------- 配置加载 --------------------------- #

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg_path = os.path.join(project_root, 'configs', 'dqn.yaml')
with open(cfg_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

env_cfg   = cfg['env']
agent_cfg = cfg['agent']

# --------------------------- 主循环 --------------------------- #
results = []
param_grid = list(itertools.product(DEATH_PENALTY_LIST, ALIVE_BONUS_LIST))
print(f"总组合数: {len(param_grid)}, 每组 {episodes_per_run} 回合\n")

# 随机抽样
if args_cli.random_sample and args_cli.random_sample < len(param_grid):
    random.seed(42)
    param_grid = random.sample(param_grid, args_cli.random_sample)

for run_id, (death_pen, alive_bonus) in enumerate(param_grid, 1):
    print(f"▶ 组合 {run_id}/{len(param_grid)} : DEATH_PENALTY={death_pen}, ALIVE_BONUS={alive_bonus}")

    # 动态修改类属性
    EvacuationEnv.DEATH_PENALTY   = float(death_pen)
    EvacuationEnv.ALIVE_BONUS     = float(alive_bonus)
    # DEATH_ACC_PENALTY 保持 1.0 不变，可按需再加搜索维度

    # 创建环境 & 随机初始化 agent（纯采样，不学习以加速）
    env   = EvacuationEnv(**env_cfg)
    agent = DQNAgent(env.state_size, env.action_size, device, agent_cfg)

    death_rates, evac_rates, healths, times = [], [], [], []

    for ep in tqdm(range(episodes_per_run), desc="   回合", leave=False):
        state = env.reset(); done = False
        while not done:
            action = agent.act(state, training=True)  # ε-greedy 随机策略即可
            state, _, done, _ = env.step(action)
        metr = env.get_performance_metrics()
        death_rates.append(metr['death_rate'])
        evac_rates.append(metr['evacuation_rate'])
        healths.append(metr['avg_health'])
        times.append(metr['total_time'])

    avg_death   = float(np.mean(death_rates))
    avg_evac    = float(np.mean(evac_rates))
    avg_health  = float(np.mean(healths))
    avg_time    = float(np.mean(times))

    print(f"   ↳ 平均死亡率 {avg_death:.3f}  平均疏散率 {avg_evac:.3f}\n")
    results.append((death_pen, alive_bonus, avg_death, avg_evac, avg_health, avg_time))

# --------------------------- 保存结果 --------------------------- #
res_dir = os.path.join(project_root, 'dqn_results')
os.makedirs(res_dir, exist_ok=True)
csv_fp = os.path.join(res_dir, 'experiment_summary.csv')
with open(csv_fp, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['death_penalty', 'alive_bonus', 'death_rate', 'evac_rate', 'avg_health', 'avg_time'])
    writer.writerows(results)

# 选最优：死亡率最小，然后疏散率最高
best = sorted(results, key=lambda x: (x[2], -x[3]))[0]
best_cfg = {'death_penalty': best[0], 'alive_bonus': best[1]}
json_fp = os.path.join(res_dir, 'best_reward_cfg.json')
with open(json_fp, 'w', encoding='utf-8') as f:
    json.dump(best_cfg, f, indent=2, ensure_ascii=False)

print("\n✅ 实验结束，最佳参数:", best_cfg)
print("CSV 结果保存到:", csv_fp)
print("最佳参数 JSON 保存到:", json_fp) 