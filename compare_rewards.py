#!/usr/bin/env python3
"""
奖励函数改进效果对比脚本
展示改进前后的奖励计算差异
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无显示环境下使用
import matplotlib.pyplot as plt

class RewardComparison:
    """对比新旧奖励函数的效果"""
    
    def __init__(self):
        self.num_people = 150
        self.scenarios = [
            # 场景1: 初期状态（无人疏散，无人死亡）
            {"name": "初期状态", "evacuated": 0, "dead": 0, "new_evac": 0, "new_dead": 0},
            
            # 场景2: 成功疏散10人
            {"name": "疏散10人", "evacuated": 10, "dead": 0, "new_evac": 10, "new_dead": 0},
            
            # 场景3: 1人死亡
            {"name": "死亡1人", "evacuated": 0, "dead": 1, "new_evac": 0, "new_dead": 1},
            
            # 场景4: 疏散50人，死亡5人
            {"name": "中期状态", "evacuated": 50, "dead": 5, "new_evac": 5, "new_dead": 1},
            
            # 场景5: 疏散100人，死亡20人
            {"name": "后期状态", "evacuated": 100, "dead": 20, "new_evac": 2, "new_dead": 2},
            
            # 场景6: 全部疏散（无死亡）
            {"name": "完美疏散", "evacuated": 150, "dead": 0, "new_evac": 5, "new_dead": 0},
        ]
    
    def calculate_old_reward(self, scenario):
        """旧奖励函数（改进前）"""
        # 旧参数
        EVAC_REWARD = 50.0
        DEATH_PENALTY = 200.0
        DEATH_ACC_PENALTY = 0.5
        ALIVE_BONUS = 1.0
        
        reward = 0
        
        # 疏散奖励
        reward += scenario["new_evac"] * EVAC_REWARD
        
        # 死亡惩罚
        reward -= scenario["new_dead"] * DEATH_PENALTY
        reward -= scenario["dead"] * DEATH_ACC_PENALTY
        
        # 存活奖励
        survivors = self.num_people - scenario["dead"]
        reward += survivors * ALIVE_BONUS
        
        # 时间惩罚（旧版本）
        remaining = self.num_people - scenario["evacuated"] - scenario["dead"]
        if remaining > 0:
            urgency_factor = remaining / self.num_people
            time_penalty = -0.05 - (urgency_factor * 0.1)
            reward += time_penalty
        
        return reward
    
    def calculate_new_reward(self, scenario):
        """新奖励函数（改进后）"""
        # 新参数
        EVAC_REWARD = 50.0
        DEATH_PENALTY = 500.0  # 提高
        DEATH_ACC_PENALTY = 5.0  # 提高
        ALIVE_BONUS = 0.1  # 降低
        
        reward = 0
        
        # 疏散奖励
        reward += scenario["new_evac"] * EVAC_REWARD
        
        # 死亡惩罚
        reward -= scenario["new_dead"] * DEATH_PENALTY
        reward -= scenario["dead"] * DEATH_ACC_PENALTY
        
        # 存活奖励
        survivors = self.num_people - scenario["dead"]
        reward += survivors * ALIVE_BONUS
        
        # 时间惩罚（新版本）
        remaining = self.num_people - scenario["evacuated"] - scenario["dead"]
        if remaining > 0:
            urgency_factor = remaining / self.num_people
            time_penalty = -0.2 - (urgency_factor * 0.3)  # 加强
            reward += time_penalty
        
        return reward
    
    def generate_comparison_report(self):
        """生成对比报告"""
        print("=" * 80)
        print("奖励函数改进效果对比")
        print("=" * 80)
        print(f"{'场景':<15} {'旧奖励':>12} {'新奖励':>12} {'差异':>12} {'改进说明':<30}")
        print("-" * 80)
        
        old_rewards = []
        new_rewards = []
        
        for scenario in self.scenarios:
            old_reward = self.calculate_old_reward(scenario)
            new_reward = self.calculate_new_reward(scenario)
            diff = new_reward - old_reward
            
            old_rewards.append(old_reward)
            new_rewards.append(new_reward)
            
            # 分析改进效果
            if scenario["new_dead"] > 0:
                improvement = "死亡惩罚大幅增加"
            elif scenario["dead"] > 0:
                improvement = "持续死亡惩罚增强"
            elif scenario["new_evac"] > 0:
                improvement = "疏散激励保持"
            else:
                improvement = "存活奖励减少，促进行动"
            
            print(f"{scenario['name']:<15} {old_reward:>12.2f} {new_reward:>12.2f} "
                  f"{diff:>12.2f} {improvement:<30}")
        
        print("-" * 80)
        
        # 生成对比图
        self.plot_comparison(old_rewards, new_rewards)
        
        # 关键洞察
        print("\n关键改进点：")
        print("1. 死亡惩罚从-200提升到-500，使避免死亡成为最高优先级")
        print("2. 持续死亡惩罚从-0.5/人·步提升到-5.0/人·步，防止忽视已死亡人员")
        print("3. 存活奖励从1.0/人·步降至0.1/人·步，避免'什么都不做'的策略")
        print("4. 时间惩罚加强，促使智能体更快完成疏散任务")
        
        print("\n预期效果：")
        print("- 智能体将主动引导人员远离危险区域")
        print("- 死亡率将显著下降")
        print("- 疏散效率将提高")
        print("- 避免出现'高奖励但高死亡率'的异常情况")
        
    def plot_comparison(self, old_rewards, new_rewards):
        """绘制对比图"""
        plt.figure(figsize=(12, 8))
        
        # 子图1: 奖励对比
        plt.subplot(2, 1, 1)
        x = range(len(self.scenarios))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], old_rewards, width, label='旧奖励函数', color='red', alpha=0.7)
        plt.bar([i + width/2 for i in x], new_rewards, width, label='新奖励函数', color='green', alpha=0.7)
        
        plt.xlabel('场景')
        plt.ylabel('奖励值')
        plt.title('新旧奖励函数对比')
        plt.xticks(x, [s["name"] for s in self.scenarios], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2: 差异分析
        plt.subplot(2, 1, 2)
        differences = [new - old for new, old in zip(new_rewards, old_rewards)]
        colors = ['green' if d > 0 else 'red' for d in differences]
        
        plt.bar(x, differences, color=colors, alpha=0.7)
        plt.xlabel('场景')
        plt.ylabel('奖励差异（新-旧）')
        plt.title('奖励函数改进效果')
        plt.xticks(x, [s["name"] for s in self.scenarios], rotation=45)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig('/workspace/reward_comparison.png', dpi=150)
        print(f"\n对比图已保存到: /workspace/reward_comparison.png")
        
    def simulate_episode_comparison(self):
        """模拟一个完整回合的奖励累积对比"""
        print("\n" + "=" * 80)
        print("模拟回合奖励累积对比")
        print("=" * 80)
        
        # 模拟场景：前100步什么都不做，后续开始有人死亡
        steps = 300
        
        old_cumulative = 0
        new_cumulative = 0
        
        old_rewards = []
        new_rewards = []
        
        for step in range(steps):
            # 模拟场景演变
            if step < 100:
                # 前期：无事发生
                scenario = {"evacuated": 0, "dead": 0, "new_evac": 0, "new_dead": 0}
            elif step < 150:
                # 中期：开始有人死亡
                dead_count = (step - 100) // 10
                scenario = {"evacuated": 0, "dead": dead_count, "new_evac": 0, 
                           "new_dead": 1 if step % 10 == 100 else 0}
            else:
                # 后期：大量死亡
                dead_count = 5 + (step - 150) // 5
                scenario = {"evacuated": 0, "dead": min(dead_count, 50), "new_evac": 0,
                           "new_dead": 1 if step % 5 == 0 else 0}
            
            old_step_reward = self.calculate_old_reward(scenario)
            new_step_reward = self.calculate_new_reward(scenario)
            
            old_cumulative += old_step_reward
            new_cumulative += new_step_reward
            
            old_rewards.append(old_cumulative)
            new_rewards.append(new_cumulative)
        
        # 绘制累积奖励对比
        plt.figure(figsize=(12, 6))
        plt.plot(old_rewards, label='旧奖励函数（累积）', color='red', linewidth=2)
        plt.plot(new_rewards, label='新奖励函数（累积）', color='green', linewidth=2)
        
        plt.xlabel('时间步')
        plt.ylabel('累积奖励')
        plt.title('不作为策略下的累积奖励对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 标注关键时刻
        plt.axvline(x=100, color='orange', linestyle='--', alpha=0.5, label='开始出现死亡')
        plt.axvline(x=150, color='red', linestyle='--', alpha=0.5, label='大量死亡')
        
        plt.tight_layout()
        plt.savefig('/workspace/cumulative_reward_comparison.png', dpi=150)
        print(f"累积奖励对比图已保存到: /workspace/cumulative_reward_comparison.png")
        
        print(f"\n最终累积奖励：")
        print(f"旧奖励函数: {old_cumulative:.2f}")
        print(f"新奖励函数: {new_cumulative:.2f}")
        print(f"差异: {new_cumulative - old_cumulative:.2f}")
        
        if old_cumulative > 0 and new_cumulative < 0:
            print("\n✓ 改进成功：旧函数下'不作为'能获得正奖励，新函数下会受到严重惩罚！")


if __name__ == "__main__":
    comparison = RewardComparison()
    comparison.generate_comparison_report()
    comparison.simulate_episode_comparison()