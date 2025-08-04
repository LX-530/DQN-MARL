#!/usr/bin/env python3
"""
奖励函数改进效果对比（简化版）
"""

class RewardComparison:
    """对比新旧奖励函数的效果"""
    
    def __init__(self):
        self.num_people = 150
        
    def calculate_old_reward(self, evacuated, dead, new_evac, new_dead):
        """旧奖励函数（改进前）"""
        # 旧参数
        EVAC_REWARD = 50.0
        DEATH_PENALTY = 200.0
        DEATH_ACC_PENALTY = 0.5
        ALIVE_BONUS = 1.0
        
        reward = 0
        
        # 疏散奖励
        reward += new_evac * EVAC_REWARD
        
        # 死亡惩罚
        reward -= new_dead * DEATH_PENALTY
        reward -= dead * DEATH_ACC_PENALTY
        
        # 存活奖励（这是问题所在！）
        survivors = self.num_people - dead
        reward += survivors * ALIVE_BONUS
        
        # 时间惩罚（旧版本）
        remaining = self.num_people - evacuated - dead
        if remaining > 0:
            urgency_factor = remaining / self.num_people
            time_penalty = -0.05 - (urgency_factor * 0.1)
            reward += time_penalty
        
        return reward
    
    def calculate_new_reward(self, evacuated, dead, new_evac, new_dead):
        """新奖励函数（改进后）"""
        # 新参数
        EVAC_REWARD = 50.0
        DEATH_PENALTY = 500.0  # 提高
        DEATH_ACC_PENALTY = 5.0  # 提高
        ALIVE_BONUS = 0.1  # 大幅降低
        
        reward = 0
        
        # 疏散奖励
        reward += new_evac * EVAC_REWARD
        
        # 死亡惩罚
        reward -= new_dead * DEATH_PENALTY
        reward -= dead * DEATH_ACC_PENALTY
        
        # 存活奖励
        survivors = self.num_people - dead
        reward += survivors * ALIVE_BONUS
        
        # 时间惩罚（新版本）
        remaining = self.num_people - evacuated - dead
        if remaining > 0:
            urgency_factor = remaining / self.num_people
            time_penalty = -0.2 - (urgency_factor * 0.3)  # 加强
            reward += time_penalty
        
        return reward
    
    def print_comparison_table(self):
        """打印对比表格"""
        print("=" * 100)
        print("奖励函数改进效果对比分析")
        print("=" * 100)
        print("\n参数对比：")
        print(f"{'参数':<20} {'旧值':>10} {'新值':>10} {'变化':>15}")
        print("-" * 60)
        print(f"{'死亡惩罚':<20} {'-200':>10} {'-500':>10} {'↑ 150%':>15}")
        print(f"{'持续死亡惩罚':<20} {'-0.5/步':>10} {'-5.0/步':>10} {'↑ 900%':>15}")
        print(f"{'存活奖励':<20} {'1.0/步':>10} {'0.1/步':>10} {'↓ 90%':>15}")
        print(f"{'时间惩罚基准':<20} {'-0.05':>10} {'-0.2':>10} {'↑ 300%':>15}")
        
        print("\n" + "=" * 100)
        print("场景对比分析")
        print("=" * 100)
        
        scenarios = [
            ("初期状态（无事发生）", 0, 0, 0, 0),
            ("疏散10人", 10, 0, 10, 0),
            ("1人死亡", 0, 1, 0, 1),
            ("中期：疏散50人，死亡5人", 50, 5, 5, 1),
            ("后期：疏散100人，死亡20人", 100, 20, 2, 2),
            ("灾难场景：疏散30人，死亡50人", 30, 50, 1, 5),
        ]
        
        print(f"{'场景':<35} {'旧奖励':>12} {'新奖励':>12} {'差异':>12} {'分析':<20}")
        print("-" * 100)
        
        for name, evac, dead, new_evac, new_dead in scenarios:
            old_reward = self.calculate_old_reward(evac, dead, new_evac, new_dead)
            new_reward = self.calculate_new_reward(evac, dead, new_evac, new_dead)
            diff = new_reward - old_reward
            
            # 分析
            if new_dead > 0:
                analysis = "死亡惩罚大幅增加"
            elif dead > 0:
                analysis = "持续惩罚防止忽视"
            elif old_reward > 100:
                analysis = "减少被动奖励"
            else:
                analysis = "保持疏散激励"
            
            print(f"{name:<35} {old_reward:>12.2f} {new_reward:>12.2f} {diff:>12.2f} {analysis:<20}")
        
        print("\n" + "=" * 100)
        print("关键问题分析")
        print("=" * 100)
        
        # 模拟"什么都不做"策略
        print("\n场景：智能体什么都不做，持续300步")
        print("-" * 60)
        
        # 前100步：无人死亡
        old_reward_100 = self.calculate_old_reward(0, 0, 0, 0) * 100
        new_reward_100 = self.calculate_new_reward(0, 0, 0, 0) * 100
        
        print(f"前100步（无死亡）：")
        print(f"  旧奖励累积: {old_reward_100:>10.2f} （每步获得150分存活奖励）")
        print(f"  新奖励累积: {new_reward_100:>10.2f} （存活奖励降至15分/步）")
        
        # 100-200步：累计死亡20人
        old_reward_200 = old_reward_100
        new_reward_200 = new_reward_100
        for i in range(100, 200):
            dead = min(20, (i - 100) // 5)
            new_dead = 1 if i % 5 == 100 and dead < 20 else 0
            old_reward_200 += self.calculate_old_reward(0, dead, 0, new_dead)
            new_reward_200 += self.calculate_new_reward(0, dead, 0, new_dead)
        
        print(f"\n100-200步（累计20人死亡）：")
        print(f"  旧奖励累积: {old_reward_200:>10.2f}")
        print(f"  新奖励累积: {new_reward_200:>10.2f}")
        
        # 200-300步：累计死亡50人
        old_reward_300 = old_reward_200
        new_reward_300 = new_reward_200
        for i in range(200, 300):
            dead = min(50, 20 + (i - 200) // 3)
            new_dead = 1 if (i - 200) % 3 == 0 and dead < 50 else 0
            old_reward_300 += self.calculate_old_reward(0, dead, 0, new_dead)
            new_reward_300 += self.calculate_new_reward(0, dead, 0, new_dead)
        
        print(f"\n200-300步（累计50人死亡）：")
        print(f"  旧奖励累积: {old_reward_300:>10.2f}")
        print(f"  新奖励累积: {new_reward_300:>10.2f}")
        
        print("\n" + "=" * 100)
        print("结论")
        print("=" * 100)
        
        print("\n问题根源：")
        print("• 旧奖励函数中，每步给予 存活人数×1.0 的奖励")
        print("• 150人存活时，每步获得150分，远超死亡惩罚")
        print("• 导致智能体倾向于'什么都不做'来累积存活奖励")
        
        print("\n改进效果：")
        print("• 存活奖励从1.0降至0.1，减少90%")
        print("• 死亡惩罚从200提升至500，增加150%")
        print("• 持续死亡惩罚从0.5提升至5.0，增加900%")
        print("• 时间惩罚加强，促进快速疏散")
        
        print("\n预期结果：")
        print("✓ 智能体将主动引导人员远离危险")
        print("✓ 死亡率将大幅下降")
        print("✓ 疏散效率将显著提高")
        print("✓ 避免'高奖励但高死亡率'的异常情况")


if __name__ == "__main__":
    comparison = RewardComparison()
    comparison.print_comparison_table()