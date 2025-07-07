# python -m Louvre_Evacuation.tests.test_fixes
import numpy as np
from ..envs.evacuation_env import EvacuationEnv

def test_fixes():
    print("=== 测试修复后的系统 ===")
    
    # 创建环境
    env = EvacuationEnv(width=36, height=30, num_people=150)
    print(f"地图尺寸: {env.width} x {env.height}")
    print(f"出口位置: {env.exit_location}")
    print(f"人员数量: {env.num_people}")
    state = env.reset()
    print(f"\n初始状态:")
    evacuated = sum(1 for p in env.people.list if p.savety)
    dead = sum(1 for p in env.people.list if p.dead)
    print(f"已疏散: {evacuated}, 死亡: {dead}, 存活: {env.num_people - evacuated - dead}")
    for step in range(50):
        action = np.random.randint(0, 5)
        next_state, reward, done = env.step(action)
        if (step + 1) % 10 == 0:
            evacuated = sum(1 for p in env.people.list if p.savety)
            dead = sum(1 for p in env.people.list if p.dead)
            remaining = env.num_people - evacuated - dead
            print(f"步骤 {step + 1}: 疏散={evacuated}, 死亡={dead}, 存活={remaining}, 奖励={reward:.2f}")
            avg_health = np.mean([p.health for p in env.people.list if not p.savety and not p.dead])
            if remaining > 0:
                print(f"  平均健康值: {avg_health:.2f}")
        if done:
            print(f"环境在第{step + 1}步结束")
            break
    evacuated = sum(1 for p in env.people.list if p.savety)
    dead = sum(1 for p in env.people.list if p.dead)
    remaining = env.num_people - evacuated - dead
    print(f"\n最终结果:")
    print(f"疏散率: {evacuated/env.num_people:.2%}")
    print(f"死亡率: {dead/env.num_people:.2%}")
    print(f"存活率: {remaining/env.num_people:.2%}")
    print(f"\n人员位置分布:")
    for i, p in enumerate(env.people.list[:5]):
        if not p.savety and not p.dead:
            dist_to_exit = np.linalg.norm(np.array(p.pos) - np.array(env.exit_location))
            print(f"人员{i}: 位置={p.pos}, 距出口={dist_to_exit:.2f}, 健康值={p.health:.2f}")
        else:
            status = "已疏散" if p.savety else "已死亡"
            print(f"人员{i}: {status}, 健康值={p.health:.2f}")

if __name__ == "__main__":
    test_fixes() 