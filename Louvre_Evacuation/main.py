# python -m Louvre_Evacuation.main --train_dqn
import argparse
import sys
import os

# 添加当前项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # 尝试绝对导入
    from Louvre_Evacuation.runners.train_dqn import main as train_dqn_main
except ImportError:
    # 如果绝对导入失败，尝试相对导入
    from runners.train_dqn import main as train_dqn_main

def main():
    parser = argparse.ArgumentParser(description='Louvre Evacuation RL Framework')
    parser.add_argument('--train_dqn', action='store_true', help='Train DQN agent')
    args = parser.parse_args()
    if args.train_dqn:
        train_dqn_main()
    else:
        print('请指定操作，例如 --train_dqn')

if __name__ == '__main__':
    main() 