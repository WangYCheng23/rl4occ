'''
Author: WANG CHENG
Date: 2024-04-20 01:46:06
LastEditTime: 2024-04-30 00:56:26

'''
import os
import pickle
import sys
import numpy as np
from env import Env
from dqnlearn import DQNAgent
from memory_profiler import profile

cwd = os.getcwd()

if sys.platform == 'linux':
    train_dir = os.path.join(cwd, 'data/train/')
    pickle_dir = os.path.join(cwd, 'pickle_data/')
elif sys.platform == 'win32':
    train_dir = f'D:\\Document\\work\\rl4occ\\data\\train'
    pickle_dir = f'D:\\Document\\work\\rl4occ\\pickle_data'
    
step_filenames = [os.path.join(train_dir, path) for path in os.listdir(train_dir)]
# pickle_dataset = np.random.permutation([os.path.join(pickle_dir, pickle_path) for pickle_path in os.listdir(pickle_dir)])
pickle_dataset = [pickle.load(open(os.path.join(pickle_dir, pickle_path),'rb')) for pickle_path in os.listdir(pickle_dir)]
env = Env(step_filenames, pickle_dataset)
buffer_size = 5000  # 定义了经验回放缓冲区的大小
dqn_agent = DQNAgent(env, buffer_size)
test = False
if test:
    states = env.reset()
    print(f"初始状态:{states}")
    # embed_dim = 16
    # hidden_dim = 64
    # num_heads = 4
    # output_dim = 1  # 输出维度
    # seq_length = len(states)  # 序列长度
    
    # policy = AttentionQNet(output_dim, hidden_dim, embed_dim, num_heads)
    # policy(torch.FloatTensor(states).reshape(1,-1,1))
    


    action = dqn_agent.choose_action(states)
    print(f"动作:{action}")
    next_state, reward, isterminated = env.step(action)
    print(f"下一个状态:{next_state}\n即时奖励:{reward}\n是否终止:{isterminated}")

episode_nums = 5000  # 定义了训练的总回合数

dqn_agent.learn(episode_nums)
dqn_agent.log.close()  # 保存训练好的模型
# dqn_agent.load_model()  # 加载训练好的dqn模型
# dqn_agent.save_optimevalue()  # 保存好得到的dqn初始解
