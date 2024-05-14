#-- coding:UTF-8 --
'''
Author: WANG CHENG
Date: 2024-04-20 01:46:06
LastEditTime: 2024-05-14 14:12:40

'''
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pickle
import sys
import numpy as np
from env import Env
from dqnlearn import DQNAgent
from multiprocessing import pool, freeze_support
from memory_profiler import profile
from functools import partial

def load_pickle(pickle_dir, pickle_path):
    return pickle.load(open(os.path.join(pickle_dir, pickle_path),'rb'))

if __name__ == '__main__':
    freeze_support()
    
    cwd = os.getcwd()

    if sys.platform == 'linux':
        train_dir = os.path.join(cwd, 'sorted_step_files/10-30个')
        pickle_dir = os.path.join(cwd, 'pickle_data/10-30个')
    elif sys.platform == 'win32':
        train_dir = os.path.join(cwd, f'sorted_step_files\\10-30个')
        pickle_dir = os.path.join(cwd, f'pickle_data\\10-30个')
        
    step_filenames = [os.path.join(train_dir, path) for path in os.listdir(train_dir)]
    # pickle_dataset = np.random.permutation([os.path.join(pickle_dir, pickle_path) for pickle_path in os.listdir(pickle_dir)])
    pickle_data_list = np.random.permutation([pickle_path for pickle_path in os.listdir(pickle_dir)])[:50]
    # pickle_dataset = [pickle.load(open(os.path.join(pickle_dir, pickle_path),'rb')) for pickle_path in pickle_data_list]
    par = partial(load_pickle, pickle_dir)

    mypool = pool.Pool(4)
    pickle_dataset = mypool.map(par, pickle_data_list)
    mypool.close()        # 关闭进程池，不再接受新的进程
    mypool.join()         # 主进程阻塞等待子进程的退出

    env = Env(step_filenames, pickle_dataset)
    buffer_size = 30000  # 定义了经验回放缓冲区的大小
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

    episode_nums = 100000  # 定义了训练的总回合数

    dqn_agent.learn(episode_nums)
    dqn_agent.log.close()  # 保存训练好的模型
    # dqn_agent.load_model()  # 加载训练好的dqn模型
    # dqn_agent.save_optimevalue()  # 保存好得到的dqn初始解
