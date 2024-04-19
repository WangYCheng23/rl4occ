
import copy
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Env import Env
from replay_buffer import ReplayBuffer
from attention_q_net import MultiHeadAttention, AttentionQNet

class DQNAgent:

    def __init__(self, env, buffer_size):  # 定义环境对象和经验回放缓冲区的大小
        print("---初始化dqn agent---")
        self.env = env  # 定义环境对象
        
        # Q-net 超参数
        self.input_dim = 10
        self.output_dim = 1
        self.hidden_dim = 64
        self.embed_dim = 16
        self.num_heads = 4
        
        self.eval_q_net = AttentionQNet(self.input_dim, self.output_dim, self.hidden_dim, self.embed_dim, self.num_heads)  # 定义q值的估计网络
        self.target_q_net = AttentionQNet(self.input_dim, self.output_dim, self.hidden_dim, self.embed_dim, self.num_heads)  # 定义q值的目标网络
        # 目标网络和估值网络权重一开始相同，为了在深度 Q 学习算法中稳定训练和提高效率
        self.target_q_net.load_state_dict(self.eval_q_net.state_dict())
        
        # 创建一个大小为buffer_size的经验回放缓冲区，用于存储智能体与环境交互的经验数据
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # 训练超参数
        self.n_steps_update = 5  # 定义每次训练时使用的步数
        self.batch_size = 64  # 定义每次训练时的批量大小
        # 使用Adam优化器来优化估计网络的参数，学习率为2e-4（α）。
        self.optimizer = torch.optim.Adam(self.eval_q_net.parameters(), lr=2e-4)
        self.replace_steps_cycle = 60  # 定义替换目标网络参数的周期步数
        self.episilon = 0.76  # 定义ε贪婪策略中的ε值
        self.gamma = 0.998  # 定义强化学习中的折扣因子，用于调节当前奖励和未来奖励的重要性
        self.save_cycyle = 10  # 定义保存模型的周期步数

    def stepped_episilon(self):
        return 0.76  # 用于ε贪婪策略，用于在探索和利用之间进行权衡

    def save_model(self, itr):  # 保存q估值网络
        # os.mkdir(f'./model/{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        torch.save(self.eval_q_net, f'./model/eval_q_net-{itr}.pth')

    def load_model(self):  # 加载深度Q学习算法中的Q值估计网络（q_net）便于继续训练和预测
        self.eval_q_net = torch.load('./model/best_eval_q_net.pth')

    def choose_action(self, state):  # 在给定状态下选择动作的过程

        # 根据ε-greedy策略来选择动作，当随机数小于ε时，以一定概率选择随机动作
        if np.random.uniform(0, 1) < self.stepped_episilon():

            # maxactionid = len(self.env.stepedparts)  # 获取已经执行的动作数量，用于限制后续动作的选择
            # p = np.ones([self.n_actions])  # 初始化动作概率数组，所有动作的概率都设为1
            # if maxactionid+1 < self.n_actions:  # 如果已执行的动作数量加1小于总动作数量，说明还有未执行的动作
            #     p[maxactionid+1:] = 1 / (self.n_actions-maxactionid-1)  # 将未执行的动作的概率设为均匀分布

            # p = p/np.sum(p)  # 将概率归一化，确保概率之和为1

            action = int(np.random.choice(self.env.unstepparts))  # 根据概率分布选择动作

        else: 
            state = torch.FloatTensor(state).reshape(1,-1,self.input_dim)
            self.eval_q_net.eval()  # 将Q网络设置为评估模式，确保在选择动作时不会更新其参数
            Q_vals = self.eval_q_net(state)  # 使用Q网络预测当前状态下各个动作的Q值
            masked_positions = self.env.stepedparts
            # 创建掩码张量
            mask = torch.ones_like(Q_vals)  # 先创建一个全 1 的张量
            # 将需要掩盖的位置置零
            mask[:,masked_positions,:] = 0
            Q_vals = Q_vals.masked_fill_(mask==0, -float('inf')).detach().cpu().numpy()[0, :]  # 将Q值张量转换为NumPy数组，以便后续处理
            action = np.argmax(Q_vals)  # 选择具有最大Q值的动作作为最优动作

        return action  # 返回选择的动作

    def learn(self, episode_nums):
        accrewards = []  # 创建一个空列表 accrewards，用于存储每轮训练的累积奖励
        rewards_per_steps = []  # 创建一个空列表 rewards_per_steps，用于存储每轮训练的奖励/步数

        step_ = 0  # 初始化步数计数器 step_ 为0
        for i in range(episode_nums):  # 循环执行训练指定次数 episode_nums

            # records = {'state': [], 'next_state': [], 'actions': [], 'r': [
            # ], 'isterminated': []}  # 创建一个字典 records，用于存储每个 episode 中的经验数据

            self.env.reset()

            accreward = 0
            count = 0
            isterminated = False

            while not isterminated:

                state = self.env.get_state()  # 获取环境特征向量

                action = self.choose_action(state)  # 采样出动作，

                next_state, reward, isterminated = self.env.step(action)  # 根据当前状态和动作，获取即时奖励和是否终止信息

                # records['state'].append(state)
                # records['next_state'].append(next_state)
                # records['actions'].append([action])
                accreward = accreward+reward
                count += 1
                # records['r'].append([reward])
                # records['isterminated'].append([isterminated])  # 将获取的各种信息添加到记录里面
                
                # 将当前状态和动作添加到经验回放缓冲区中    
                self.replay_buffer.add_experience(state, action, reward, next_state, isterminated)
                
            print('episode:', i, ' accreward:', accreward, 'reward_per_step:', accreward/(count+1e-6))
            accrewards.append(accreward)
            rewards_per_steps.append(accreward/(count+1e-6))
            # 将记录的经验数据转换为 NumPy 数组格式，以便存储到经验回放缓冲区中
            # records = {key: np.array(records[key]) for key in records.keys()}
            # # 把采样出来的经验存储到replay_buffer缓存（经验回访缓冲区）
            # self.replay_buffer.store_records(records)
            # episilon每一轮都要减少一个小数值，在每一轮训练中逐渐减小 ε（epsilon）值，即探索率。ε是在DQN中用于控制探索和利用之间的平衡的重要参数。通过逐渐减小 ε，模型在训练的早期会更多地进行探索，随着训练的进行，模型会更多地利用已经学到的知识。
            self.episilon = max(self.episilon-0.004, 0.0003)

            # 每10个episode进行一次训练
            if i % 10 == 0:
                for step in range(self.n_steps_update):  # 用于每轮训练的核心部分
                    self.eval_q_net.train()  # 将Q网络设置为训练模式（反向传播时会计算梯度），以便在下一轮训练中更新其参数
                    self.target_q_net.eval()  # 将目标 Q 网络设置为评估模式，以确保在训练过程中不会更新其参数，保持其稳定性。
                    records = self.replay_buffer.sample(self.batch_size)  # 从经验回放缓冲区中随机抽样一批经验数据，大小为batch_size

                    r = torch.FloatTensor(np.array(records['rewards'])) # 将即时奖励转换为PyTorch的张量格式

                    # 将当前状态转换为PyTorch的张量格式
                    state = torch.FloatTensor(np.array(records['states']))

                    # 将下一个状态转换为PyTorch的张量格式
                    next_state = torch.FloatTensor(np.array(records['next_states']))
                    q_value = self.eval_q_net(state)  # 使用Q网络计算当前状态的Q值

                    # 将动作转换为PyTorch的张量格式
                    actions = torch.LongTensor(np.array(records['actions'])).unsqueeze(dim=-1).unsqueeze(dim=-1)
                    isterminated = torch.BoolTensor(np.array(records['terminals']))
                    
                    q_value = q_value.gather(1, actions)  # 根据当前状态和动作获取的q值

                    target = r+self.gamma*torch.max(self.target_q_net(next_state), 1, keepdims=True)[
                        0]*(1-torch.tensor(isterminated).float())  # 根据即时奖励和下个状态下做出最优动作后的q值得到的目标q值

                    # dqn的拟合q值的损失函数 目标值减去q估计值的平方取平均值来确定loss函数
                    loss = torch.mean((target.detach()-q_value)**2)

                    self.optimizer.zero_grad()
                    loss.backward()  # 反向传播，计算梯度
                    self.optimizer.step()  # 优化器根据梯度更新网络参数

                    step_ = step_+1  # 更新步数计数器

                    if step_ % self.replace_steps_cycle == 0:  # 判断是否到了更新目标Q网络的周期 是否走完了c steps

                        self.target_q_net.load_state_dict(
                            self.eval_q_net.state_dict())  # 更新目标Q网络的参数

                self.save_model(i)  # 保存模型的参数

    # def save_optimevalue(self):  

    #     self.env.reset()  # 重置环境，以便开始新的 episode

    #     accreward = 0  # 初始化累积奖励为 0
    #     isterminated = False  # 初始化是否终止标志为 False

    #     while not isterminated:

    #         state = self.env.get_state()  # 获取当前环境的状态

    #         # 根据当前状态选择动作，这里的episilon设置为0，表示完全按照当前策略选择动作（训练时需要探索，测试时直接取1）
    #         action = self.choose_action(state, 0)

    #         reward, isterminated = self.env.step(
    #             action)  # 执行选定的动作，获取即时奖励和是否终止的信息
    #     # 使用训练好的dqn模型来采样一个episode，并且记录下最后的零件组装顺序到res，这个就是dqn产生的初始解，保存到文件中
    #     # 将最终的零件组装顺序保存到变量res中，self.env.stepedparts保存了整个episode中的动作序列
    #     res = np.array(self.env.stepedparts)
    #     # 将最终的零件组装顺序保存到文件中，文件名以原始文件名加上 '_dqnvalue.npy' 结尾，以区分其他文件
    #     np.save(self.env.step_filename+'_dqnvalue.npy', res)


if __name__ == '__main__':
    train_dir = '/home/wangc/Documents/rl4occ/data/train'
    step_filenames = [os.path.join(train_dir, path) for path in os.listdir(train_dir)]
    env = Env(step_filenames)
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
    
    episode_nums = 2000  # 定义了训练的总回合数

    dqn_agent.learn(episode_nums)
    dqn_agent.load_model()  # 加载训练好的dqn模型
    dqn_agent.save_optimevalue()  # 保存好得到的dqn初始解
