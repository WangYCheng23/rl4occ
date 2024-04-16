
import numpy as np
import torch
import torch.nn as nn

from Env import Env
from mytool import Replay_Buffer


class QNet(nn.Module):  # 定义q值网络，定义了一个名为QNet的类，该类继承自nn.Module类，表示这个QNet类是一个PyTorch模型，用于构建神经网络模型

    # 定义了QNet类的初始化方法，该方法接收三个参数：inputdim表示输入的特征维度，hiddendim表示隐藏层的神经元数量，outputdim表示输出的维度（通常对应着动作的数量）
    def __init__(self, inputdim, hiddendim, outputdim):
        super(QNet, self).__init__()  # 调用了父类nn.Module的初始化方法，确保正确地初始化神经网络模型
        # 定义了一个全连接层linear1，将输入特征的维度映射到隐藏层的维度。这里使用了nn.Linear类，表示线性变换
        self.linear1 = nn.Linear(inputdim, hiddendim)
        # 定义了另一个全连接层linear2，将隐藏层的维度映射到输出的维度。同样使用了nn.Linear类，表示线性变换。
        self.linear2 = nn.Linear(hiddendim, outputdim)

    def forward(self, x):  # 定义了QNet类的前向传播方法forward，该方法接收输入x并返回输出。在PyTorch中，所有的模型都必须实现这个方法，用于定义模型的计算流程

        # 对输入x先经过第一个全连接层linear1进行线性变换，然后通过torch.nn.Tanh()函数进行激活，将激活后的结果作为下一层的输入
        x = torch.nn.Tanh()(self.linear1(x))
        x = self.linear2(x)  # 将经过Tanh激活后的结果再经过第二个全连接层linear2进行线性变换，得到最终的输出

        return x  # 返回经过两层全连接层处理后的最终输出结果


class DQNAgent:

    def __init__(self, env, size):  # 定义环境对象和经验回放缓冲区的大小

        self.env = env  # 定义环境对象
        self.n_actions = self.env.n_actions  # 定义动作数量
        self.n_state = self.env.n_state  # 定义状态数量
        self.q_net = QNet(self.n_state, 20, self.n_actions)  # 定义q值的估计网络
        self.target_q_net = QNet(self.n_state, 20, self.n_actions)  # 定义q值的目标网络
        # 目标网络和估值网络权重一开始相同，为了在深度 Q 学习算法中稳定训练和提高效率
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        # 创建一个大小为size的经验回放缓冲区，用于存储智能体与环境交互的经验数据
        self.replay_buffer = Replay_Buffer(size, self.n_state)
        self.n_steps = 10  # 定义每次训练时使用的步数
        self.batch_size = 80  # 定义每次训练时的批量大小
        # 使用Adam优化器来优化估计网络的参数，学习率为2e-4（α）。
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=2e-4)
        self.replace_steps_cycle = 60  # 定义替换目标网络参数的周期步数
        self.episilon = 0.76  # 定义ε贪婪策略中的ε值
        self.gamma = 0.998  # 定义强化学习中的折扣因子，用于调节当前奖励和未来奖励的重要性
        self.save_cycyle = 10  # 定义保存模型的周期步数

    def set_episilon(self):
        self.episilon = 0.76  # 用于ε贪婪策略，用于在探索和利用之间进行权衡

    def save_model(self):  # 保存q估值网络
        torch.save(self.q_net, 'q_net.pth')

    def load_model(self):  # 加载深度Q学习算法中的Q值估计网络（q_net）便于继续训练和预测
        self.q_net = torch.load('q_net.pth')

    def learn(self, episode_nums):
        accrewards = []  # 创建一个空列表 accrewards，用于存储每轮训练的累积奖励

        step_ = 0  # 初始化步数计数器 step_ 为0
        self.set_episilon()  # 设置初始的ε-贪心策略的ε值
        for i in range(episode_nums):  # 循环执行训练指定次数 episode_nums

            records = {'state': [], 'next_state': [], 'actions': [], 'r': [
            ], 'isterminated': []}  # 创建一个字典 records，用于存储每个 episode 中的经验数据

            self.env.reset()

            accreward = 0
            isterminated = False

            while not isterminated:

                state = self.env.get_state()  # 获取环境特征向量

                action = self.choose_action(state, self.episilon)  # 采样出动作，

                reward, isterminated = self.env.step(
                    action)  # 根据当前状态和动作，获取即时奖励和是否终止信息

                next_state = self.env.get_state()  # 获取做出动作后的下一步状态
                records['state'].append(state)
                records['next_state'].append(next_state)
                records['actions'].append([action])
                accreward = accreward+reward
                records['r'].append([reward])
                records['isterminated'].append(
                    [isterminated])  # 将获取的各种信息添加到记录里面

            print('episode:', i, ' accreward:', accreward)
            accrewards.append(accreward)
            # 将记录的经验数据转换为 NumPy 数组格式，以便存储到经验回放缓冲区中
            records = {key: np.array(records[key]) for key in records.keys()}
            # 把采样出来的经验存储到replay_buffer缓存（经验回访缓冲区）
            self.replay_buffer.store_records(records)
            # episilon每一轮都要减少一个小数值，在每一轮训练中逐渐减小 ε（epsilon）值，即探索率。ε是在DQN中用于控制探索和利用之间的平衡的重要参数。通过逐渐减小 ε，模型在训练的早期会更多地进行探索，随着训练的进行，模型会更多地利用已经学到的知识。
            self.episilon = max(self.episilon-0.004, 0.0003)
            self.q_net.train()  # 将Q网络设置为训练模式（反向传播时会计算梯度），以便在下一轮训练中更新其参数
            self.target_q_net.eval()  # 将目标 Q 网络设置为评估模式，以确保在训练过程中不会更新其参数，保持其稳定性。
            for step in range(self.n_steps):  # 用于每轮训练的核心部分

                records = self.replay_buffer.sample_records(
                    self.batch_size)  # 从经验回放缓冲区中随机抽样一批经验数据，大小为batch_size

                r = torch.tensor(records['r']).float()  # 将即时奖励转换为PyTorch的张量格式

                # 将当前状态转换为PyTorch的张量格式
                state = torch.tensor(records['state']).float()

                # 将下一个状态转换为PyTorch的张量格式
                next_state = torch.tensor(records['next_state']).float()
                q_value = self.q_net(state)  # 使用Q网络计算当前状态的Q值

                # 将动作转换为PyTorch的张量格式
                actions = torch.tensor(records['actions']).long()
                isterminated = records['isterminated']
                q_value = q_value.gather(-1, actions)  # 根据当前状态和动作获取的q值

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
                        self.q_net.state_dict())  # 更新目标Q网络的参数

            self.save_model()  # 保存模型的参数

    def save_optimevalue(self):  # 保存DQN模型产生的初始解的零件组装顺序到文件中的方法

        self.env.reset()  # 重置环境，以便开始新的 episode

        accreward = 0  # 初始化累积奖励为 0
        isterminated = False  # 初始化是否终止标志为 False

        while not isterminated:

            state = self.env.get_state()  # 获取当前环境的状态

            # 根据当前状态选择动作，这里的episilon设置为0，表示完全按照当前策略选择动作（训练时需要探索，测试时直接取1）
            action = self.choose_action(state, 0)

            reward, isterminated = self.env.step(
                action)  # 执行选定的动作，获取即时奖励和是否终止的信息
        # 使用训练好的dqn模型来采样一个episode，并且记录下最后的零件组装顺序到res，这个就是dqn产生的初始解，保存到文件中
        # 将最终的零件组装顺序保存到变量res中，self.env.stepedparts保存了整个episode中的动作序列
        res = np.array(self.env.stepedparts)
        # 将最终的零件组装顺序保存到文件中，文件名以原始文件名加上 '_dqnvalue.npy' 结尾，以区分其他文件
        np.save(self.env.step_filename+'_dqnvalue.npy', res)

    def choose_action(self, state, episilon):  # 在给定状态下选择动作的过程

        # 定义了一个方法，接受当前状态和ε-greedy策略中的ε作为输入
        state = torch.tensor(state[None, :]).float()
        self.q_net.eval()  # 将Q网络设置为评估模式，确保在选择动作时不会更新其参数
        o = self.q_net(state)  # 使用Q网络预测当前状态下各个动作的Q值

        o = o.detach().cpu().numpy()[0, :]  # 将Q值张量转换为NumPy数组，以便后续处理

        # 根据ε-greedy策略来选择动作，当随机数小于ε时，以一定概率选择随机动作
        if np.random.uniform(0, 1) < episilon:

            maxactionid = len(self.env.stepedparts)  # 获取已经执行的动作数量，用于限制后续动作的选择
            p = np.ones([self.n_actions])  # 初始化动作概率数组，所有动作的概率都设为1
            if maxactionid+1 < self.n_actions:  # 如果已执行的动作数量加1小于总动作数量，说明还有未执行的动作
                p[maxactionid+1:] = 1 / \
                    (self.n_actions-maxactionid-1)  # 将未执行的动作的概率设为均匀分布

            p = p/np.sum(p)  # 将概率归一化，确保概率之和为1

            action = int(np.random.choice(
                self.n_actions, 1, p=p)[0])  # 根据概率分布选择动作

        else:  # 如果随机数大于ε，则以1-ε的概率选择最优动作
            action = np.argmax(o)  # 选择具有最大Q值的动作作为最优动作

        return action  # 返回选择的动作


if __name__ == '__main__':

    step_filename = 'assembly.step'  # 模型文件名字

    env = Env(step_filename)  # 初始化环境对象

    size = 3000  # 定义了经验回放缓冲区的大小
    dqn = DQNAgent(env, size)

    episode_nums = 2000  # 定义了训练的总回合数

    dqn.learn(episode_nums)
    dqn.load_model()  # 加载训练好的dqn模型
    dqn.save_optimevalue()  # 保存好得到的dqn初始解
