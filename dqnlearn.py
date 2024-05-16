# -- coding:UTF-8 --
import copy
import datetime
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from env import Env
from replay_buffer import ReplayBuffer
from attention_q_net import MultiHeadAttention, AttentionQNet
from pointer_network import PointerNet
from transformerQnet import TransformerQnet
from utils import pad_sequences_and_create_mask, pad_sequences

from memory_profiler import profile
from tqdm import trange


class DQNAgent:

    def __init__(self, env, buffer_size):  # 定义环境对象和经验回放缓冲区的大小
        print("---初始化dqn agent---")
        self.env = env  # 定义环境对象
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # 定义设备类型，如果GPU可用则使用GPU，否则使用CPU

        self.n_max_nodes = 30
        self.input_dim = 10
        self.d_model = 128
        self.nhead = 8
        self.num_encoder_layers = 6
        self.num_decoder_layers = 6
        self.dim_feedforward = 2048
        self.dropout = 0
        self.batch_first = True

        self.eval_q_net = TransformerQnet(
            n_max_nodes = self.n_max_nodes,
            input_dim=self.input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=self.batch_first,
            device=self.device
        )
        self.target_q_net = TransformerQnet(
            n_max_nodes=self.n_max_nodes,
            input_dim=self.input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=self.batch_first,
            device=self.device
        )

        self.target_q_net.load_state_dict(self.eval_q_net.state_dict())

        # 创建一个大小为buffer_size的经验回放缓冲区，用于存储智能体与环境交互的经验数据
        self.replay_buffer = ReplayBuffer(buffer_size)

        # 训练超参数
        self.n_steps_update = 1  # 定义每次训练时使用的步数
        self.batch_size = 64  # 定义每次训练时的批量大小
        # 使用Adam优化器来优化估计网络的参数，学习率为2e-4（α）。
        self.optimizer = torch.optim.Adam(self.eval_q_net.parameters(), lr=5e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=150, eta_min=0
        )
        self.update_cycle = 20
        self.replace_steps_cycle = 200  # 定义替换目标网络参数的周期步数

        self.init_episilon = 0.98
        self.final_episilon = 0.04

        self.gamma = (
            0.998  # 定义强化学习中的折扣因子，用于调节当前奖励和未来奖励的重要性
        )
        self.save_cycyle = 10  # 定义保存模型的周期步数

        self.step = 0  # 定义步数计数器
        self.episode = 0  # 定义轮数计数器
        self.datetime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.log = SummaryWriter(f"./logs/{self.datetime}")

    def update_episilon(self, step):
        return self.final_episilon + (self.init_episilon - self.final_episilon) * math.exp(-1. * step / 5000)
        # return 0

    def save_model(self, itr):  # 保存q估值网络
        if not os.path.exists(f"./model/{self.datetime}"):
            os.mkdir(f"./model/{self.datetime}")
        # os.mkdir(f'./model/{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        torch.save(self.eval_q_net, f"./model/{self.datetime}/eval_q_net.pth")

    def load_model(self):  # 加载深度Q学习算法中的Q值估计网络（q_net）便于继续训练和预测
        self.eval_q_net = torch.load("./model/best_eval_q_net.pth")

    # @profile(precision=4, stream=open("memory_profiler.log", "w+"))
    def choose_action(self, state):  # 在给定状态下选择动作的过程

        # 根据ε-greedy策略来选择动作，当随机数小于ε时，以一定概率选择随机动作
        if np.random.uniform(0, 1) < self.episilon:

            action = int(np.random.choice(self.env.unstepparts))  # 根据概率分布选择动作

        else:
            # src = torch.FloatTensor(state[:, :-1]).unsqueeze(0).to(self.device)
            # mask = state[:, -1]

            # tgt = src
            src = torch.FloatTensor(state.src).unsqueeze(0).to(self.device)
            tgt = torch.FloatTensor(state.tgt).unsqueeze(0).to(self.device)
            mask = torch.BoolTensor(state.mask).unsqueeze(0).to(self.device)

            # tgt_mask = (
            #     torch.BoolTensor(mask.reshape(1, -1).repeat(len(state[:, -1]), 0))
            #     .unsqueeze(0)
            #     .expand(self.nhead, -1, -1)
            #     .to(self.device)
            # )  # 创建 atten mask
            # tgt_padding_mask = torch.BoolTensor(mask).reshape(1, -1).to(self.device)
            self.eval_q_net.eval()  # 将Q网络设置为评估模式，确保在选择动作时不会更新其参数
            with torch.no_grad():
                qvals = self.eval_q_net(
                    src=src, tgt=tgt, mask=mask
                )  # 使用Q网络预测当前状态下各个动作的Q值
                action = np.argmax(qvals.detach().cpu().numpy())

        return action  # 返回选择的动作

    # @profile(precision=4, stream=open("memory_profiler.log", "w+"))
    def update(self, i):
        mean_loss = 0
        for update_step in range(self.n_steps_update):  # 用于每轮训练的核心部分
            records = self.replay_buffer.sample(
                self.batch_size
            )  # 从经验回放缓冲区中随机抽样一批经验数据，大小为batch_size

            r = (
                torch.FloatTensor(np.array(records["reward"]))
                .unsqueeze(-1)
                .to(self.device)
            )  # batch_size x seq_len
            # **************************************** State **************************************** #
            max_src_seq = np.max([len(state.src) for state in records["state"]])
            max_tgt_seq = np.max([len(state.tgt) for state in records["state"]])
            # print("max_seq:", max_seq)
            state_src = [state.src for state in records["state"]]
            state_tgt = [state.tgt for state in records["state"]]
            state_mask = [state.mask for state in records["state"]]
            # state_tgt_mask = [state.mask for state in records["state"]]
            (
                state_src,
                state_tgt,
                state_mask,
            ) = pad_sequences(
                state_src, state_tgt, state_mask, self.n_max_nodes, batch_size=self.batch_size
            )  # batch_size x seq_len
            state_mask = torch.BoolTensor(state_mask).to(self.device)
            state_src = torch.FloatTensor(state_src).to(self.device)
            state_tgt = torch.FloatTensor(state_tgt).to(self.device)
            # state_src_mask = (
            #     state_src[:, :, 0]
            #     .clone()
            #     .detach()
            #     .unsqueeze(0)
            #     .unsqueeze(-2)
            #     .expand(self.nhead, -1, self.n_max_nodes, -1)
            #     .permute(1, 0, 2, 3)
            #     .reshape(-1, self.n_max_nodes, self.n_max_nodes)
            #     == -1e4
            # )  # batch_size*nhead x seq_len x seq_len
            state_src_padding_mask = (state_src[:, :, 0].clone().detach() == -1e4)    # batch_size x seq_len
            # state_tgt_mask = (
            #     state_tgt[:, :, 0]
            #     .clone()
            #     .detach()
            #     .unsqueeze(0)
            #     .unsqueeze(-2)
            #     .expand(self.nhead, -1, self.n_max_nodes, -1)
            #     .permute(1, 0, 2, 3)
            #     .reshape(-1, self.n_max_nodes, self.n_max_nodes)
            #     == -1e4
            # ) # batch_size*nhead x seq_len x seq_len
            state_tgt_padding_mask = (state_tgt[:, :, 0].clone().detach() == -1e4)    # batch_size x seq_len
            # state_tgt_memory_mask = None
            state_tgt_memory_padding_mask = (state_tgt[:, :, 0].clone().detach() == -1e4)   # batch_size x seq_len
            ################################################################
            max_src_seq = np.max([len(next_state.src) for next_state in records["next_state"]])
            max_tgt_seq = np.max([len(next_state.tgt) for next_state in records["next_state"]])
            # print("max_seq:", max_seq)
            next_state_src = [next_state.src for next_state in records["next_state"]]
            next_state_tgt = [next_state.tgt for next_state in records["next_state"]]
            next_state_mask = [next_state.mask for next_state in records["next_state"]]
            # next_state_tgt_mask = [next_state.mask for next_state in records["next_state"]]
            (
                next_state_src,
                next_state_tgt,
                next_state_mask,
            ) = pad_sequences(
                next_state_src, next_state_tgt, next_state_mask, self.n_max_nodes, batch_size=self.batch_size
            )  # batch_size x seq_len
            next_state_mask = torch.BoolTensor(next_state_mask).to(self.device)
            next_state_src = torch.FloatTensor(next_state_src).to(self.device)
            next_state_tgt = torch.FloatTensor(next_state_tgt).to(self.device)
            # next_state_src_mask = (
            #     next_state_src[:, :, 0]
            #     .clone()
            #     .detach()
            #     .unsqueeze(0)
            #     .unsqueeze(-2)
            #     .expand(self.nhead, -1, self.n_max_nodes, -1)
            #     .permute(1, 0, 2, 3)
            #     .reshape(-1, self.n_max_nodes, self.n_max_nodes)
            #     == -1e4
            # ) # batch_size*nhead x seq_len x seq_len
            next_state_src_padding_mask = (next_state_src[:, :, 0].clone().detach() == -1e4)    # batch_size x seq_len
            # next_state_tgt_mask = (
            #     next_state_tgt[:, :, 0]
            #     .clone()
            #     .detach()
            #     .unsqueeze(0)
            #     .unsqueeze(-2)
            #     .expand(self.nhead, -1, self.n_max_nodes, -1)
            #     .permute(1, 0, 2, 3)
            #     .reshape(-1, self.n_max_nodes, self.n_max_nodes)
            #     == -1e4
            # )  # batch_size*nhead x seq_len x seq_len
            next_state_tgt_padding_mask = (next_state_tgt[:, :, 0].clone().detach() == -1e4)   # batch_size x seq_len
            # next_state_tgt_memory_mask = None
            next_state_tgt_memory_padding_mask = (next_state_tgt[:, :, 0].clone().detach() == -1e4)   # batch_size x seq_len
            # ***************************************** Next State ***************************************** #
            actions = (
                torch.LongTensor(np.array(records["action"]))
                .unsqueeze(dim=-1)
                .to(self.device)
            )  # batch_size x 1
            isterminated = (
                torch.BoolTensor(np.array(records["terminal"]))
                .unsqueeze(dim=-1)
                .to(self.device)
            )  # batch_size x 1

            q_value = self.eval_q_net(
                src=state_src,
                tgt=state_tgt,
                memory_key_padding_mask = state_tgt_memory_padding_mask,
                src_key_padding_mask=state_src_padding_mask,
                tgt_key_padding_mask=state_tgt_padding_mask,
                mask = state_mask,
            )  # batch_size x seq_len
            q_value = q_value.gather(
                -1, actions
            )  # 根据当前状态和动作获取的q值
            self.log.add_scalar("training/q_val", q_value.detach().mean().cpu(), i)
            # DDQN
            with torch.no_grad():
                next_q_value = self.eval_q_net(
                    src=next_state_src,
                    tgt=next_state_tgt,
                    memory_key_padding_mask = next_state_tgt_memory_padding_mask,
                    src_key_padding_mask=next_state_src_padding_mask,
                    tgt_key_padding_mask=next_state_tgt_padding_mask,
                    mask = next_state_mask
                )  # 获取下一个状态下的q值
                max_action_id = torch.argmax(
                    next_q_value, dim=-1, keepdim=True
                )  # batch_size x 1
                target_q_value = self.target_q_net(
                    src=next_state_src,
                    tgt=next_state_tgt,
                    memory_key_padding_mask = next_state_tgt_memory_padding_mask,
                    src_key_padding_mask=next_state_src_padding_mask,
                    tgt_key_padding_mask=next_state_tgt_padding_mask,
                    mask = next_state_mask
                )
                target = r + self.gamma * target_q_value.gather(-1, max_action_id) * (
                    ~isterminated
                )  # 根据即时奖励和下个状态下做出最优动作后的q值得到的目标q值 batch_size x 1
                
                self.log.add_scalar("training/target", target.detach().mean().cpu(), i)
            # dqn的拟合q值的损失函数 目标值减去q估计值的平方取平均值来确定loss函数
            # loss = torch.mean((target.detach()-q_value)**2)
            criterion = nn.MSELoss()
            loss = criterion(q_value, target.detach())

            self.optimizer.zero_grad()
            loss.backward()  # 反向传播，计算梯度
            torch.nn.utils.clip_grad_norm_(parameters=self.eval_q_net.parameters(), max_norm=10, norm_type=2)
            self.optimizer.step()  # 优化器根据梯度更新网络参数

            # step_ = step_+1  # 更新步数计数器
            # print(f"episode{i}-{update_step}:{loss}")
            mean_loss += loss.item() / self.n_steps_update
        self.log.add_scalar("training/loss", loss.item(), i)
        # if i % self.replace_steps_cycle == 0:  # 判断是否到了更新目标Q网络的周期 是否走完了c steps

        #     self.target_q_net.load_state_dict(
        #         self.eval_q_net.state_dict()
        #     )  # 更新目标Q网络的参数

        #     self.save_model(i)  # 保存模型的参数
        w = 0.0001
        for target_param, evaluation_param in zip(self.target_q_net.parameters(), self.eval_q_net.parameters()):
            target_param.data.copy_(w * evaluation_param.data + (1 - w) * target_param.data)

    def learn(self, episode_nums):
        accrewards = []  # 创建一个空列表 accrewards，用于存储每轮训练的累积奖励
        rewards_per_steps = (
            []
        )  # 创建一个空列表 rewards_per_steps，用于存储每轮训练的奖励/步数

        # self.step = 0  # 初始化步数计数器 step_ 为0
        for i in trange(episode_nums):  # 循环执行训练指定次数 episode_nums
            self.episilon = self.update_episilon(i)
            self.episode += 1  # 更新轮数计数器
            # records = {'state': [], 'next_state': [], 'actions': [], 'r': [
            # ], 'isterminated': []}  # 创建一个字典 records，用于存储每个 episode 中的经验数据

            self.env.reset()

            accreward = 0
            count = 0
            isterminated = False

            while not isterminated:

                state = self.env.get_state()  # 获取环境特征向量

                action = self.choose_action(state)  # 采样出动作，
                # self.log.add_scalar('action', action, i)

                next_state, reward, isterminated = self.env.step(
                    action
                )  # 根据当前状态和动作，获取即时奖励和是否终止信息

                # records['state'].append(state)
                # records['next_state'].append(next_state)
                # records['actions'].append([action])
                accreward = accreward + reward
                count += 1
                self.step += 1
                # records['r'].append([reward])
                # records['isterminated'].append([isterminated])  # 将获取的各种信息添加到记录里面

                # 将当前状态和动作添加到经验回放缓冲区中
                self.replay_buffer.add_experience(
                    state, action, reward, next_state, isterminated
                )

            # print(
            #     "episode:",
            #     i,
            #     " accreward:",
            #     accreward,
            #     "reward_per_step:",
            #     accreward / (count + 1e-6),
            # )
            self.log.add_scalar("training/exploration_rate", self.episilon, i)
            self.log.add_scalar(
                "training/reward_per_episode", accreward / (count + 1e-6), i
            )
            self.log.add_scalar(
                "experience_replay_buffer_size", len(self.replay_buffer), i
            )
            # self.log.add_scalar('buffer_mb', self.replay_buffer.size(), i)
            accrewards.append(accreward)
            rewards_per_steps.append(accreward / (count + 1e-6))
            # 将记录的经验数据转换为 NumPy 数组格式，以便存储到经验回放缓冲区中
            # records = {key: np.array(records[key]) for key in records.keys()}
            # # 把采样出来的经验存储到replay_buffer缓存（经验回访缓冲区）
            # self.replay_buffer.store_records(records)
            # episilon每一轮都要减少一个小数值，在每一轮训练中逐渐减小 ε（epsilon）值，即探索率。ε是在DQN中用于控制探索和利用之间的平衡的重要参数。通过逐渐减小 ε，模型在训练的早期会更多地进行探索，随着训练的进行，模型会更多地利用已经学到的知识。

            # 每20个episode进行一次训练
            if i % self.update_cycle == 0 and self.replay_buffer.can_sample(
                self.batch_size
            ):
                self.update(i)

