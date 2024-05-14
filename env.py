#-- coding:UTF-8 --
import collections
import gc
import os
import pickle
import random
import sys
import copy
from typing import List, Dict
import gym
import numpy as np
from assembly import OCCAssembly
from memory_profiler import profile

class Env(gym.Env):  # 定义一个名为Env的类，表示装配体的环境

    def __init__(self, step_files, pickle_dataset):  # 定义环境
        print("---初始化环境---")
        self.step_filenames = step_files
        self.pickle_dataset = pickle_dataset
        # self.reset()

    def get_state(self):  # 将装配体的状态转换成一个适合神经网络处理的向量形式
        """状态空间"""
        states = collections.nametuple("src", "tgt", "mask")
        # d_model = 10
        src_inputs = np.array([
            self.assembly.boom_transform[id].direction+\
            [self.assembly.bboxes[id].CornerMin().X(), 
            self.assembly.bboxes[id].CornerMax().X(), 
            self.assembly.bboxes[id].CornerMin().Y(), 
            self.assembly.bboxes[id].CornerMax().Y(),
            self.assembly.bboxes[id].CornerMin().Z(), 
            self.assembly.bboxes[id].CornerMax().Z()]+\
            [id]
            for id in self.allparts
        ], dtype=np.float32)
        decoder_mask = np.zeros(len(self.allparts), dtype=np.float32)
        decoder_mask[self.stepedparts] = 1
        decoder_mask.shape = (len(self.allparts), 1)
        
        # if self.stepedparts == []:
            # src_inputs = np.random.random(src_inputs.shape)
            # decoder_mask = np.zeros((len(self.allparts),1), dtype=np.float32)
        # tgt_inputs = copy.deepcopy(src_inputs)
        # tgt_inputs[self.stepedparts] = float(1e-9)    
        return np.hstack((src_inputs, decoder_mask), dtype=np.float32)
        # return np.hstack((src_inputs, tgt_inputs), dtype=np.float32)  

    def comp_fit(self, one_path):
        """奖励函数"""
        # print(one_path)
        if len(one_path) == 0:
            return 0
        interference_count = 0
        for i in range(len(one_path)):
            for j in range(i+1, len(one_path)):  # 利用提前计算好的邻居两量碰撞次数来统计出总的碰撞次数

                interference_count = interference_count + self.assembly.countij[one_path[j], one_path[i]]
                # interference_count = interference_count + self.assembly.get_ijcount(i,j)

        # 统计装配方向改变次数
        pre_direction = self.assembly.boom_transform[one_path[0]].sign
        direction_change_num = 0
        for sign in range(1, len(one_path)):
            dir = self.assembly.boom_transform[one_path[sign]].sign
            if dir != pre_direction:
                direction_change_num += 1
                # print(dir, pre_direction)
            pre_direction = dir

        return interference_count * 50 + 10 * direction_change_num

    def reset(self, seed=23):  # 环境重置，
        print("---环境重置，随机选择新的装配体---")
        # step_filename = random.choice(self.step_filenames)
        # pickle_data = random.choice(self.pickle_dataset)
        # with open(pickle_data, 'rb') as f:
        #     self.assembly = pickle.load(f)
        self.assembly = random.choice(self.pickle_dataset)
        # self.assembly = OCCAssembly(step_filename)  # 读取模型文件
        # self.step_filename = step_filename
        # self.assembly.create_boom()  # 创建装配模型的爆炸视图，用于显示零件的装配顺序。
        self.part_num = self.assembly.part_num # 获取装配模型中的零件数量
        print("零件数量：", self.part_num)
        # self.assembly.compute_countij()  # 提前计算每个零件排在在某个其他零件后发生碰撞次数，为了加速?
        # self.n_state = self.part_num*2*9
        # self.n_actions = self.part_num  # 将动作空间的大小设置为零件的数量，表示每个动作是选择一个零件进行装配
        self.allparts = list(range(self.part_num))  # 所有零件的编号
        self.stepedparts = []  # 已装配的
        self.unstepparts = list(range(self.part_num))   # 未装配的
        self.maxabsx = max([abs(self.assembly.boom_transform[i].direction[0])
                        for i in range(self.part_num)])+1e-3

        self.maxabsy = max([abs(self.assembly.boom_transform[i].direction[1])
                        for i in range(self.part_num)])+1e-3

        self.maxabsz = max([abs(self.assembly.boom_transform[i].direction[2])
                        for i in range(self.part_num)])+1e-3
        # 计算出每个零件的外围包裹立方体的顶点的xyz的坐标的绝对值最大值，用于后面做状态特征的归一化
        self.maxabscornnerx = max([abs(self.assembly.bboxes[i].CornerMax().X()) for i in range(
            self.part_num)]+[abs(self.assembly.bboxes[i].CornerMin().X()) for i in range(self.part_num)])+1e-3

        self.maxabscornnery = max([abs(self.assembly.bboxes[i].CornerMax().Y()) for i in range(
            self.part_num)]+[abs(self.assembly.bboxes[i].CornerMin().Y()) for i in range(self.part_num)])+1e-3

        self.maxabscornnerz = max([abs(self.assembly.bboxes[i].CornerMax().Z()) for i in range(
            self.part_num)]+[abs(self.assembly.bboxes[i].CornerMin().Z()) for i in range(self.part_num)])+1e-3
        
        return self.get_state()
        
    def step(self, action: int):
        """_summary_

        Args:
            action (int): pick one element from unstepparts

        Returns:
            _type_: int
        """
        print(f"---执行动作, 选取零件{action}---")
        if action>=self.part_num:
            raise ValueError(f"action {action} out of range {self.part_num}")
        if action in self.stepedparts:
            raise ValueError(f"action {action} already steped")
        if action not in self.unstepparts:
            raise ValueError(f"action {action} not in unstepparts {self.unstepparts}")
        # if action < len(self.unstepparts):
        
        f1 = self.comp_fit(self.stepedparts)

        # if action < len(self.stepedparts):
        self.unstepparts.remove(action)
        self.stepedparts.append(action)

        # else:
        #     self.stepedparts.append(id)

        f2 = self.comp_fit(self.stepedparts)

        reward = f1-f2
        isterminated = False
        if self.unstepparts == []:
            isterminated = True

        return self.get_state(), reward, isterminated
    

