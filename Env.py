import os
import random
from assembly import OCCAssembly
import gym
import numpy as np


class Env(gym.Env):  # 定义一个名为Env的类，表示装配体的环境

    def __init__(self, step_files):  # 定义环境
        self.step_filenames = step_files

    def get_state(self):  # 将装配体的状态转换成一个适合神经网络处理的向量形式
        """状态空间"""

        unsteppartstate = [self.assembly.boom_transform[id].direction+\
                           [self.assembly.bboxes[id].CornerMin().X(), 
                            self.assembly.bboxes[id].CornerMax().X(), 
                            self.assembly.bboxes[id].CornerMin().Y(), 
                            self.assembly.bboxes[id].CornerMax().Y(),
                            self.assembly.bboxes[id].CornerMin().Z(), 
                            self.assembly.bboxes[id].CornerMax().Z()] for id in self.unstepparts]  # 获取每个未装配好零件的装配方向向量以及包裹立方体的顶点的坐标的最大最小值

        stepedpartstate = [self.assembly.boom_transform[id].direction+\
                           [self.assembly.bboxes[id].CornerMin().X(), 
                            self.assembly.bboxes[id].CornerMax().X(), 
                            self.assembly.bboxes[id].CornerMin().Y(), 
                            self.assembly.bboxes[id].CornerMax().Y(),
                            self.assembly.bboxes[id].CornerMin().Z(), 
                            self.assembly.bboxes[id].CornerMax().Z()] for id in self.stepedparts]

        unsteppartstate = unsteppartstate+[[0]*9]*(self.part_num-len(unsteppartstate))
        stepedpartstate = stepedpartstate+[[0]*9]*(self.part_num-len(stepedpartstate))
        state = unsteppartstate+stepedpartstate
        state = np.array(state)
        state[:, 0] = state[:, 0]/self.maxabsx
        state[:, 1] = state[:, 1]/self.maxabsy
        state[:, 2] = state[:, 2]/self.maxabsz
        state[:, 3:5] = state[:, 3:5]/self.maxabscornnerx  # 外围包裹顶点坐标特征归一化到-1到1之间
        state[:, 5:7] = state[:, 5:7]/self.maxabscornnery
        # 对状态向量进行归一化处理，将各个特征的值归一化到 [-1, 1] 的范围内，以便神经网络更好地处理。
        state[:, 7:9] = state[:, 7:9]/self.maxabscornnerz
        state = np.reshape(state, (-1))  # 将状态向量展开成一维数组，并返回该数组作为当前环境的状态

        return state

    def comp_fit(self, one_path):
        """奖励函数"""
        # print(one_path)
        if len(one_path) == 0:
            return 0
        interference_count = 0
        for i in range(len(one_path)):
            for j in range(i+1, len(one_path)):  # 利用提前计算好的邻居两量碰撞次数来统计出总的碰撞次数

                interference_count = interference_count + \
                    self.assembly.countij[one_path[j], one_path[i]]

        # 统计装配方向改变次数
        pre_direction = self.assembly.boom_transform[one_path[0]].sign
        direction_change_num = 0
        for sign in range(1, len(one_path)):
            dir = self.assembly.boom_transform[one_path[sign]].sign
            if dir != pre_direction:
                direction_change_num += 1
                # print(dir, pre_direction)
            pre_direction = dir

        return interference_count * 5 + 5 * direction_change_num

    def reset(self, seed=23):  # 环境重置，
        print("---环境重置，随机选择新的装配体---")
        step_filename = random.choice(self.step_filenames)
        
        self.assembly = OCCAssembly(step_filename)  # 读取模型文件
        self.step_filename = step_filename
        self.assembly.create_boom()  # 创建装配模型的爆炸视图，用于显示零件的装配顺序。
        self.part_num = self.assembly.get_part_num()  # 获取装配模型中的零件数量
        self.assembly.compute_countij()  # 提前计算每个零件排在在某个其他零件后发生碰撞次数，为了加速
        self.n_state = self.part_num*2*9
        self.n_actions = self.part_num  # 将动作空间的大小设置为零件的数量，表示每个动作是选择一个零件进行装配
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
        

    def step(self, action):

        f1 = self.comp_fit(self.stepedparts)
        id = self.unstepparts.pop(0)

        if action < len(self.stepedparts):

            self.stepedparts = self.stepedparts[:action] + \
                [id]+self.stepedparts[action:]

        else:
            self.stepedparts.append(id)

        f2 = self.comp_fit(self.stepedparts)

        reward = f1-f2
        isterminated = False
        if self.unstepparts == []:
            isterminated = True

        return reward, isterminated
    

if __name__ == '__main__':
    train_dir = '/home/wangc/Documents/rl4occ/data/train'
    step_filenames = [os.path.join(train_dir, path) for path in os.listdir(train_dir)]
    env = Env(step_filenames)
    next_states = env.reset()
    print('debug!')
