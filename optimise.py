# coding=utf-8
from math import floor
import numpy as np
import time
import matplotlib.pyplot as plt  # 导入所需要的库
from assembly import OCCAssembly
import time

import copy

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体


class Gena_TSP(object):

    def __init__(self,
                 assembly_model,
                 part_num,
                 maxgen=50,
                 size_pop=20,
                 cross_prob=0.80,
                 pmuta_prob=0.02,
                 select_prob=0.8):
        self.maxgen = maxgen  # 最大迭代次数
        self.size_pop = size_pop  # 群体个数
        self.cross_prob = cross_prob  # 交叉概率
        self.pmuta_prob = pmuta_prob  # 变异概率
        self.select_prob = select_prob  # 选择概率
        self.num = part_num
        # self.data = data  # 城市的左边数据
        # self.num = len(data)  # 城市个数 对应染色体长度
        # 距离矩阵n*n, 第[i,j]个元素表示城市i到j距离matrix_dis函数见下文
        # self.matrix_distance = self.matrix_dis()

        # 通过选择概率确定子代的选择个数
        self.select_num = max(floor(self.size_pop * self.select_prob + 0.5), 2)

        # 父代和子代群体的初始化（不直接用np.zeros是为了保证单个染色体的编码为整数，np.zeros对应的数据类型为浮点型）
        self.chrom = np.array([0] * self.size_pop * self.num).reshape(self.size_pop,
                                                                      self.num)  # 父 print(chrom.shape)(200, 14)
        self.sub_sel = np.array([0] * int(self.select_num) *
                                self.num).reshape(self.select_num, self.num)  # 子 (160, 14)

        # 存储群体中每个染色体的路径总长度，对应单个染色体的适应度就是其倒数  #print(fitness.shape)#(200,)
        self.fitness = np.zeros(self.size_pop)

        self.best_fit = []  # 最优距离
        self.best_path = []  # 最优路径
        self.best_direction = []
        self.direction_list = []
        self.direction = []
        self.assembly_model = assembly_model
        self.assembly_model.create_boom()
        self.assembly_model.compute_countij()
        self.assembly_model.checking_num = 0

    # # 计算城市间的距离函数  n*n, 第[i,j]个元素表示城市i到j距离
    # def matrix_dis(self):
    #     res = np.zeros((self.num, self.num))
    #     for i in range(self.num):
    #         for j in range(i + 1, self.num):
    #             res[i, j] = np.linalg.norm(self.data[i, :] - self.data[j, :])  # 求二阶范数 就是距离公式
    #             res[j, i] = res[i, j]
    #     return res

    # 随机产生初始化群体函数
    def rand_chrom(self, initgene):
        # rand_ch = np.array(range(self.num))  ## num 零件个数 对应染色体长度  =14
        self.chrom = initgene
        for i in range(self.size_pop):  # size_pop  # 群体个数 200

            self.fitness[i] = self.comp_fit_(self.chrom[i, :])

    # 新的计算适应度值的函数，旧的运算太久
    def comp_fit_(self, one_path):
        self.assembly_model.checking_num = self.assembly_model.checking_num+2*self.num
        interference_count = 0
        for i in range(len(one_path)):
            for j in range(i+1, len(one_path)):

                interference_count = interference_count + \
                    self.assembly_model.countij[one_path[j], one_path[i]]

        pre_direction = self.assembly_model.boom_transform[one_path[0]].sign
        direction_change_num = 0
        for sign in range(1, len(one_path)):
            dir = self.assembly_model.boom_transform[one_path[sign]].sign
            if dir != pre_direction:
                direction_change_num += 1
                # print(dir, pre_direction)
            pre_direction = dir

        self.direction = [
            self.assembly_model.boom_transform[_].sign for _ in one_path]
        return interference_count * 5 + 5 * direction_change_num

    # 评价函数计算

    def comp_fit(self, one_path):

        # print(one_path)
        interference_count, self.direction = self.assembly_model.transform_parts(
            one_path)
        pre_direction = self.direction[0]
        direction_change_num = 0
        for sign in range(1, len(self.direction)):
            dir = self.direction[sign]
            if dir != pre_direction:
                direction_change_num += 1
                # print(dir, pre_direction)
            pre_direction = dir
        # print(self.direction)

        res = interference_count * 5 + 5 * direction_change_num
        # print(res, interference_count, direction_change_num)
        # for i in range(self.num - 1):
        #     res += self.matrix_distance[one_path[i], one_path[i + 1]]  # matrix_distance n*n, 第[i,j]个元素表示城市i到j距离
        # res += self.matrix_distance[one_path[-1], one_path[0]]  # 最后一个城市 到起点距离

        return res

    # 路径可视化函数
    def out_path(self, one_path, direction):
        res = ''

        for i in range(self.num):
            res += '( ' + str(one_path[i] + 1) + \
                ' , ' + direction[i] + ' )  , '
        # res += str(one_path[0] + 1) + '\n'
        print(res)
        return res

    # 子代选取，根据选中概率与对应的适应度函数，采用随机遍历选择方法
    def select_sub(self):
        # if self.fitness == 0:
        #     fit = 1008610086
        # else:
        #     fit = 1. / (self.fitness)  # 适应度函数
        fit_arr = np.zeros(self.size_pop)
        # cumsum_fit = np.zeros(self.size_pop)
        for fit_num in range(self.fitness.size):
            fit = self.fitness[fit_num]
            if fit == 0:
                fit_arr[fit_num] = 0
            else:
                fit_arr[fit_num] = 1. / fit

        # 累积求和   a = np.array([1,2,3]) b = np.cumsum(a) b=1 3 6
        cumsum_fit = np.cumsum(fit_arr)
        pick = cumsum_fit[-1] / self.select_num * (
            np.random.rand() + np.array(range(int(self.select_num))))  # select_num  为子代选择个数 160
        i, j = 0, 0
        index = []
        while i < self.size_pop and j < self.select_num:
            if cumsum_fit[i] >= pick[j]:
                index.append(i)
                j += 1
            else:
                i += 1
        self.sub_sel = self.chrom[index, :]  # chrom 父

    # 交叉，依概率对子代个体进行交叉操作
    def cross_sub(self):
        if self.select_num % 2 == 0:  # select_num160
            num = range(0, int(self.select_num), 2)
        else:
            num = range(0, int(self.select_num - 1), 2)
        for i in num:
            if self.cross_prob >= np.random.rand():
                self.sub_sel[i, :], self.sub_sel[i + 1,
                                                 :] = self.intercross(self.sub_sel[i, :], self.sub_sel[i + 1, :])

    def intercross(self, ind_a, ind_b):  # ind_a，ind_b 父代染色体 shape=(1,14) 14=14个城市
        r1 = np.random.randint(self.num)  # 在num内随机生成一个整数 ，num=14.即随机生成一个小于14的数
        r2 = np.random.randint(self.num)
        while r2 == r1:  # 如果r1==r2
            r2 = np.random.randint(self.num)  # r2重新生成
        left, right = min(r1, r2), max(r1, r2)  # left 为r1,r2小值 ，r2为大值
        ind_a1 = ind_a.copy()  # 父亲
        ind_b1 = ind_b.copy()  # 母亲
        for i in range(left, right + 1):
            ind_a2 = ind_a.copy()
            ind_b2 = ind_b.copy()
            ind_a[i] = ind_b1[i]  # 交叉 （即ind_a  （1,14） 中有个元素 和ind_b互换
            ind_b[i] = ind_a1[i]
            x = np.argwhere(ind_a == ind_a[i])
            y = np.argwhere(ind_b == ind_b[i])

            """
                   下面的代码意思是 假如 两个父辈的染色体编码为【1234】，【4321】 
                   交叉后为【1334】，【4221】
                   交叉后的结果是不满足条件的，重复个数为2个
                   需要修改为【1234】【4321】（即修改会来
                   """
            if len(x) == 2:
                ind_a[x[x != i]] = ind_a2[i]  # 查找ind_a 中元素=- ind_a[i] 的索引
            if len(y) == 2:
                ind_b[y[y != i]] = ind_b2[i]
        return ind_a, ind_b

    # 变异模块  在变异概率的控制下，对单个染色体随机交换两个点的位置。
    def mutation_sub(self):
        for i in range(int(self.select_num)):  # 遍历每一个 选择的子代
            if np.random.rand() <= self.cross_prob:  # 如果随机数小于变异概率
                r1 = np.random.randint(self.num)  # 随机生成小于num==可设置 的数
                r2 = np.random.randint(self.num)
                while r2 == r1:  # 如果相同
                    r2 = np.random.randint(self.num)  # r2再生成一次
                self.sub_sel[i, [r1, r2]] = self.sub_sel[i,
                                                         [r2, r1]]  # 随机交换两个点的位置。

    # 进化逆转  将选择的染色体随机选择两个位置r1:r2 ，将 r1:r2 的元素翻转为 r2:r1 ，如果翻转后的适应度更高，则替换原染色体，否则不变
    def reverse_sub(self):
        for i in range(int(self.select_num)):  # 遍历每一个 选择的子代
            r1 = np.random.randint(self.num)  # 随机生成小于num==14 的数
            r2 = np.random.randint(self.num)
            while r2 == r1:  # 如果相同
                r2 = np.random.randint(self.num)  # r2再生成一次
            left, right = min(r1, r2), max(r1, r2)  # left取r1 r2中小值，r2取大值
            sel = self.sub_sel[i, :].copy()  # sel 为父辈染色体 shape=（1,14）

            # 将染色体中(r1:r2)片段 翻转为（r2:r1)
            sel[left:right + 1] = self.sub_sel[i, left:right + 1][::-1]
            # 如果翻转后的适应度小于原染色体，则不变
            if self.comp_fit_(sel) < self.comp_fit_(self.sub_sel[i, :]):
                self.sub_sel[i, :] = sel

    # 子代插入父代，得到相同规模的新群体
    def reins(self):
        index = np.argsort(self.fitness)[::-1]  # 替换最差的（倒序）
        self.chrom[index[:self.select_num], :] = self.sub_sel


def main(step_filename, initgene):
    assembly = OCCAssembly(step_filename)
    part_num = assembly.get_part_num()

    Path_short = Gena_TSP(assembly, part_num)  # 根据位置坐标，生成一个遗传算法类
    Path_short.rand_chrom(initgene)  # 初始化父类

    # x = data[:, 0]
    # y = data[:, 1]
    import time
    now_time = time.strftime('%Y%m%d%H%M%S', time.localtime())
    f = open(step_filename+'_'+str(now_time) + '.txt', 'w')
    # 循环迭代遗传过程
    bestfitnesses = []
    for i in range(Path_short.maxgen):
        print('gen:', i)
        Path_short.direction_list = []
        Path_short.select_sub()  # 选择子代
        Path_short.cross_sub()  # 交叉
        Path_short.mutation_sub()  # 变异
        Path_short.reverse_sub()  # 进化逆转
        Path_short.reins()  # 子代插入

        # 重新计算新群体的距离值
        for j in range(Path_short.size_pop):
            Path_short.fitness[j] = Path_short.comp_fit_(
                Path_short.chrom[j, :])
            Path_short.direction_list.append(Path_short.direction)

        # 每隔5步显示当前群体的最优装配路径
        index = Path_short.fitness.argmin()
        # 获取当前时间戳 记录运算时间
        timestamp = time.time()
        formatted_time = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

        print(formatted_time)
        print('累计碰撞检测次数' + str(Path_short.assembly_model.checking_num))
        print('第' + str(i + 1) + '代后的最优装配的评价函数: ' +
              str(Path_short.fitness[index]))
        print('第' + str(i + 1) + '代后的最优装配规划:')
        print('累计碰撞检测次数' + str(Path_short.assembly_model.checking_num), file=f)
        print('第' + str(i + 1) + '代后的最优装配的评价函数: ' +
              str(Path_short.fitness[index]), file=f)
        print('第' + str(i + 1) + '代后的最优装配规划:', file=f)
        out_path = Path_short.out_path(
            Path_short.chrom[index, :], Path_short.direction_list[index])  # 显示每一步的最优路径
        print(out_path, file=f)
        # Path_short.out_path(Path_short.chrom[index, :], Path_short.direction_list[index])  # 显示每一步的最优路径
        bestfitnesses.append(Path_short.fitness[index])
        # 存储每一步的最优路径及距离
        Path_short.best_fit.append(Path_short.fitness[index])
        Path_short.best_path.append(Path_short.chrom[index, :])
        Path_short.best_direction.append(Path_short.direction_list[index])

    f.close()
    # 显示优化过程
    # plotObj(Path_short.best_fit)
    # Path_short.assembly_model.display_result(Path_short.best_path[-1])

    return Path_short, bestfitnesses  # 返回遗传算法结果类


def plotObj(obj_list):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # show chinese
    plt.rcParams['axes.unicode_minus'] = False  # Show minus sign
    plt.plot(np.arange(1, len(obj_list) + 1), obj_list)
    plt.xlabel('Iterations')
    plt.ylabel('Obj Value')
    plt.grid()
    plt.xlim(1, len(obj_list) + 1)
    plt.show()


step_filename = 'assembly.step'
assembly = OCCAssembly(step_filename)
part_num = assembly.get_part_num()
initgene = []  # 初始化初始种群
for i in range(20):
    gene = np.arange(part_num)
    np.random.shuffle(gene)
    initgene.append(gene)
initgene1 = copy.deepcopy(np.array(initgene))

Path_short, oribestfitnesses = main(
    step_filename, initgene=initgene1)  # 这个先使用原始种群来做遗传算法优化
# print(Path_short)
initgene = np.array(initgene)
# 从对应模型的dqn产生的初始解里面加载初始解（1个）并且赋值给原始种群的第一个基因
initgene[0, :] = np.load(step_filename+'_dqnvalue.npy')

Path_short, bestfitnesses = main(
    step_filename, initgene=initgene)  # 做对加入了dqn初始解的遗传算法对比实验

plt.figure(1)

plt.plot(range(len(oribestfitnesses)), oribestfitnesses, label='原始遗传算法')

plt.plot(range(len(bestfitnesses)), bestfitnesses, label='加入dqn解的遗传算法')
plt.xlabel('iteration')

plt.ylabel('bestfitness')
plt.legend()

plt.savefig(step_filename+'_iteration.png')
