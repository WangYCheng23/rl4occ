import OCC
# 导入OCC模块
from OCC.Extend.DataExchange import read_step_file, STEPControl_Reader, STEPControl_Writer
from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Trsf, gp_Vec
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.GProp import GProp_GProps
import random
import math
import numpy as np
from tqdm import tqdm


class Transform:
    def __init__(self):
        self.direction = [1, 0, 0]  # 平移
        # self.direction_sign = ['x', '-x', 'y', '-y', 'z', '-z']
        self.sign = 'x'


def direction_sign(x, y, z):
    a = math.fabs(x)
    b = math.fabs(y)
    c = math.fabs(z)
    sign = 'x'
    if a > b:
        if a > c:
            max_num = 1
        else:
            max_num = 3
    else:
        if b > c:
            max_num = 2
        else:
            max_num = 3
    if max_num == 1 and x > 0:
        sign = 'x'
    if max_num == 1 and x < 0:
        sign = '-x'
    if max_num == 2 and y > 0:
        sign = 'y'
    if max_num == 2 and y < 0:
        sign = '-y'
    if max_num == 3 and z > 0:
        sign = 'z'
    if max_num == 3 and z < 0:
        sign = '-z'  # 判断哪个坐标轴的绝对值最大，并确定该坐标轴的符号，从而确定了方向的符号。例如，如果x的绝对值最大且x为正数，则方向符号为'x'；如果y的绝对值最大且y为负数，则方向符号为'-y'；以此类推。

    return sign


class OCCAssembly:
    def __init__(self, part_file):
        print(f'选择装配体:{part_file}')
        self.part_file = part_file  # 装配体路径
        self.part_list = []  # 零件集合
        self.result_shells = []  # 平移后的零件集合
        self.boom_transform = []  # 爆炸方向、距离
        self.boom_shells = []  # 爆炸后的零件集合
        self.transformed_ids = []  # 平移后的零件集合
        self.origin_shells = []
        self.boom_step = 5
        self.trans_step = 20
        self.trans_count = 2
        self.checking_num = 0
        # 初始化了一个变量center_position，表示装配体的中心位置，默认为[0, 0, 0]
        self.center_position = [0, 0, 0]
        self.bboxes = []  # 初始化了一个空列表bboxes，用于存储装配体的包围盒

    def get_part_num(self):  # 用于获取装配体中零件的数量
        # 读取 STEP 模型文件
        filename = self.part_file
        # step_shape = read_step_file(step_filename)
        reader = STEPControl_Reader()
        reader.ReadFile(filename)
        reader.TransferRoots()
        step_shape = reader.OneShape()
        topo = TopologyExplorer(step_shape)
        shells = list(topo.shells())
        self.part_num = len(shells)
        return len(shells)

    #     创建爆炸图
    def create_boom(self):
        self.result_shells = []
        # 读取 STEP 模型文件
        step_filename = self.part_file
        # step_shape = read_step_file(step_filename)
        reader = STEPControl_Reader()
        reader.ReadFile(step_filename)
        reader.TransferRoots()
        step_shape = reader.OneShape()

        # 求平移方向
        bbox = Bnd_Box()
        # 计算形状的包围盒
        brepbndlib.Add(step_shape, bbox)
        center_x = (bbox.CornerMin().X() + bbox.CornerMax().X()) / 2
        center_y = (bbox.CornerMin().Y() + bbox.CornerMax().Y()) / 2
        center_z = (bbox.CornerMin().Z() + bbox.CornerMax().Z()) / 2
        self.center_position = [center_x, center_y, center_z]
        # 遍历装配图中的shell
        topo = TopologyExplorer(step_shape)

        shells = list(topo.shells())
        # shell_list = []

        # 生成爆炸后的shells
        for i in range(len(shells)):
            shell = shells[i]
            bbox = Bnd_Box()
            # 计算形状的包围盒
            brepbndlib.Add(shell, bbox)
            centert_x = (bbox.CornerMin().X() + bbox.CornerMax().X()) / 2
            centert_y = (bbox.CornerMin().Y() + bbox.CornerMax().Y()) / 2
            centert_z = (bbox.CornerMin().Z() + bbox.CornerMax().Z()) / 2
            self.bboxes.append(bbox)

            trsf = gp_Trsf()
            # 使用gp_Vec定义平移向量
            # 修改爆炸距离
            x = center_x - centert_x * self.boom_step
            y = center_y - centert_y * self.boom_step
            z = center_z - centert_z * self.boom_step
            sign = direction_sign(x, y, z)
            # if sign == 'x' or sign == '-x':
            #     y = center_y
            #     z = center_z
            # if sign == 'y' or sign == '-y':
            #     x = center_x
            #     z = center_z
            # if sign == 'z' or sign == '-z':
            #     y = center_y
            #     x = center_x
            if sign == 'x' or sign == '-x':
                y = 0
                z = 0
            if sign == 'y' or sign == '-y':
                x = 0
                z = 0
            if sign == 'z' or sign == '-z':
                y = 0
                x = 0
            # print('爆炸', x, y, z, sign, i)
            translation_vector = gp_Vec(x, y, z)
            # 设置平移转换
            trsf.SetTranslation(translation_vector)
            # 应用转换到shape
            transform = BRepBuilderAPI_Transform(shell, trsf)
            # 获取转换后的shape
            transformed_shape = transform.Shape()
            transform_vec = Transform()
            # transform_vec.direction = (i + 1) * 10
            transform_vec.direction = [x, y, z]
            transform_vec.sign = sign

            self.origin_shells.append(shell)
            self.boom_transform.append(transform_vec)
            self.boom_shells.append(transformed_shape)

    def display(self):
        # 初始化可视化界面
        display, start_display, add_menu, add_function_to_menu = init_display()
        for i in range(len(self.result_shells)):
            # if self.transformed_ids[i] not in [9, 0, 10]:
            #     continue
            # 添加显示
            display.DisplayShape(self.result_shells[i], update=True)
        start_display()

    def display_boom(self):
        # 初始化可视化界面
        display, start_display, add_menu, add_function_to_menu = init_display()
        for i in range(len(self.boom_shells)):
            # 添加显示
            display.DisplayShape(self.boom_shells[i], update=True)
        start_display()

    def display_result(self, ids):
        for id in ids:
            # 初始化可视化界面
            display, start_display, add_menu, add_function_to_menu = init_display()
            # 添加显示
            display.DisplayShape(self.boom_shells[id], update=True)
            start_display()

    #     平移零件
    def transform_parts(self, assembly_ids):
        if len(self.boom_transform) == 0:
            self.create_boom()
        # print('装配序列', assembly_ids)
        # print('Pingyi', self.boom_transform)
        self.transformed_ids = []
        self.result_shells = []
        count = 0
        transform_num = 0
        trans_direction = []
        for id in assembly_ids:
            trans_direction.append(self.boom_transform[id].sign)

            # print('---------------------')
            # print('移动', id)
            self.transformed_ids.append(id)
            trans = self.boom_transform[id]
            # print(trans.sign, trans.direction, id)
            # distance = trans.distance - self.trans_step
            self.result_shells.append(self.boom_shells[id])
            for i in range(self.trans_count):
                shell = self.boom_shells[id]
                distance = (self.trans_count - i - 1) * 1.0 / \
                    self.trans_count * self.trans_step
                x = trans.direction[0] - distance
                y = trans.direction[1] - distance
                z = trans.direction[2] - distance

                if trans.sign == 'x' or trans.sign == '-x':
                    y = 0
                    z = 0
                if trans.sign == 'y' or trans.sign == '-y':
                    x = 0
                    z = 0
                if trans.sign == 'z' or trans.sign == '-z':
                    y = 0
                    x = 0
                # if i == self.trans_count - 1:
                #     print('平移', x, y, z, id)

                trsf = gp_Trsf()
                # 使用gp_Vec定义平移向量
                # 修改爆炸距离
                translation_vector = gp_Vec(-x, -y, -z)
                # 设置平移转换
                trsf.SetTranslation(translation_vector)
                # 应用转换到shape
                transform = BRepBuilderAPI_Transform(shell, trsf)
                # 获取转换后的shape
                transformed_shape = transform.Shape()
                self.result_shells[transform_num] = transformed_shape
                count += self.interference_checking(transformed_shape)
            transform_num += 1
        # print(count)
        return count, trans_direction

    def get_ijcount(self, i, j):
        ij = 0
        transi = self.boom_transform[i]
        x = transi.direction[0]
        y = transi.direction[1]
        z = transi.direction[2]
        if transi.sign == 'x' or transi.sign == '-x':
            y = 0
            z = 0
        if transi.sign == 'y' or transi.sign == '-y':
            x = 0
            z = 0
        if transi.sign == 'z' or transi.sign == '-z':
            y = 0
            x = 0
        trsf = gp_Trsf()
        translation_vector = gp_Vec(-x, -y, -z)
        # 设置平移转换
        trsf.SetTranslation(translation_vector)
        # 应用转换到shape
        shelli = self.boom_shells[i]
        transformi = BRepBuilderAPI_Transform(shelli, trsf)
        # 获取转换后的shape
        transformed_shapei = transformi.Shape()
        if i != j:
            transj = self.boom_transform[j]
            for _ in range(self.trans_count):

                distance = (self.trans_count - _ - 1) * \
                    1.0 / self.trans_count * self.trans_step
                x = transj.direction[0] - distance
                y = transj.direction[1] - distance
                z = transj.direction[2] - distance

                if transj.sign == 'x' or transj.sign == '-x':
                    y = 0
                    z = 0
                if transj.sign == 'y' or transj.sign == '-y':
                    x = 0
                    z = 0
                if transj.sign == 'z' or transj.sign == '-z':
                    y = 0
                    x = 0

                trsf = gp_Trsf()
                # 使用gp_Vec定义平移向量
                # 修改爆炸距离
                translation_vector = gp_Vec(-x, -y, -z)
                # 设置平移转换
                trsf.SetTranslation(translation_vector)
                # 应用转换到shape
                shellj = self.boom_shells[j]
                transformj = BRepBuilderAPI_Transform(shellj, trsf)
                # 获取转换后的shape
                transformed_shapej = transformj.Shape()

                self.result_shells = [
                    transformed_shapei, transformed_shapej]
                ij += self.interference_checking(transformed_shapej)
        return ij
        
    def compute_countij(self):
        self.countij = np.zeros((self.part_num, self.part_num))
        count = 0
        with tqdm(total=self.part_num * (self.part_num - 1) * self.trans_count) as pbar:
            pbar.set_description(f'累计碰撞检测次数:')
            for i in range(self.part_num):
                transi = self.boom_transform[i]
                x = transi.direction[0]
                y = transi.direction[1]
                z = transi.direction[2]
                if transi.sign == 'x' or transi.sign == '-x':
                    y = 0
                    z = 0
                if transi.sign == 'y' or transi.sign == '-y':
                    x = 0
                    z = 0
                if transi.sign == 'z' or transi.sign == '-z':
                    y = 0
                    x = 0
                trsf = gp_Trsf()
                translation_vector = gp_Vec(-x, -y, -z)
                # 设置平移转换
                trsf.SetTranslation(translation_vector)
                # 应用转换到shape
                shelli = self.boom_shells[i]
                transformi = BRepBuilderAPI_Transform(shelli, trsf)
                # 获取转换后的shape
                transformed_shapei = transformi.Shape()
                for j in range(self.part_num):

                    if i != j:
                        transj = self.boom_transform[j]
                        for _ in range(self.trans_count):

                            distance = (self.trans_count - _ - 1) * \
                                1.0 / self.trans_count * self.trans_step
                            x = transj.direction[0] - distance
                            y = transj.direction[1] - distance
                            z = transj.direction[2] - distance

                            if transj.sign == 'x' or transj.sign == '-x':
                                y = 0
                                z = 0
                            if transj.sign == 'y' or transj.sign == '-y':
                                x = 0
                                z = 0
                            if transj.sign == 'z' or transj.sign == '-z':
                                y = 0
                                x = 0

                            trsf = gp_Trsf()
                            # 使用gp_Vec定义平移向量
                            # 修改爆炸距离
                            translation_vector = gp_Vec(-x, -y, -z)
                            # 设置平移转换
                            trsf.SetTranslation(translation_vector)
                            # 应用转换到shape
                            shellj = self.boom_shells[j]
                            transformj = BRepBuilderAPI_Transform(shellj, trsf)
                            # 获取转换后的shape
                            transformed_shapej = transformj.Shape()

                            self.result_shells = [
                                transformed_shapei, transformed_shapej]
                            self.countij[j, i] += self.interference_checking(transformed_shapej)
                            self.checking_num += 1
                            pbar.update(1)
                            count += 1

    def interference_checking(self, transformed_shape):
        # if self.checking_num % 100 == 0:
        #     print('累计碰撞检测次数' + str(self.checking_num))
        count = 0
        for j in range(len(self.result_shells) - 1):
            # 获取两个形状
            shape_i = transformed_shape
            shape_j = self.result_shells[j]

            # 创建一个 BRepAlgoAPI_Common 对象计算它们的布尔交
            boolean_common = BRepAlgoAPI_Common(shape_i, shape_j)

            # 检查运算是否成功
            if boolean_common.IsDone():
                # 获取布尔交的结果
                common_shape = boolean_common.Shape()

                # 创建一个GProp_GProps对象，用来存储体积属性
                props = GProp_GProps()
                # 计算体积属性
                brepgprop.VolumeProperties(common_shape, props)
                # 获取体积
                volume = props.Mass()
                if volume > 1:
                    count = count + 1

        return count

    def test_checking(self):
        trans = self.transformed_ids
        count = 0
        for i in range(len(trans)):
            for j in range(i + 1, len(trans)):
                display, start_display, add_menu, add_function_to_menu = init_display()
                display.DisplayShape(
                    self.result_shells[i], update=True, color="RED")
                display.DisplayShape(
                    self.result_shells[j], update=True, color="YELLOW")
                start_display()

                # 获取两个形状
                shape_i = self.result_shells[i]
                shape_j = self.boom_shells[j]

                # 创建一个 BRepAlgoAPI_Common 对象计算它们的布尔交
                boolean_common = BRepAlgoAPI_Common(shape_i, shape_j)
                # myCut3 = BRepAlgoAPI_Common(shape_i, shape_j).Shape()
                # display.DisplayShape(myCut3, update=True, color="RED")

                # 检查运算是否成功
                if boolean_common.IsDone():
                    # 获取布尔交的结果
                    common_shape = boolean_common.Shape()

                    # 创建一个GProp_GProps对象，用来存储体积属性
                    props = GProp_GProps()
                    # 计算体积属性
                    brepgprop.VolumeProperties(common_shape, props)
                    # 获取体积
                    volume = props.Mass()
                    if volume > 1:
                        count = count + 1

        # print(count)


if __name__ == '__main__':
    step_filename = 'assembly.step'
    assembly = OCCAssembly(step_filename)
    part_num = assembly.get_part_num()

    ids = list(range(part_num))
    # ids = [9, 18, 5, 29, 6, 19, 4, 24, 0, 16, 1, 3, 13, 28, 2, 10, 12, 27, 25, 20, 34, 30, 37, 31, 33, 11, 21, 22, 7,
    #        32, 8, 17, 36, 35, 14, 23, 15, 26]

    random.shuffle(ids)
    print(ids)

    assembly.create_boom()
    # assembly.display_result(ids)
    res = assembly.transform_parts(ids)
    print('碰撞次数', res[0])
    # assembly.display_boom()
    assembly.display()
    # assembly.interference_checking()
    # assembly.display()
