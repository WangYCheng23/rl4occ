'''
Author: WANG CHENG
Date: 2024-04-17 20:18:16
LastEditTime: 2024-04-20 01:43:07
'''
from assembly import OCCAssembly
import os
import numpy as np

# step_filenames = [file for file in os.listdir('./data/train')]
# part_nums = []
# for step_filename in step_filenames:    
#     assembly = OCCAssembly(os.path.join('./data/train', step_filename))
#     part_num = assembly.get_part_num()
#     part_nums.append(part_num)

# # 保存part_nums到npy文件
# np.save('./part_nums.npy', np.array(part_nums))
# print(max(part_nums))

# 读取npy文件
part_nums = np.load('./misc/part_nums.npy')
print(min(part_nums))
print(max(part_nums))