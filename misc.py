#-- coding:UTF-8 --
'''
Author: WANG CHENG
Date: 2024-04-17 20:18:16
LastEditTime: 2024-05-13 11:03:07
'''
import multiprocessing
import time
import torch
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
# part_nums = np.load('./misc/part_nums.npy')
# print(min(part_nums))
# print(max(part_nums))

###########################################

# 定义注意力得分矩阵和掩码
# scores = torch.randn(64, 4, 19, 19)  # 假设 batch_size=64, num_heads=4, seq_len=19
# mask = torch.zeros(64, 19, dtype=torch.bool)  # 假设每个序列长度为19

# # 为每个序列的第一个位置创建掩码（假设这些位置为填充位置）
# for i in range(64):
#     mask[i, 0] = 1

# # 扩展掩码的维度
# expanded_mask = mask.unsqueeze(1).unsqueeze(1)  # (64, 1, 1, 19)

# # 将掩码应用于 scores
# masked_scores = scores.masked_fill(expanded_mask == True, -float('inf'))

# # 检查结果
# print("Original Scores:")
# print(scores[0, 0])  # 打印第一个序列的第一个头的注意力得分矩阵
# print("\nMasked Scores:")
# print(masked_scores[0, 0])  # 打印应用掩码后的注意力得分矩阵


###########################################
# batch_size = 16
# num_heads = 4
# sequence_length = 19


# # 注意力矩阵
# attention_matrix = torch.randn(batch_size, num_heads, sequence_length, sequence_length)

# # mask矩阵 batch_size*sequence_length
# mask_matrix = torch.randint(0, 2, (batch_size, sequence_length))
# # mask_matrix = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]])  # 举例一个3x3的mask矩阵

# # 将mask矩阵扩展到与注意力矩阵相同的维度
# expanded_mask = mask_matrix.unsqueeze(1).unsqueeze(2).expand(-1, num_heads, sequence_length, sequence_length)

# # 将需要mask的位置置为负无穷
# attention_matrix.masked_fill_(expanded_mask == 1, float('-inf'))
# print(attention_matrix)
###########################################
from assembly import OCCAssembly
import pickle

def save_assembly_to_pickle(assembly_path, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    filename = os.path.join(out_path, assembly_path.split('/')[-1].replace('.step', '.pkl'))
    assembly = OCCAssembly(assembly_path)
    print(assembly.get_part_num())
    assembly.create_boom()
    assembly.compute_countij() 
    with open(filename, 'wb') as f:
        pickle.dump(assembly, f)

def load_assembly_from_pickle(assembly_path):
    with open(assembly_path, 'rb') as f:
        assembly = pickle.load(f)
    return assembly

def worker(assembly_path, out_path):
    try:
        # 这里是你的工作逻辑
        print(f"Process {assembly_path} is running")
        save_assembly_to_pickle(assembly_path, out_path)  # 假设我们在这里模拟工作
        # 如果发生异常，则会跳到 except 块
        # raise ValueError("Something went wrong!")  # 假设这里发生了一个错误
    except Exception as e:
        print(f"Process {assembly_path} encountered an exception: {e}")
        # 可以选择在这里记录日志或者执行其他清理工作
        # 由于异常，我们结束这个进程
        return

    # 如果没有异常，正常结束进程
    print(f"Process {assembly_path} finished successfully")

def parallel_pack_step_files(step_files, out_path, num_processes):
    """并行打包.step文件的函数""" 
    # 创建一个进程池
    pool = multiprocessing.Pool(processes=num_processes)
    
    # 使用多进程映射执行worker函数
    pool.starmap(worker, list(zip(step_files, [out_path]*len(step_files))))
    
    # 关闭进程池
    pool.close()
    pool.join()
    
if __name__ == "__main__":
    cwd = os.getcwd()
    
    for dir_path in os.listdir(os.path.join(cwd, 'sorted_step_files')):
    
        # 假设您有一个包含所有.step文件路径的列表
        step_files_list = [os.path.join(cwd, 'sorted_step_files', dir_path, file_path) for file_path in os.listdir(os.path.join(cwd, 'sorted_step_files', dir_path))]

        # 选择您想要使用的并行进程数量
        num_processes = multiprocessing.cpu_count()  # 使用所有可用的CPU核心

        out_path = os.path.join(cwd, 'pickle_data', dir_path)
        # 调用函数开始并行打包
        parallel_pack_step_files(step_files_list, out_path, num_processes)