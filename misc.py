'''
Author: WANG CHENG
Date: 2024-04-17 20:18:16
LastEditTime: 2024-04-22 00:34:24
'''
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

def save_assembly_to_pickle(assembly_path):
    filename = os.path.join('./pickle_data',assembly_path.split('/')[-1].replace('.step', '.pkl'))
    assembly = OCCAssembly(assembly_path)
    print(assembly.get_part_num())
    assembly.create_boom()
    assembly.compute_countij() 
    with open(filename, 'wb') as f:
        pickle.dump(assembly, f)

ass_path = './data/all/assembly_120.step'
save_assembly_to_pickle(ass_path)

def load_assembly_from_pickle(assembly_path):
    with open(assembly_path, 'rb') as f:
        assembly = pickle.load(f)
    return assembly

assembly = load_assembly_from_pickle('./pickle_data/assembly_120.pkl')  
# assembly.display_boom()
print(assembly.countij)