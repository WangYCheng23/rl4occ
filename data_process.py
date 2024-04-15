'''
Author: WANG CHENG
Date: 2024-04-15 22:28:33
LastEditTime: 2024-04-15 23:44:49
'''
import os
import shutil
import random
from tqdm import tqdm

root_dir = "/media/wangc/Data/数据集/"

# 指定目标目录
target_directory = './data/'
# 使用os.makedirs()创建目录，如果目录不存在
os.makedirs(target_directory, exist_ok=True)

def delete_files_in_folder(folder_path):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        os.remove(file_path)

def copy_and_rename_files(root_dir, target_directory):
    # 获取所有子目录中的assembly.step文件
    assembly_files = [os.path.join(root,f) for root, dirs, files in os.walk(root_dir) for f in files if f == 'assembly.step']
    
    for i, file in tqdm(enumerate(assembly_files), total=len(assembly_files)):
        # 检查文件是否存在
        if os.path.exists(file):
            shutil.copy2(file, target_directory)
            new_file_name = f'assembly_{i}.step'
            new_file_path = os.path.join(target_directory, new_file_name)
            # 如果目标目录中已经存在同名文件，则增加序号直到找到一个可用的文件名为止
            while os.path.exists(new_file_path):
                i += 1
                new_file_name = f'assembly_{i}.step'
                new_file_path = os.path.join(target_directory, new_file_name)
            # 正确地重命名复制的文件
            os.rename(os.path.join(target_directory, 'assembly.step'), new_file_path)
        else:
            print(f"File {file} does not exist.")

# 删除目标目录中的所有文件
delete_files_in_folder(target_directory)

# 调用函数进行复制和重命名
copy_and_rename_files(root_dir, target_directory)
    
# 获取目标目录下的所有文件，并存储到列表中
files = os.listdir(target_directory)

# 打乱文件顺序
random.shuffle(files)

# 划分训练集和验证集，这里假设训练集占80%
train_set_size = int(len(files) * 0.8)
train_set = files[:train_set_size]
val_set = files[train_set_size:]

# 创建训练集和验证集的目录
os.makedirs(os.path.join(target_directory, 'train'), exist_ok=True)
os.makedirs(os.path.join(target_directory, 'val'), exist_ok=True)

# 复制文件到相应的目录
for file in train_set:
    src_file = os.path.join(target_directory, file)
    dest_file = os.path.join(target_directory, 'train', file)
    shutil.move(src_file, dest_file)

for file in val_set:
    src_file = os.path.join(target_directory, file)
    dest_file = os.path.join(target_directory, 'val', file)
    shutil.move(src_file, dest_file)