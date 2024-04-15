
import numpy as np

import threading

class Replay_Buffer: #定义了一个经验回放缓冲区类 Replay_Buffer，用于存储训练过程中的经验数据，以供后续训练时使用


    def __init__(self,size,n_state): #类的初始化方法，接受两个参数 size 和 n_state，分别表示缓冲区的大小和状态向量的维度
    
    
        self.size=size
        self.n_state=n_state #将输入的 size 和 n_state 分别赋值给类的 size 和 n_state 属性
        
        self.buffers={} #初始化一个空字典 buffers，用于存储不同类型的经验数据
        
        self.buffers['state']=np.empty((self.size,self.n_state))
        
        
        self.buffers['next_state']=np.empty((self.size,self.n_state))
        
        self.buffers['r']=np.empty((self.size,1))
        
        self.buffers['isterminated']=np.empty((self.size,1))
        
        self.buffers['actions']=np.empty((self.size,1)) #初始化一个形状为 (size, 1) 的空数组，用于存储动作、奖励、状态、下一个状态、存储是否终止
        
        self.current_idx=0
        
        self.current_size=0 #初始化当前索引为 0，当前大小为 0，用于跟踪当前经验数据存储的位置
        self.lock = threading.Lock() #初始化一个线程锁，用于在多线程环境下保护对缓冲区的访问
        
    def store_records(self,records): #定义了 Replay_Buffer 类中的 store_records 方法，用于将经验数据存储到经验回放缓冲区中
    
        with self.lock: #使用线程锁，确保在多线程环境下对缓冲区的操作是安全的
            samplenums=records['state'].shape[0] #获取要存储的经验数据样本数
            
            if samplenums+self.current_idx<self.size:
                idxs=np.arange(self.current_idx,self.current_idx+samplenums)
            else:
                idxs=np.concatenate([np.arange(self.current_idx,self.size),np.arange(samplenums-self.size+self.current_idx)]) #判断当前经验数据加上已存储的数据是否超过缓冲区大小。如果未超过，直接将数据存储在当前索引位置开始的连续空间中。如果超过，需要分两次存储，一部分存储在当前索引位置开始的空间中，另一部分存储在数组开头的空间中，替换掉旧的经验值，以实现循环存储的效果。
                
            self.buffers['state'][idxs]=records['state'] #将状态数据存储到缓冲区的 state 数组中
            
            self.buffers['next_state'][idxs]=records['next_state'] #将下一个状态数据存储到缓冲区的 next_state 数组中
                
            self.buffers['r'][idxs]=records['r'] #将即时奖励数据存储到缓冲区的 r 数组中

            self.buffers['isterminated'][idxs]=records['isterminated'] #将是否终止的标志数据存储到缓冲区的 isterminated 数组中
            
            self.buffers['actions'][idxs]=records['actions'] #将动作数据存储到缓冲区的 actions 数组中
            
            self.current_idx=(self.current_idx+samplenums)%self.size #更新当前索引，使其指向下一个可用的存储位置
            
            self.current_size=min(self.current_size+samplenums,self.size) #更新当前大小，确保不超过缓冲区的大小
        
    def sample_records(self,sample_nums): #定义了 Replay_Buffer 类中的 sample_records 方法，用于从经验回放缓冲区中随机抽样出一批经验数据
    
    
        sample_nums=min(sample_nums,self.current_size) #确保要抽样的数量不超过当前缓冲区中存储的经验数据数量。
        
        idxs=np.arange(sample_nums) #创建一个长度为 sample_nums 的数组，表示抽样的索引
        np.random.shuffle(idxs) #将索引数组随机打乱，用于随机抽样
        
        records={} #创建一个空字典，用于存储抽样出的经验数据
        
        for key in self.buffers.keys(): #遍历缓冲区中的所有数据类型
        
            records[key]=self.buffers[key][idxs] #将抽样得到的索引对应的经验数据存储到 records 字典中
            
        return records #返回包含抽样数据的字典 records
        