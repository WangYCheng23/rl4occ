'''
Author: WANG CHENG
Date: 2024-04-20 13:36:05
LastEditTime: 2024-04-20 17:19:53
'''
import numpy as np
import torch


def pad_sequence_to_max_length(sequences, padding_value=float('-inf')):
    # Determine the maximum sequence length
    max_length = max(seq.size(0) for seq in sequences)
    
    # Pad sequences to the max length
    padded_sequences = []
    for seq in sequences:
        pad_length = max_length - seq.size(0)
        padded_seq = torch.cat([seq, torch.full((pad_length,) + seq.size()[1:], padding_value, dtype=seq.dtype)], dim=0)
        padded_sequences.append(padded_seq)
    
    # Stack padded sequences into a tensor
    return torch.stack(padded_sequences, dim=0)

def pad_sequences_and_create_mask(sequences, padding_value=float('-inf')):
    # 将列表转换为张量
    tensor_sequences = [torch.FloatTensor(seq) for seq in sequences]

    # 找到最大长度
    max_length = max(len(seq) for seq in sequences)

    # 填充序列
    padded_sequences = pad_sequence_to_max_length(tensor_sequences, padding_value) 
    
    # Create mask
    mask = (padded_sequences != padding_value)  # batch seq_len input_dim
    
    # padded_sequences 中等于 -inf的地方换成0
    padded_sequences[padded_sequences == padding_value] = 0
    
    return padded_sequences, mask

if __name__ == "__main__":
    
    # Example usage:
    sequences = [torch.FloatTensor([[1, 2, 3], [4, 5, 6]]), torch.FloatTensor([[7, 8, 9]])]
    padded_sequences = pad_sequence_to_max_length(sequences)
    print(padded_sequences)
        
    sequences = [np.array([[1, 2, 3], [4, 5, 6]]), np.array([[7, 8, 9]])]
    padded_sequences, mask = pad_sequences_and_create_mask(sequences)
    print("Padded Sequences:\n", padded_sequences)
    print("\nMask:\n", mask)