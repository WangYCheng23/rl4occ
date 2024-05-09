'''
Author: WANG CHENG
Date: 2024-04-20 13:36:05
LastEditTime: 2024-05-10 01:07:46
'''
import copy
import numpy as np
import torch
from memory_profiler import profile

def pad_sequences(sequences, atten_mask, max_len, padding_value=1):
    # x = np.ones((len(sequences), max_len, 10), dtype=np.float32)
    out = []
    src_key_mask = np.zeros((len(sequences), max_len))
    for i, seq in enumerate(sequences):
        if len(seq) < max_len:
            src_key_mask[i,len(seq):] = 1   # src_key_mask
            atten_mask[i] = np.pad(np.array(atten_mask[i]), (0, max_len - len(seq)), 'constant', constant_values=1) # atten_mask
            # key_mask[i] = copy.deepcopy(atten_mask[i])   # key_mask
            out.append(np.concatenate((seq, np.array([[padding_value for _ in range(10)]]  * (max_len - len(seq))))))
        else:
            out.append(seq)
    out = np.array(out)
    atten_mask = np.array(atten_mask)
    tgt_key_mask = copy.deepcopy(atten_mask)    # tgt_key_mask
    return copy.deepcopy(out), copy.deepcopy(out), src_key_mask, tgt_key_mask, atten_mask

# @profile(precision=4, stream=open("memory_profiler.log", "w+"))
def pad_sequences_and_create_mask(sequences, padding_value=0):
    # 将列表转换为张量
    tensor_sequences = [torch.FloatTensor(seq) for seq in sequences]

    max_length = max(seq.size(0) for seq in tensor_sequences)
    
    # Pad sequences to the max length
    padded_sequences = []
    mask_sequences = []
    for seq in tensor_sequences:
        pad_length = max_length - seq.size(0)
        padded_seq = torch.cat([seq, torch.full((pad_length,) + seq.size()[1:], padding_value, dtype=seq.dtype)], dim=0)
        mask_sequences.append(torch.concatenate((torch.ones(seq.size(0), dtype=torch.bool),
                                                torch.zeros(pad_length, dtype=torch.bool)))
                              )
        padded_sequences.append(padded_seq)
    
    # Stack padded sequences into a tensor
    return torch.stack(padded_sequences, dim=0), torch.stack(mask_sequences, dim=0)
    

if __name__ == "__main__":
        
    sequences = [np.array([[1, 2, 3], [4, 5, 6]]), np.array([[7, 8, 9]])]
    padded_sequences, mask = pad_sequences_and_create_mask(sequences)
    print("Padded Sequences:\n", padded_sequences)
    print("\nMask:\n", mask)