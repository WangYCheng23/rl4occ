#-- coding:UTF-8 --
'''
Author: WANG CHENG
Date: 2024-04-19 00:44:42
LastEditTime: 2024-05-06 00:32:15
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"
        
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.concat = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        
        # Linear transformations for query, key, value
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        
        # Reshape to split into multiple heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float())  # (batch_size, num_heads, seq_len, seq_len)
        
        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(1).expand(batch_size, -1, seq_len, seq_len)==False, float('-inf'))
            # mask = int(~mask[:,:,0].unsqueeze(1).unsqueeze(1))*(float('-inf'))
            # scores = scores*mask
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        
        # Apply dropout
        attention_weights = F.dropout(attention_weights, p=0.1)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)  # (batch_size, seq_len, embed_dim)
        
        # Final linear transformation
        output = self.concat(attention_output)
        
        return output

class AttentionQNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, embed_dim, num_heads, device=None):
        super(AttentionQNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not device else device
        
        self.embedding = nn.Linear(input_dim, embed_dim).to(self.device)  # Embedding layer for input
        self.attention = MultiHeadAttention(embed_dim, num_heads).to(self.device)
        self.norm1 = nn.LayerNorm(embed_dim).to(self.device)        
        self.fc1 = nn.Linear(embed_dim, hidden_dim).to(self.device)
        self.norm2 = nn.LayerNorm(hidden_dim).to(self.device)     
        self.fc2 = nn.Linear(hidden_dim, output_dim).to(self.device)
        
        # 添加一个残差连接层，用于匹配embedding到fc1的维度
        self.res_fc1 = nn.Linear(embed_dim, hidden_dim).to(self.device)
   
    def forward(self, x, padding_mask=None):
        x = self.embedding(x)
        x_attention = self.attention(x, x, x, padding_mask)  # 注意力机制
        x_norm1 = self.norm1(x_attention)
        x_fc1 = F.relu(self.fc1(x_norm1))  # 第一个全连接层

        # 添加残差链接，将embedding层的输出通过一个线性层映射到fc1的维度，并与fc1的输出相加
        x_res_fc1 = self.res_fc1(x) + x_fc1

        x_norm2 = self.norm2(x_res_fc1)
        x_fc2 = self.fc2(x_norm2)  # 第二个全连接层

        return x_fc2.squeeze(-1)
    
class Q_Transformer(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Q_Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

if __name__ == "__main__":
    input_dim = 9
    embed_dim = 64
    hidden_dim = 64
    num_heads = 4
    output_dim = 1  # 输出维度
    seq_length = 10  # 序列长度

    policy_net = AttentionQNet(input_dim, output_dim, hidden_dim, embed_dim, num_heads)
    input_tensor = torch.randn(1, seq_length, input_dim)  # Example input tensor with shape (batch_size, seq_len, input_dim)
    masked_positions = [4,7]
    output_tensor = policy_net(input_tensor)
    # 创建掩码张量
    mask = torch.ones_like(output_tensor)  # 先创建一个全 1 的张量
    # 将需要掩盖的位置置零
    mask[:,masked_positions] = 0
    output_tensor.masked_fill_(mask==0, -float('inf'))
    print(output_tensor)
