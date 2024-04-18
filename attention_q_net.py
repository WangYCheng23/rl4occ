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
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
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
    def __init__(self, output_dim, hidden_dim, embed_dim, num_heads):
        super(AttentionQNet, self).__init__()
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.embedding = nn.Linear(1, embed_dim)  # Embedding layer for input
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.attention(x, x, x, mask)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    seq_len = 10
    output_dim = 5
    embed_dim = 32
    num_heads = 4

    policy_net = AttentionQNet(output_dim, embed_dim, num_heads)
    input_tensor = torch.randn(1, seq_len, 1)  # Example input tensor with shape (batch_size, seq_len, input_dim)
    output_tensor = policy_net(input_tensor)
    print(output_tensor)
