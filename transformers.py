import torch
import math
import torch.nn as nn
import torch.nn.functional as F

def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor
    ) -> tuple:
    
    bsz, seq_len, d_model = q.size()
    freq_range = torch.arange(d_model // 2, dtype = torch.float32)
    freq_cis = torch.pow(10_000, -2*freq_range/d_model)
    pos = torch.arange(seq_len, dtype = torch.float32)
    freq = torch.einsum('i,j->ij', pos, freq_cis)
    
    sin_embd = torch.sin(freq).unsqueeze(0)
    cos_embd = torch.cos(freq).unsqueeze(0)
    
    q_odd , q_even = q[..., 1::2], q[..., ::2]
    k_odd, k_even = k[..., 1::2], k[..., ::2]
    
    q_rot = torch.cat([q_even * cos_embd - q_odd * sin_embd,
                       q_even * sin_embd - q_odd * cos_embd], dim =  -1)
    k_rot = torch.cat([k_even * cos_embd - k_odd * sin_embd,
                       k_even * sin_embd - k_odd * cos_embd], dim =  -1)
    
    return q_rot, k_rot

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert (cfg['emb_dim'] % cfg['n_heads'] == 0), \
            "d_out has to be divisble by n_heads"

        self.d_out = cfg['emb_dim']
        self.n_head = cfg['n_heads']
        self.head_dim = self.d_out // self.n_head

        self.W_k = nn.Linear(cfg['emb_dim'], cfg['emb_dim'])
        self.W_v = nn.Linear(cfg['emb_dim'], cfg['emb_dim'])
        self.W_q = nn.Linear(cfg['emb_dim'], cfg['emb_dim'])
        self.dropout = nn.Dropout(cfg['drop_rate'])
        self.out_proj = nn.Linear(cfg['emb_dim'], cfg['emb_dim'])
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(cfg['context_length'], cfg['context_length']),
                       diagonal=1)
        )

    def forward(self, x):
        bsz, n, d_in = x.shape
        
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # apply rotary embedding
        Q, K = apply_rotary_emb(Q, K) 
        
        
        K = K.view(bsz, n, self.n_head, self.head_dim).transpose(1,2)
        Q = Q.view(bsz, n, self.n_head, self.head_dim).transpose(1,2)
        V = V.view(bsz, n, self.n_head, self.head_dim).transpose(1,2)

        attention_scores = Q @ K.transpose(2,3)
        mask_bool = self.mask.bool()[:n, :n]
        attention_scores.masked_fill_(mask_bool, -torch.inf)
        attention_weight = torch.softmax(attention_scores / K.shape[-1] ** 0.5, dim = -1)
        attention_weight = self.dropout(attention_weight)
        context_vector = (attention_weight @ V).transpose(1, 2)
        context_vector = context_vector.contiguous().view(bsz, n, self.d_out)
        context_vector = self.out_proj(context_vector)
        return context_vector

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm = (x - mean)/torch.sqrt(var + self.eps)
        return self.scale * norm + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return  0.5 * x * torch.tanh((1 + torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3)) ))

class FeedForward(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.fc = nn.Linear(emb_dim, 4 * emb_dim)
        # self.gelu = GELU()
        self.silu = nn.SiLU()
        self.fc2 = nn.Linear(4 * emb_dim, emb_dim)
    def forward(self, x):
        return self.fc2(self.silu(self.fc(x)))

class Tranformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mha = MultiHeadAttention(cfg)
        self.ff = FeedForward(cfg['emb_dim'])
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.dropout = nn.Dropout(cfg['drop_rate'])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.mha(x)
        x = self.dropout(x)
        x = shortcut + x

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = shortcut + x
        return x

class GPT1(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.dropout = nn.Dropout(cfg['drop_rate'])
        self.blocks = nn.Sequential(
            *[Tranformer(cfg) for _ in range(cfg['n_layers'])]
        )
        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.final_layer = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)
    def forward(self, ids):
        bsz, n = ids.shape
        tok_embds = self.tok_emb(ids)
        pos_embds = self.pos_emb(torch.arange(n, device = ids.device))
        x = tok_embds + pos_embds
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.final_norm(x)
        x = self.final_layer(x)
        return x