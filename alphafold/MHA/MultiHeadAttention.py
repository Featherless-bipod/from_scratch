import torch
import math
from torch import nn

class MultiHeadAttention(nn.Module):

    def __init__(self, c_in, c, N_head, attn_dim, gated=False, is_global=False, use_bias_for_embeddings=False):

        super().__init__()

        self.c_in = c_in
        self.c = c
        self.N_head = N_head
        self.gated = gated
        self.attn_dim = attn_dim
        self.is_global = is_global

        self.linear_q = nn.Linear(c_in,N_head*c,bias = use_bias_for_embeddings)
        
        c_kv = c if is_global else c*N_head
        self.linear_k = nn.Linear(c_in,c_kv, bias = use_bias_for_embeddings)
        self.linear_v = nn.Linear(c_in,c_kv, bias = use_bias_for_embeddings)

        self.linear_o = nn.Linear(N_head*c,c_in)

        if gated:
            self.linear_g = nn.Linear(c_in,c*N_head)


    def prepare_qkv(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):

        q = q.movedim(self.attn_dim,-2)
        k = k.movedim(self.attn_dim,-2)
        v = v.movedim(self.attn_dim,-2)

        q_shape = q.shape[:-1] + (self.N_head, -1)
        k_shape = k.shape[:-1] + (self.N_head, -1)
        v_shape = v.shape[:-1] + (self.N_head, -1)

        q = q.view(q_shape)
        k = k.view(k_shape)
        v = v.view(v_shape)

        q = q.transpose(-3,-2)
        k = k.transpose(-3,-2)
        v = v.transpose(-3,-2)

        return q, k, v

    def prepare_qkv_global(self, q, k, v):

        q = q.movedim(self.attn_dim,-2)
        k = k.movedim(self.attn_dim,-2)
        v = v.movedim(self.attn_dim,-2)

        q_shape = q.shape[:-1] + (self.N_head, self.c)
        q = q.view(q_shape)
        q = q.transpose(-2,-3)
        q = torch.mean(q,dim= -2,keepdim=True)

        k = k.unsqueeze(-3)
        v = v.unsqueeze(-3)

        return q, k, v

    def forward(self, x, bias=None, attention_mask=None):
        out = None

        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        if self.is_global:
            q,k,v = self.prepare_qkv_global(q,k,v)
        else:
            q,k,v = self.prepare_qkv(q,k,v)
            
        q = q/math.sqrt(self.c)

        a = torch.einsum('...qc,...kc->...qk',q,k)
        if bias is not None:   
            bias_batch_shape = bias.shape[:-3]
            bias_bc_shape = bias_batch_shape + (1,) * (a.ndim - len(bias_batch_shape) - 3) + bias.shape[-3:]
            bias = bias.view(bias_bc_shape)

            a = a + bias
            
        if attention_mask is not None:
            attention_mask = attention_mask[...,None,None,:]
            offset = (attention_mask == 0 )* -1e8
            a = a + offset
        
        a = torch.softmax(a,dim=-1)

        o = torch.einsum('...qk,...kc->...qc',a, v)
        o = o.transpose(-3,-2)
        o = torch.flatten(o, start_dim=-2)
        o = o.movedim(-2,self.attn_dim)
        if self.gated:
            g = nn.Sigmoid(self.linear_g(x))
            o = g * o
        
        out =  self.linear_o(o)

        return out
