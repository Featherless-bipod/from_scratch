import torch
from torch import nn
from MHA.MultiHeadAttention import MultiHeadAttention


class MsaAttnRow(nn.Module):
    def __init__(self, c_m, c_p, c = 16, N_head = 8):
        # m is msa rep (*, N_seq, N_res, c_m)
        # p is pair rep (*, N_res, N_res, c_p)

        self.linear_p  = nn.Linear(c_p, N_head)
        self.layer_norm = nn.LayerNorm(c_m) 
        self.mha = MultiHeadAttention(c_m,c, N_head, attn_dim=-2, gated = True, use_bias_for_embeddings=True)

    def forward(self, m, p):
        m = self.layer_norm(m)
        p = self.linear_p(p)

        p = p.movedim(-1,-3)
        out = self.mha(m, bias = p)

        return out
    

class MsaAttnCol(nn.Module):
    def __init__(self, c_m, c = 16, N_head = 8):
        # m is msa rep (*, N_seq, N_res, c_m)
        self.mha = MultiHeadAttention(c_m, c, N_head, attn_dim= -3)
    
    def forward(self, m):
        out = self.mha(m)

        return out


class Transition(nn.Module):
    def __init__(self, c_m, n = 4):
        self.transition = nn.Sequential(
            nn.LayerNorm(c_m),
            nn.Linear(c_m, c_m*n),
            nn.ReLU(),
            nn.Linear(c_m*n, c_m)
        )
    def forward(self, m):
        out = self.transition(m)

        return out

class OuterProductMean(nn.Module): #REMEMBER THE LEAST still kinda confused ???
    def __init__(self, c_m, c_p, c=32):
        self.layer_norm = nn.LayerNorm(c_m)
        self.linear_a = nn.Linear(c_m, c)
        self.linear_b = nn.Linear(c_m,c)
        self.linear_out = nn.Linear(c*c, c_p)
    
    def forward(self, m):
        N_seq = m.shape[-3]

        m = self.layer_norm(m)
        a = self.linear_a(m)
        b = self.linear_b(m)
        o = torch.einsum('...abc, ...xyz->...bycz') #keep the second to last N_res and ending dimension C 
        sum = torch.flatten(o, start_dim = -2)
        out = self.linear_out(sum)/N_seq

        return out
    

class SharedDropout(nn.Module):
    """
    A module for dropout, that is shared along one dimension,
    i.e. for dropping out whole rows or columns.
    """
    def __init__(self, shared_dim: int, p: float):
        super().__init__()

        ##########################################################################
        # TODO: Store shared_dim for later use and initialize an                 #
        #        nn.Dropout module for the forward pass.                         #
        ##########################################################################

        self.dropout = nn.Dropout(p)
        self.shared_dim = shared_dim

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, x: torch.tensor):
        """
        Forward pass for shared dropout. The dropout mask is broadcasted along
        the shared dimension.

        Args:
            x (torch.tensor): Input tensor of arbitrary shape.

        Returns:
            torch.tensor: Output tensor of the same shape as x.
        """

        out = None

        ##########################################################################
        # TODO: Apply shared dropout by implementing the following steps:        #
        #        * Create a mask of ones with the same shape as x, but with      #
        #           dim 1 at the shared dimension.                               #
        #        * Apply dropout to the mask and multiply it against x to mask   #
        #           out the values. The mask is implicitly broadcasted to the    #
        #           shape of x.                                                  #
        ##########################################################################

        mask_shape = list(x.shape)
        mask_shape[self.shared_dim] = 1
        mask = torch.ones(mask_shape, device=x.device)
        mask = self.dropout(mask)

        out = x * mask

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return out

class DropoutRowwise(SharedDropout):
    def __init__(self, p: float):
        ##########################################################################
        # TODO: Initialize the super class by choosing the right shared          #
        #        dimension for row-wise dropout.                                 #
        ##########################################################################
        super().__init__(shared_dim=-2, p=p)
        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

class DropoutColumnwise(SharedDropout):
    def __init__(self, p: float):
        ##########################################################################
        # TODO: Initialize the super class by choosing the right shared          #
        #        dimension for column-wise dropout.                              #
        ##########################################################################
        super().__init__(shared_dim=-3, p=p)
        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################


class PairTriMult(nn.Module):
    def __init__(self, c_p,c=32):
        
        self.layernorm1 = nn.LayerNorm(c_p)
        self.layernorm2 = nn.LayerNorm(c)

        self.linear_a_g = nn.Linear(c_p, c)
        self.linear_b_g = nn.Linear(c_p, c)
        self.sigmoid = nn.Sigmoid()
        self.linear_a_p = nn.Linear(c_p, c)
        self.linear_b_p = nn.Linear(c_p, c)

        self.linear_g = nn.Linear(c_p, c)
        self.linear_out = nn.Linear(c, c_p)
        self.rowdropout = DropoutRowwise(0.05)

    def forwrard(self, p, direction: str):
        p = self.layernorm1(p)
        a = self.sigmoid(self.linear_a_g(p)) * self.linear_a_p(p)
        b = self.sigmoid(self.linear_b_g(p)) * self.linear_b_p(p)

        if direction == 'outgoing':
            o = torch.einsum('...ikc, ...jkc -> ...ijc', a,b)
        else: 
            o = torch.einsum('...kic, ...kjc -> ...ijc', a,b)

        o = self.sigmoid(self.linear_g(o)) * o
        out = self.rowdropout(self.linear_out(o))
        return out


class PairTriAttn(nn.Module):
    def __init__(self, c_p, direction: str, c=32, N_head = 8): 
        
        if direction is not {'start_node', 'end_node'}:
            return ValueError
        self.direction = direction

        if direction == 'start_node':
            self.mha = MultiHeadAttention(c_p, c, N_head, attn_dim = -2, use_bias_for_embeddings=True)
        else: 
            self.mha = MultiHeadAttention(c_p, c, N_head, attn_dim = -3, use_bias_for_embeddings= True)
        
        self.dropout_row = DropoutRowwise(0.05)
        self.dropout_col = DropoutColumnwise(0.05)


    def forward(self, p):

        p = self.layer_norm(p)
        bias = self.linear_bias(p) 
        bias = bias.movedim(-1, -3)
        if self.direction == 'end_node': 
            bias = bias.movedim(-1,-2)
        
        out = self.mha(p, bias = bias)
        if self.direction == 'start_node':
            out = self.dropout_row(out)
        else:
            out = self.dropout_col(out)

        return out
    
class EvoformerBlock(nn.Module):
    def __init__(self, c_m, c_p): 

        self.dropout_row = SharedDropout(0.15)

        self.msa_row_attn = MsaAttnRow(c_m, c_p)
        self.msa_col_attn = MsaAttnCol(c_m)
        self.msa_transition = Transition(c_m)
        self.outer_product_mean = OuterProductMean(c_m, c_p)
        self.pair_tri_mult = PairTriMult(c_p)
        self.pair_tri_attn_start = PairTriAttn(c_p, 'start_node')
        self.pair_tri_attn_end = PairTriAttn(c_p, 'end_node')
        self.pair_transition = Transition(c_p)

    def forward(self, m, p):
        
        m = m + self.dropout_row(self.msa_row_attn(m))
        m = m + self.msa_col_attn(m)
        m = m + self.msa_transition(m)
        
        p = p + self.outer_product_mean(m,p)
        p = p + self.pair_tri_mult(p, 'start_node')
        p = p + self.pair_tri_mult(p, 'end_node')
        p = p + self.pair_tri_attn_start(p)
        p = p + self.pair_tri_attn_end(p)
        p = p + self.pair_transition(p)

        return m, p

class EvoformerStack(nn.Module):
    def __init__(self, c_m, c_p, num_blocks, c_s=384):
        super().__init__()

        self.blocks = nn.ModuleList([EvoformerBlock(c_m, c_p) for _ in range(num_blocks)])
        self.linear = nn.Linear(c_m, c_s) #for single embedding 

    def forward(self, m, p):
        for block in self.blocks: 
            m, p = block(m, p)
        
        single = self.linear(m[...,0,:,:])

        return m, p, single