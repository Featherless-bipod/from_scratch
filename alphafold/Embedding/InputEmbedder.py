import torch
from torch import nn

class InputEmbedder(nn.Module):


    def __init__(self, c_m, c_z, tf_dim, msa_feat_dim=49, vbins=32):

        super().__init__()
        self.tf_dim = tf_dim
        self.vbins = vbins

        # Replace "pass" statement with your code
        self.linear_tf_z_i = nn.Linear(tf_dim,c_z)
        self.lienar_tf_z_j = nn.Linear(tf_dim,c_z)
        self.linear_tf_m = nn.Linear(tf_dim, c_m)
        self.linear_msa_m = nn.Linear(msa_feat_dim,c_m)
        self.linear_rel_pos = nn.Linear(2*vbins+1, c_z)

    def relpos(self, residue_index):

        out = None
        dtype = self.linear_relpos.weight.dtype

        res_index = residue_index.long()
        o = res_index.unsqueeze(-1)-res_index.unsqueeze(-2)
        o = torch.clamp(o, -self.vbins, self.vbins) + self.vbins
        o_onehot = nn.functional.one_hot(o, num_classes = 2*self.vbins+1).to(dtype=dtype)
        out = self.linear_rel_pos(o_onehot)

        return out
        

    def forward(self, batch):

        m = None
        z = None

        msa_feat = batch['msa_feat']
        target_feat = batch['target_feat']
        residue_index = batch['residue_index']

        a = self.linear_tf_z_i(target_feat)
        b = self.linear_tf_z_j(target_feat)
        z = a.unsqueeze(-2) + b.unsqueeze(-3)

        z += self.relpos(residue_index)
        target_feat = target_feat.unsqueeze(-3)
        m = self.linear_msa_m(msa_feat) + self.linear_tf_m(target_feat)

        return m, z
