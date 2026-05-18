import torch
from torch import nn

from Embedding.InputEmbedder import InputEmbedder
from Embedding.RecyclingEmbedder import RecyclingEmbedder
from Embedding.ExtraMSAEmbedder import ExtraMsaStack, ExtraMsaEmbedder
from Evoformer.Evoformer import EvoformerStack
from Structure.StructureModule import StructureModule

class Model(nn.Module):
    
    def __init__(self, c_m=256, c_z=128, c_e=64, f_e = 25, tf_dim=21, c_s=384, num_blocks_extra_msa=4, num_blocks_evoformer=48):
        """
        Initializes the Alphafold model.

        Args:
            c_m (int, optional): Number of channels for the MSA representation. Defaults to 256.
            c_z (int, optional): Number of channels for the pair representation. Defaults to 128.
            c_e (int, optional): Number of channels for the extra MSA representation. Defaults to 64.
            f_e (int, optional): Number of channels of the extra MSA feature. Defaults to 25.
            tf_dim (int, optional): Number of channels of the target feature. Defaults to 22.
            c_s (int, optional): Number of channels for the single representation. Defaults to 384.
            num_blocks_extra_msa (int, optional): Number of blocks for the extra MSA stack. Defaults to 4.
            num_blocks_evoformer (int, optional): Number of blocks for the Evoformer. Defaults to 48.
        """
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_e = c_e
        self.c_s = c_s

        self.input_embedder = InputEmbedder(c_m, c_z, tf_dim)
        self.extra_msa_embedder = ExtraMsaEmbedder(f_e, c_e)
        self.recycling_embedder = RecyclingEmbedder(c_m, c_z)
        self.extra_msa_stack = ExtraMsaStack(c_e, c_z, num_blocks=num_blocks_extra_msa)
        self.evoformer = EvoformerStack(c_m, c_z, num_blocks=num_blocks_evoformer)
        self.structure_module = StructureModule(c_s, c_z)


    def forward(self, batch):
        """
        Forward pass for the Alphafold model.

        Args:
            batch (dict): A dictionary containing the following features:
                * msa_feat:  Tensor of shape (*, N_seq, N_res, msa_feat_dim, N_cycle).
                * extra_msa_feat: Tensor of shape (*, N_extra, N_res, f_e, N_cycle).
                * target_feat: Tensor of shape (*, N_res, tf_dim, N_cycle). One-hot encoding of the target sequence.
                * residue_index: Tensor of shape (*, N_res, N_cycle). The index of each residue, which is [0,...,N_res-1].

        Returns:
            dict: A dictionary with the following entries:
                * final_positions: Heavy-atom positions in Angstrom of shape (*, N_res, 37, 3, N_cycle).
                * position_mask: Boolean tensor of shape (*, N_res, 37, N_cycle), masking atoms that
                    aren't present in the amino acids.
                * angles: Torsion angles of shape (*, N_layers, N_res, n_torsion_angles, 2, N_cycle) for 
                    every iteration of the Structure Module in every cycle.
                * frames: Backbone frames of shape (*, N_layers, N_res, 4, 4, N_cycle) for every iteration
                    of the Structure Module in every cycle.
        """
        N_cycle = batch['msa_feat'].shape[-1]
        N_seq, N_res = batch['msa_feat'].shape[-4:-2]
        batch_shape = batch['msa_feat'].shape[:-4]
        device = batch['msa_feat'].device
        dtype = batch['msa_feat'].dtype

        c_m = self.c_m
        c_z = self.c_z

        outputs = {}

        #Algorithm 2

        prev_m = torch.zeros(batch_shape+(N_seq, N_res, c_m), device=device, dtype=dtype)
        prev_z = torch.zeros(batch_shape+(N_res,N_res,c_z), device=device, dtype=dtype)
        prev_pseudo_beta_x = torch.zeros((N_res, 3), device=device, dtype=dtype)


        for i in range(N_cycle):
            print(f'starting iteration {i}')
            current_batch = {
                key: value[...,i] for key, value in batch.items()
            }
            m, z = self.input_embedder(current_batch)

            m_rec, z_rec = self.recycling_embedder(prev_m, prev_z, prev_pseudo_beta_x)
            m[..., 0, :, :] += m_rec
            z += z_rec

            e = self.extra_msa_embedder(current_batch)
            z = self.extra_msa_stack(e, z)
            del e

            m, z, s = self.evoformer(m, z)

            F = current_batch['target_feat'].argmax(dim=-1)
            structure_output = self.structure_module(s, z, F)
            prev_m = m
            prev_z = z
            prev_pseudo_beta_x = structure_output['pseudo_beta_positions']

            for key,value in structure_output.items():
                if key in outputs:
                    outputs[key].append(value)
                else:
                    outputs[key] = [value]

        outputs = {
            key: torch.stack(value, dim=-1) for key, value in outputs.items()
        }


        return outputs