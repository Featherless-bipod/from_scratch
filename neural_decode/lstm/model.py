# Copyright (c) NXAI GmbH and its affiliates 2023
# Korbininan Pöppel

import os
import sys
import torch
from torch import nn
import pytorch_lightning as pl

from utils import CoordinatedDropout, SampleValidation


from typing import Dict, Tuple

import os
from torch.utils.cpp_extension import load

# Compile and load the C++ extension on-the-fly
_curr_dir = os.path.dirname(os.path.abspath(__file__))
slstm_cpp = load(
    name="slstm_cpp",
    sources=[os.path.join(_curr_dir, "slstm_cell.cpp")],
    verbose=True,
    extra_cflags=["-O3"]
)

def slstm_forward_pointwise(
    Wx: torch.Tensor,  # dim [B, 4*H]
    Ry: torch.Tensor,  # dim [B, 4*H]
    b: torch.Tensor,  # dim [1, 4*H]
    states: torch.Tensor,  # dim [4, B, H]
    constants: Dict[str, float],
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
]:
    # Call the highly optimized C++ implementation
    new_states, gates = slstm_cpp.slstm_forward_pointwise(Wx, Ry, b, states)
    return new_states, gates


class sLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wx = nn.Linear(input_size, 4 * hidden_size, bias=False)
        self.Ry = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.b = nn.Parameter(torch.zeros(1, 4 * hidden_size))
        
    def forward(self, x, states, constants=None):
        if constants is None:
            constants = {}
        wx = self.Wx(x)
        ry = self.Ry(states[0]) # states[0] is y (the previous output state)
        new_states, gates = slstm_forward_pointwise(wx, ry, self.b, states, constants)
        return new_states, gates


class BaseAutoencoder(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        fwd_steps: int,
        learning_rate: float,
        weight_decay: float,
        dropout: float,
        cd_rate: float,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dropout = nn.Dropout(p=dropout)
        self.readout = nn.Linear(hidden_size, output_size)
        
        self.cd = CoordinatedDropout(cd_rate=cd_rate)
        self.sv = SampleValidation(sv_rate=0.8, fwd_steps=fwd_steps, heldin_neurons=input_size)
        self.switch_epoch_l2 = 500.

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer):
        l2_ramp = (self.current_epoch) / (self.switch_epoch_l2)
        l2_ramp = torch.clamp(torch.tensor(l2_ramp), 0, 1)
        optimizer.param_groups[0]["weight_decay"] = l2_ramp * self.hparams.weight_decay

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=0.95,
            patience=10,
            threshold=0.0,
            min_lr=1e-5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "hp_metric",
        }

    def training_step(self, batch, batch_ix):
        input_data, recon_data, behavior = batch
        cd_data, cd_mask = self.cd.process_batch(input_data)
        
        cd_preds, _ = self.forward(cd_data, use_logrates=True)

        target_len = recon_data.shape[1]
        cd_preds = cd_preds[:, :target_len, :]

        cd_nll_loss = nn.functional.poisson_nll_loss(cd_preds, recon_data, reduction='none')
        cd_nll_loss = self.cd.process_losses(cd_nll_loss, cd_mask)
        loss = cd_nll_loss.mean()

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_ix):
        torch.set_grad_enabled(True)
        if len(batch) == 1:
            (input_data,) = batch
            preds, _ = self.forward(input_data, use_logrates=True)
            _, n_obs, n_heldin = input_data.shape
            preds = preds[:, :n_obs, :n_heldin]
            recon_data = input_data
        else:
            input_data, recon_data, behavior = batch
            preds, _ = self.forward(input_data, use_logrates=True)

        target_time = recon_data.shape[1]
        target_neurons = recon_data.shape[2]
        
        preds = preds[:, :target_time, :target_neurons]

        sv_data, sv_mask = self.sv.process_batch(recon_data)
        
        loss = nn.functional.poisson_nll_loss(preds, sv_data, reduction='none')
        loss = self.sv.process_losses(loss, sv_mask)
        loss = loss.mean()
        
        self.log("valid/loss", loss, prog_bar=True)
        self.log("hp_metric", loss)
        return loss


class LSTMAutoencoder(BaseAutoencoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cell = nn.LSTMCell(self.hparams.input_size, self.hparams.hidden_size)
        
    def forward(self, observ, use_logrates=True):
        batch_size, obs_steps, _ = observ.shape
        hx = torch.zeros(batch_size, self.hparams.hidden_size, device=observ.device)
        cx = torch.zeros(batch_size, self.hparams.hidden_size, device=observ.device)
        
        logrates_list = []
        latents_list = []
        
        for t in range(obs_steps + self.hparams.fwd_steps):
            if t < obs_steps:
                inp = observ[:, t]
            else:
                # Autoregressive / forecasting fallback to zero inputs
                inp = torch.zeros_like(observ[:, 0])
                
            hx, cx = self.cell(inp, (hx, cx))
            
            hx_dropped = self.dropout(hx)
            logrates = self.readout(hx_dropped)
            
            logrates_list.append(logrates)
            latents_list.append(hx)
            
        logrates = torch.stack(logrates_list, dim=1)
        latents = torch.stack(latents_list, dim=1)
        
        if not use_logrates:
            return torch.exp(logrates), latents
        return logrates, latents


class sLSTMAutoencoder(BaseAutoencoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cell = sLSTMCell(self.hparams.input_size, self.hparams.hidden_size)
        
    def forward(self, observ, use_logrates=True):
        batch_size, obs_steps, _ = observ.shape
        
        # states tensor has shape [4, batch_size, hidden_size] mapping to (y, c, n, m)
        states = torch.zeros(4, batch_size, self.hparams.hidden_size, device=observ.device)
        
        logrates_list = []
        latents_list = []
        
        for t in range(obs_steps + self.hparams.fwd_steps):
            if t < obs_steps:
                inp = observ[:, t]
            else:
                inp = torch.zeros_like(observ[:, 0])
                
            states, gates = self.cell(inp, states)
            
            # y is states[0]
            hx = states[0]
            
            hx_dropped = self.dropout(hx)
            logrates = self.readout(hx_dropped)
            
            logrates_list.append(logrates)
            latents_list.append(hx)
            
        logrates = torch.stack(logrates_list, dim=1)
        latents = torch.stack(latents_list, dim=1)
        
        if not use_logrates:
            return torch.exp(logrates), latents
        return logrates, latents