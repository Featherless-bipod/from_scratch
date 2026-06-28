import torch
from torch.distributions.bernoulli import Bernoulli

def pad_mask(mask, data, value):
    """Adds padding to I/O masks for CD and SV in cases where
    reconstructed data is not the same shape as the input data.
    """
    t_forward = data.shape[1] - mask.shape[1]
    n_heldout = data.shape[2] - mask.shape[2]
    pad_shape = (0, n_heldout, 0, t_forward)
    return torch.nn.functional.pad(mask, pad_shape, value=value)

class CoordinatedDropout:
    def __init__(self, cd_rate):
        self.cd_rate = cd_rate
        self.cd_input_dist = Bernoulli(cd_rate)
        self.cd_pass_dist = Bernoulli(1.-cd_rate)

    def process_batch(self, batch):
        encod_data = batch
        maskable_data = encod_data
        device = encod_data.device
        cd_mask = self.cd_input_dist.sample(maskable_data.shape).to(device)
        pass_mask = self.cd_pass_dist.sample(maskable_data.shape).to(device)
        grad_mask = torch.logical_or(torch.logical_not(cd_mask), pass_mask).float()
        cd_masked_data = maskable_data * cd_mask / self.cd_rate
        cd_input = cd_masked_data

        return cd_input, grad_mask

    def process_losses(self, recon_loss, cd_mask):
        cd_mask = pad_mask(cd_mask, recon_loss, 1.0)
        grad_loss = recon_loss * cd_mask
        nograd_loss = (recon_loss * (1 - cd_mask)).detach()
        cd_loss = grad_loss + nograd_loss
        return cd_loss

class SampleValidation:
    def __init__(self, sv_rate, fwd_steps, heldin_neurons):
        self.sv_rate = sv_rate
        self.sv_input_dist = Bernoulli(sv_rate)
        self.heldin_neurons = heldin_neurons
        self.fwd_steps = fwd_steps

    def process_batch(self, batch):

        unmasked_data1 = batch[:,:-self.fwd_steps,:]
        unmasked_data2 = batch[:,-self.fwd_steps:,:self.heldin_neurons]
        masked_data = batch[:,-self.fwd_steps:,self.heldin_neurons:]

        device = batch.device
        sv_mask = self.sv_input_dist.sample(masked_data.shape).to(device)
        pass_mask1 = torch.ones_like(unmasked_data1).to(device)
        pass_mask2 = torch.ones_like(unmasked_data2).to(device)
        masked_data = masked_data * sv_mask / self.sv_rate

        sv_data = torch.cat([unmasked_data2,masked_data],dim=2)
        sv_data = torch.cat([unmasked_data1,sv_data], dim=1)
        sv_mask = torch.cat([pass_mask2,sv_mask],dim=2)
        sv_mask = torch.cat([pass_mask1,sv_mask],dim=1)

        return sv_data, sv_mask

    def process_losses(self, recon_loss, sv_mask):
        grad_loss = recon_loss * sv_mask
        nograd_loss = (recon_loss * (1 - sv_mask)).detach()
        sv_loss = grad_loss + nograd_loss
        return sv_loss
