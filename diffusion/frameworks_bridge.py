import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

class SchrodingerBridge:
    def __init__(self, num_sampling_steps=50, sigma=0.1, epsilon=0.01, iterations=10):
        self.num_sampling_steps = num_sampling_steps
        self.sigma = sigma              # bridge volatility
        self.epsilon = epsilon          # sinkhorn smoothing
        self.iterations = iterations    # sinkhorn rounds

    def compute_loss_simple(self, model, x0, x1, class_labels):
        batch_size = x0.shape[0]
        t = torch.rand((batch_size, 1, 1, 1), device=x0.device)
        
        std_dev = torch.sqrt(t * (1 - t) + 1e-5) * self.sigma
        xt = (1 - t) * x0 + t * x1 + std_dev * torch.randn_like(x0)

        target_velocity = x1 - x0
        pred_velocity = model(xt, (t.squeeze() * 1000).long(), class_labels)
        
        return F.mse_loss(pred_velocity, target_velocity)

    def compute_loss_ot_sinkhorn(self, model, x0, x1, class_labels):
        batch_size = x0.shape[0]
        device = x0.device

        x0_flat, x1_flat = x0.view(batch_size, -1), x1.view(batch_size, -1)
        with torch.no_grad():
            C = torch.cdist(x0_flat, x1_flat, p=2)**2
            # Normalize C to prevent exponential underflow (NaNs)
            C = C / (C.max() + 1e-8)
            
            P = torch.exp(-C / self.epsilon)
            for _ in range(self.iterations):
                P /= (P.sum(dim=0, keepdim=True) + 1e-8)
                P /= (P.sum(dim=1, keepdim=True) + 1e-8)
        
        # 2. construct otptimally loss targets
        x1_opt = P @ x1_flat
        x1_opt = x1_opt.view_as(x1)

        # 3. standard bridge loss with sinkhorn implementation
        return self.compute_loss_simple(model, x0, x1_opt, class_labels)

    def compute_loss_adversarial(self, model, discriminator, x0, x1, class_labels, lambda_adv=0.1):
        # 1. base schrodinger loss
        loss_sb = self.compute_loss_ot_sinkhorn(model, x0, x1, class_labels)

        # 2. critic
        t_zero = torch.zeros((x0.shape[0],), device=x0.device).long()
        v_pred = model(x0, t_zero, class_labels)
        x_fake = torch.tanh(x0 + v_pred * 1.0) # 1-step leap

        # 3. dscriminator loss
        p_real, p_fake = discriminator(x1), discriminator(x_fake.detach())
        loss_D = (F.binary_cross_entropy_with_logits(p_real, torch.ones_like(p_real)) + 
                  F.binary_cross_entropy_with_logits(p_fake, torch.zeros_like(p_fake))) / 2

        # 4. generator adv loss
        loss_G_adv = F.binary_cross_entropy_with_logits(discriminator(x_fake), torch.ones_like(p_real))
        
        return loss_sb + (lambda_adv * loss_G_adv), loss_D

    @torch.no_grad()
    def sample_euler(self, model, x_start, class_labels):
        model.eval()
        xt = x_start.clone()
        dt = 1.0 / self.num_sampling_steps
        for step in range(self.num_sampling_steps):
            t_int = torch.full((xt.shape[0],), int((step/self.num_sampling_steps)*1000), device=xt.device, dtype=torch.long)
            xt += model(xt, t_int, class_labels) * dt
        return xt

    @torch.no_grad()
    def sample_pc(self, model, x_start, class_labels, r=0.1):
        model.eval()
        xt = x_start.clone()
        dt = 1.0 / self.num_sampling_steps
        for step in range(self.num_sampling_steps):
            t_int = torch.full((xt.shape[0],), int((step/self.num_sampling_steps)*1000), device=xt.device, dtype=torch.long)
            # predictor
            v = model(xt, t_int, class_labels)
            xt += v * dt
            # langevin steps
            if step < self.num_sampling_steps - 1:
                v_corr = model(xt, t_int, class_labels)
                eps = 2 * (r * np.sqrt(np.prod(xt.shape[1:])) / (torch.norm(v_corr.view(xt.shape[0], -1), dim=-1).mean() + 1e-8))**2
                xt += (eps * v_corr) + torch.sqrt(2 * eps) * torch.randn_like(xt)
        return xt

    @torch.no_grad()
    def sample_aln(self, model, x_start, class_labels, n_steps=5, r=0.1):
        model.eval()
        xt = x_start.clone()
        for step in range(self.num_sampling_steps):
            t_int = torch.full((xt.shape[0],), int((step/self.num_sampling_steps)*1000), device=xt.device, dtype=torch.long)
            for _ in range(n_steps):
                v = model(xt, t_int, class_labels)
                eps = 2 * (r * np.sqrt(np.prod(xt.shape[1:])) / (torch.norm(v.view(xt.shape[0], -1), dim=-1).mean() + 1e-8))**2
                xt += (eps * v) + torch.sqrt(2 * eps) * torch.randn_like(xt)
        return xt