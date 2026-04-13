import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F
import torch.nn as nn
from model import Discriminator

class FlowMatcher:
    def __init__(self, num_sampling_steps=50, epsilon=0.01, iterations=10):
        self.num_sampling_steps = num_sampling_steps
        self.epsilon = epsilon
        self.iterations = iterations

    def compute_loss_flow(self, model, x0, class_labels):
        """
        x0: The batch of clean images from your dataset (e.g., CIFAR or MedMNIST).
        class_labels: The integer labels for what those images are.
        """
        batch_size = x0.shape[0]
        device = x0.device

        t = torch.rand((batch_size,), device=device)
        t_expand = t.view(batch_size, 1, 1, 1)

        xT = torch.randn_like(x0)

        xt = (1 - t_expand) * xT + t_expand * x0

        target_velocity = x0 - xT

        t_int = (t * 1000).long()
        predicted_velocity = model(xt, t_int, class_labels)
        loss = F.mse_loss(predicted_velocity, target_velocity)
        
        return loss

    def compute_loss_hungarian_OT(self, model, x0, class_labels):
        batch_size = x0.shape[0]
        device = x0.device

        t = torch.rand((batch_size,), device=device)
        t_expand = t.view(batch_size, 1, 1, 1)

        xT = torch.randn_like(x0)

        x0_flat = x0.view(batch_size, -1)
        xT_flat = xT.view(batch_size, -1)

        #calculate cost
        with torch.no_grad():
            cost_matrix = torch.cdist(x0_flat, xT_flat, p=2)

            # hungarian algorithm (used for discrete)
            cost_np = cost_matrix.cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np) #makes each cell in the pariwise matrix binary (0 or 1) to indicate optimal pairs

        # reorder
        xT_optimal = xT[col_ind]
        
        xt = (1 - t_expand) * xT_optimal + t_expand * x0
        target_velocity = x0 - xT_optimal
        
        t_int = (t * 1000).long()
        predicted_velocity = model(xt, t_int, class_labels)
        loss = F.mse_loss(predicted_velocity, target_velocity)
        
        return loss

    def compute_loss_sinkhorn_OT(self, model, x0, class_labels):
        batch_size = x0.shape[0]
        device = x0.device

        t = torch.rand((batch_size,), device=device)
        t_expand = t.view(batch_size, 1, 1, 1)

        xT = torch.randn_like(x0)

        x0_flat = x0.view(batch_size, -1)
        xT_flat = xT.view(batch_size, -1)

        with torch.no_grad(): #almost think of as applying softmax to smash the distribution
            cost_matrix = torch.cdist(x0_flat, xT_flat, p=2)
            # Normalize to securely prevent exponential underflow (NaNs)
            cost_matrix = cost_matrix / (cost_matrix.max() + 1e-8)

            P = torch.exp(-cost_matrix / self.epsilon) #epsilon controls the smoothness of the distribution
            
            P = P / (P.sum(dim=1, keepdim=True) + 1e-8)
            
            for _ in range(self.iterations):
                P = P / (P.sum(dim=0, keepdim=True) + 1e-8)
                P = P / (P.sum(dim=1, keepdim=True) + 1e-8) #makes sure sum to 1 both column and row-wise 

        xT_optimal = (P @ xT_flat).view_as(xT) #calculates the distance (essentially a sum over all distance*probability in the plan)

        xt = (1 - t_expand) * xT_optimal + t_expand * x0

        target_velocity = x0 - xT_optimal

        t_int = (t * 1000).long()
        predicted_velocity = model(xt, t_int, class_labels)
        loss = F.mse_loss(predicted_velocity, target_velocity)
        
        return loss
    
    def compute_loss_adversarial(self, model, discriminator, x0, class_labels, lambda_adv=0.1):
        #NOTE: treats DiT as a generator

        batch_size = x0.shape[0]
        device = x0.device

        loss_flow = self.compute_loss_sinkhorn_OT(model, x0, class_labels)

        #create false image
        xT = torch.randn_like(x0)
        t_zero = torch.zeros((batch_size,), device=device).long()
        
        #one step euler jump
        v_pred = model(xT, t_zero, class_labels)
        x_fake = torch.tanh(xT + v_pred * 1.0)

        #calculate discriminator loss
        pred_real = discriminator(x0)
        # One-Sided Label Smoothing to prevent Discriminator over-confidence
        loss_D_real = F.binary_cross_entropy_with_logits(pred_real, torch.full_like(pred_real, 0.9))#all_real

        pred_fake = discriminator(x_fake.detach()) # .detach() so don't update DiT yet
        loss_D_fake = F.binary_cross_entropy_with_logits(pred_fake, torch.zeros_like(pred_fake)) #all fake

        loss_D = (loss_D_real + loss_D_fake) / 2 #for training discirminator

        #generator loss
        pred_fake_for_G = discriminator(x_fake)
        loss_G_adv = F.binary_cross_entropy_with_logits(pred_fake_for_G, torch.ones_like(pred_fake_for_G))

        # total loss
        total_gen_loss = loss_flow + (lambda_adv * loss_G_adv)

        return total_gen_loss, loss_D
        

    @torch.no_grad()
    def sample_euler(self, model, num_images, device, class_labels, img_size=32):
        model.eval()
        xt = torch.randn(num_images, 3, img_size, img_size, device=device)
        dt = 1.0 / self.num_sampling_steps

        for step in range(self.num_sampling_steps):
            t_float = step / self.num_sampling_steps
            t_int = torch.full((num_images,), int(t_float * 1000), device=device, dtype=torch.long)
            
            velocity = model(xt, t_int, class_labels)
            xt = xt + velocity * dt
            
        return xt

    @torch.no_grad()
    def sample_aln(self, model, num_images, device, class_labels, n_steps_per_t=5, r=0.1):
        model.eval()
        xt = torch.randn(num_images, 3, 32, 32, device=device)
        
        for step in range(self.num_sampling_steps):
            t_float = step / self.num_sampling_steps
            t_int = torch.full((num_images,), int(t_float * 1000), device=device, dtype=torch.long)
            
            for _ in range(n_steps_per_t):
                v = model(xt, t_int, class_labels)
                
                grad_norm = torch.norm(v.view(num_images, -1), dim=-1).mean()
                noise_norm = np.sqrt(np.prod(xt.shape[1:]))
                eps = 2 * (r * noise_norm / grad_norm)**2
                
                z = torch.randn_like(xt)
                xt = xt + (eps * v) + torch.sqrt(2 * eps) * z
                
        return xt

    @torch.no_grad()
    def sample_pc(self, model, num_images, device, class_labels, r=0.1, img_size=32):
        model.eval()
        xt = torch.randn(num_images, 3, img_size, img_size, device=device)
        dt = 1.0 / self.num_sampling_steps

        for step in range(self.num_sampling_steps):
            t_float = step / self.num_sampling_steps
            t_int = torch.full((num_images,), int(t_float * 1000), device=device, dtype=torch.long)
            
            v = model(xt, t_int, class_labels)
            xt = xt + v * dt 

            if step < self.num_sampling_steps - 1:
                v_corr = model(xt, t_int, class_labels)
                grad_norm = torch.norm(v_corr.view(num_images, -1), dim=-1).mean()
                noise_norm = np.sqrt(np.prod(xt.shape[1:]))
                step_size = 2 * (r * noise_norm / grad_norm)**2
                
                z = torch.randn_like(xt)
                xt = xt + step_size * v_corr + torch.sqrt(2 * step_size) * z
                
        return xt