import torch
import numpy as np
import torch.nn.functional as F

class StandardDiffusion:
    def __init__(self, num_sampling_steps=1000):
        self.num_sampling_steps = num_sampling_steps

    def compute_loss(self, model, x_start, class_labels):
        '''
        Args:
            model: The DiT model
            x_start: The batch of clean images from your dataset (e.g., CIFAR or MedMNIST).
            class_labels: The integer labels for what those images are.
        '''

        batch_size = x_start.shape[0] #each image is a batch
        device = x_start.device

        t = torch.rand((batch_size,), device=device) #selects the timestep each image/sample is analyzed at
        t_expand = t.view(batch_size, 1, 1, 1) #reshape t to be broadcastable with x_start

        noise = torch.randn_like(x_start) #get xT noised image for all batch

        # 3. cosine scheduler instead of beta scheduler
        alpha_t = torch.cos(t_expand * 3.14159 / 2) ** 2
        sigma_t = torch.sin(t_expand * 3.14159 / 2) ** 2

        xt = torch.sqrt(alpha_t) * x_start + torch.sqrt(sigma_t) * noise #obtain the noised image through simplified parametrization

        # 4. model predict
        t_int = (t * 1000).long()
        predicted_noise = model(xt, t_int, class_labels) #if want TEXT BASED (instead of class based), replace with LLM embeddings

        # 5. calculate loss
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def sample_simple(self, model, num_images, device, class_labels, img_size=32, in_channels=3):

        model.eval()
        xt = torch.randn(num_images, in_channels, img_size, img_size, device=device) #
        dt = 1.0 / self.num_sampling_steps

        for step in reversed(range(self.num_sampling_steps)):
            t_float = step / self.num_sampling_steps
            t_int = torch.full((num_images,), int(t_float * 1000), device=device, dtype=torch.long)

            predicted_noise = model(xt, t_int, class_labels)

            sigma_t = torch.sin(torch.tensor(t_float * 3.14159 / 2, device=device)) + 1e-5
            score = -predicted_noise / sigma_t
            xt = xt + score * dt 

        model.train()
        return xt

    
    @torch.no_grad()
    def sample_ALN(self, model, num_images, device, class_labels, r=0.5, n_steps_per_t=5):
        model.eval()
        xt = torch.randn(num_images, 3, 32, 32, device=device)
        
        for step in reversed(range(self.num_sampling_steps)):
            t_float = step / self.num_sampling_steps
            t_int = torch.full((num_images,), int(t_float * 1000), device=device, dtype=torch.long) #creates vector of timestep that has len batch_size
            sigma_t = torch.sin(torch.tensor(t_float * 3.14159 / 2, device=device)) + 1e-5
            
            for _ in range(n_steps_per_t):
                predicted_noise = model(xt, t_int, class_labels)
                score = -predicted_noise / sigma_t


                grad_norm = torch.norm(score.view(score.shape[0], -1), dim=-1).mean()
                noise_norm = np.sqrt(np.prod(xt.shape[1:]))
                eps = 2 * (r * noise_norm / grad_norm) ** 2
                
                # x += force + sqrt(2 * eps) * z
                z = torch.randn_like(xt)
                xt = xt + (eps * score) + torch.sqrt(2 * eps) * z
                
        return xt

    @torch.no_grad()
    def sample_pc(self, model, num_images, device, class_labels, r=0.5):
        model.eval()
        xt = torch.randn(num_images, 3, 32, 32, device=device)
        dt = 1.0 / self.num_sampling_steps

        for step in reversed(range(self.num_sampling_steps)):
            t_float = step / self.num_sampling_steps
            t_int = torch.full((num_images,), int(t_float * 1000), device=device, dtype=torch.long)
            sigma_t = torch.sin(torch.tensor(t_float * 3.14159 / 2, device=device)) + 1e-5
            
            score_pred = -model(xt, t_int, class_labels) / sigma_t
            
            xt = xt + score_pred * dt 

            if step > 0:
                score_corr = -model(xt, t_int, class_labels) / sigma_t
                
                grad_norm = torch.norm(score_corr.view(score_corr.shape[0], -1), dim=-1).mean()
                noise_norm = np.sqrt(np.prod(xt.shape[1:]))
                eps = 2 * (r * noise_norm / grad_norm) ** 2
                
                z = torch.randn_like(xt)
                xt = xt + (eps * score_corr) + torch.sqrt(2 * eps) * z

        model.train()
        return xt