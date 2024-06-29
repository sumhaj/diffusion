import torch
import numpy as np
import math

class NoiseScheduler:
    def __init__(self, num_timesteps):
        self.num_timesteps = num_timesteps

    def linear_noise_scheduler(self, beta_start, beta_end):
        self.betas = torch.linspace(beta_start, beta_end, self.num_timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cum_prod = self.alphas_cum_prod.sqrt()
        self.sqrt_one_minus_alphas_cum_prod = (1 - self.alphas_cum_prod).sqrt()

    def quadratic_noise_scheduler(self, beta_start, beta_end):
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, self.num_timesteps) ** 2
        self.alphas = 1 - self.betas
        self.alphas_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cum_prod = self.alphas_cum_prod.sqrt()
        self.sqrt_one_minus_alphas_cum_prod = (1 - self.alphas_cum_prod).sqrt()

    
    def cosine_noise_scheduler(self, max_beta=0.999):
        def cosine_alpha_cum_prod(t, s=0.008):
            angle = (t/self.num_timesteps + s) / (1 + s) * math.pi / 2
            return torch.cos(angle) ** 2
        
        self.alphas_cum_prod = [cosine_alpha_cum_prod(t)/cosine_alpha_cum_prod(0) for t in range(self.num_timesteps)]
        self.alphas = [min(self.alphas_cum_prod[t+1] / self.alphas_cum_prod[t], max_beta) for t in range(self.num_timesteps)]
        self.alphas = torch.tensor(self.alphas)
        self.betas = 1 - self.alphas
        self.alphas_cum_prod = torch.tensor(self.alphas_cum_prod)
        self.sqrt_alphas_cum_prod = self.alphas_cum_prod.sqrt()
        self.sqrt_one_minus_alphas_cum_prod = (1 - self.alphas_cum_prod).sqrt()
        
    def sigmoid_noise_scheduler(self, beta_start, beta_end, start=-6, end=6):
        def sigmoid(x):
            return 1 / (torch.exp(-x) + 1)
        
        betas = torch.linspace(start, end, self.num_timesteps)
        self.betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
        self.alphas = 1 - self.betas
        self.alphas_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cum_prod = self.alphas_cum_prod.sqrt()
        self.sqrt_one_minus_alphas_cum_prod = (1 - self.alphas_cum_prod).sqrt()


    def apply_noise(self, original, noise, t):
        batch_sqrt_one_minus_alphas_cum_prod = self.sqrt_one_minus_alphas_cum_prod.to(original.device)[t]
        batch_sqrt_alphas_cum_prod = self.sqrt_alphas_cum_prod.to(original.device)[t]
        
        for _ in range(len(original.shape)-1):
            batch_sqrt_one_minus_alphas_cum_prod = batch_sqrt_one_minus_alphas_cum_prod.unsqueeze(-1)
            batch_sqrt_alphas_cum_prod = batch_sqrt_alphas_cum_prod.unsqueeze(-1)

        return batch_sqrt_one_minus_alphas_cum_prod * original + batch_sqrt_alphas_cum_prod * noise
    
    def sample_prev_timestep(self, xt, noise_pred, t):
        x0 = ((xt - (self.sqrt_one_minus_alphas_cum_prod.to(xt.device)[t] * noise_pred)) / 
              torch.sqrt(self.alphas_cum_prod.to(xt.device)[t]))
        x0 = torch.clamp(x0, -1., 1.)

        mean = (xt - self.betas.to(xt.device)[t] / self.sqrt_one_minus_alphas_cum_prod.to(xt.device)[t] * noise_pred) / self.alphas.to(xt.device)[t].sqrt()

        if(t == 0):
            return mean, x0
        
        variance = (1 - self.alphas_cum_prod.to(xt.device)[t-1]) / (1 - self.alphas_cum_prod.to(xt.device)[t]) * self.betas.to(xt.device)[t]
        sigma = variance ** 0.5
        
        z = torch.randn(size=xt.shape).to(xt.device)

        return mean + sigma * z, x0