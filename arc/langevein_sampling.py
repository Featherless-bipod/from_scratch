#try to sample from a normal distribution 
import numpy as np
import matplotlib.pyplot as plt

a = 0
b = 1

def F_uniform(x, potential_walls):
    force = np.zeros_like(x)
    

    mask_left = x < a
    force[mask_left] = -potential_walls * (x[mask_left] - a)
    
    mask_right = x > b
    force[mask_right] = -potential_walls * (x[mask_right] - b) 
    
    return force

def sample_langevin_vectorized(n_steps, eps, k, n_samples):
    x = np.full(n_samples, 0.5)
    

    for _ in range(n_steps):
        z = np.random.normal(0, 1, n_samples)
        x += F_uniform(x, k) + np.sqrt(2 * eps) * z
        
    return x
samples = sample_langevin_vectorized(
    n_steps = 30_000,
    eps = 0.00001, 
    k = 3,
    n_samples = 10_000
)


plt.hist(samples, bins=100, density=True, alpha=0.6, color='b')
plt.title("Langevin Sampling Distribution (Uniform Potential)")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()