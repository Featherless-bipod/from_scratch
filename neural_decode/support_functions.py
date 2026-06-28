import torch


def to_tensor(array):
    """
    loaded numpy array into tensor with correct dtype
    """
    return torch.tensor(array, dtype = torch.float)


def create_recon_data(heldin, heldin_forward, heldout, heldout_forward):
    """
    creates the recon_data via concatenating the heldin and heldout (obs and forward)

    Args:
    - heldin (B, T, N)
    - heldin_foward (B, T_fwd, N)
    - heldout (B, T, N)
    - heldout_forward (B, T_fwd, N)

    Return: 
    - recon_data (B, T + T_fwd, 2 * N)

    """
    heldin_all = torch.cat([heldin, heldin_forward], dim = 1)
    heldout_all = torch.cat([heldout, heldout_forward], dim = 1)
    recon_data = torch.cat([heldin_all, heldout_all], dim = 2)
    return recon_data

def create_gaussian_kernel(num_channel, kernel_size, sigma):
    mean = (kernel_size - 1) / 2.0
    x_cord = torch.arange(kernel_size).view(1,kernel_size).repeat(num_channel,1).to(sigma)
    gaussian_kernel = (1./(2.*np.pi))*torch.exp(-(x_cord-mean)**2/(2*sigma**2) )
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel,dim=1,keepdim=True)
    gaussian_kernel = gaussian_kernel.view(1, num_channel, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(num_channel, 1, 1)
    return gaussian_kernel.float()
