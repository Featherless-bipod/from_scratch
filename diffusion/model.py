import torch
import torch.nn as nn
import math

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        def block(in_f, out_f, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, 4, stride, 1, bias=False),
                nn.BatchNorm2d(out_f),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.model = nn.Sequential(
            block(in_channels, 64),   # 16x16
            block(64, 128),            # 8x8
            block(128, 256),           # 4x4
            nn.Conv2d(256, 1, 4, 1, 0, bias=False), # 1x1
        )

    def forward(self, x):
        return self.model(x).view(-1, 1)

class PatchEmbed(nn.Module):
    '''
    Embedding module for image input
    '''
    def __init__(self, img_size=32, patch_size=4, in_channels=3, hidden_dim=256):
        '''
        Args:
            img_size (int): The size of the input image.
            patch_size (int): The size of the patches.
            in_channels (int): The number of input channels (eg. Greyscale: 1, RGB: 3)
            hidden_dim (int): The dimension of the hidden layer.
        '''
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        #divides the image into patch_size*patch_size patches and represents each patch as a hidden_dim sized embedding vector 

    def forward(self, x):
        '''
        Args:
            x (torch.tensor): Input tensor of shape (Batch, Channels, Height, Width)
        
        Returns:
            torch.tensor: Output tensor of shape (Batch, Num_Patches, Hidden_Dim)
        '''
        x = self.proj(x)  # shape: (Batch, Hidden_Dim, grid_H, grid_W)
        x = x.flatten(2)  # shape: (Batch, Hidden_Dim, Num_Patches)
        x = x.transpose(1, 2) # shape: (Batch, Num_Patches, Hidden_Dim)—> cuz expect word desc before word embedding
        return x


def modulate(x, shift, scale):
    #used to encorporate the condition vector into the convoluted patches
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, cond_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # creates predictors for the adaLN modulate that will be used
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 4 * hidden_dim)
        )
        #set all initialized values to 0
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c):
        #converts the conditioning vector "c"
        shift_msa, scale_msa, shift_mlp, scale_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
        
        # 1) Attention Step (with adaLN)
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm) #self.attn(query, key, value)
        x = x + attn_out
        
        # 2) MLP Step (with adaLN)
        x_norm2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + self.mlp(x_norm2)
        
        return x



class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim, freq_dim=256):
        """
        freq_dim: How wide the initial sine/cosine barcode should be.
        hidden_dim: Your DiT's hidden dimension (to match the image tokens).
        """
        super().__init__()
        self.freq_dim = freq_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, t):
        # t shape: (Batch_Size,) containing integer timesteps like [500, 20, 999]
        
        # 1) Create the frequencies (the speeds of the different waves)
        half_dim = self.freq_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        
        # 2) Multiply the timestep 't' by the frequencies
        emb = t[:, None] * emb[None, :]
        
        # 3. Apply Sine to the first half, Cosine to the second half, and glue them together
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        
        # 4. Pass the resulting barcode through the MLP
        return self.mlp(emb)


class MicroDiT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, hidden_dim=256, depth=6, num_heads=8, num_classes=10):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, hidden_dim)
        self.num_patches = self.patch_embed.num_patches
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        
        #self.time_embedder = TimestepEmbedder(hidden_dim = hidden_dim, freq_dim = 256) #LEARN HOW TO IMPLEMENT THE COSINE VERSION

        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),#can replace for cosine 
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.class_emb = nn.Embedding(num_classes, hidden_dim)
        
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, cond_dim=hidden_dim) for _ in range(depth)
        ])
        

        #for sending back into an image
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_layer = nn.Linear(hidden_dim, patch_size * patch_size * in_channels)

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.img_size = img_size

    def forward(self, x, t, class_labels):
        # 1)Embed
        x = self.patch_embed(x) + self.pos_embed
        
        # 2)Condition
        t_emb = self.time_mlp(t.unsqueeze(-1).float())
        c_emb = self.class_emb(class_labels)
        condition = t_emb + c_emb 
        
        # 3) DiT
        for block in self.blocks:
            x = block(x, condition)
            
        # 4) Deconvolute
        x = self.norm_final(x)
        x = self.final_layer(x) # Shape: (Batch, Num_Patches, Pixels_Per_Patch)
        
        # 5) Fold back into image (Batch, Channels, Height, Width)
        p = self.patch_size
        h = w = self.img_size // p
        x = x.reshape(x.shape[0], h, w, p, p, self.in_channels)
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(x.shape[0], self.in_channels, self.img_size, self.img_size)
        
        return x


#Gemini Test
if __name__ == "__main__":
    # Simulate images
    dummy_images = torch.randn(8, 3, 32, 32)
    
    # Simulate random timesteps and class labels
    dummy_timesteps = torch.randint(0, 1000, (8,))
    dummy_classes = torch.randint(0, 10, (8,))
    
    # Initialize the model
    model = MicroDiT(img_size=32, in_channels=3)
    
    # Forward pass
    output = model(dummy_images, dummy_timesteps, dummy_classes)
    
    print(f"Input shape: {dummy_images.shape}")
    print(f"Output shape: {output.shape}")
    
    if dummy_images.shape == output.shape:
        print("SUCCESS! The engine works.")
    else:
        print("ERROR: Shapes do not match.")