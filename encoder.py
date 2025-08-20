import torch
import torch.nn as nn
from transformer_block import Block

# FeatureEncoder class for encoding input features with optional dynamic masking
class FeatureEncoder(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, encoder_depth=6, mask_ratio=0.5, num_patches=32):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.num_patches = num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))  
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(encoder_depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    # Random masking function to mask input features based on the specified mask ratio
    def random_masking(self, x, mask_ratio=None):
        """
        Randomly masks input features.
        :param x: Input tensor of shape [N, L, D]
        :param mask_ratio: Ratio of patches to mask (optional). If None, use self.mask_ratio.
        :return: Masked tensor.
        """
        N, L, D = x.shape
        mask_ratio = self.mask_ratio if mask_ratio is None else mask_ratio 
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        return x_masked
    
    
    def forward(self, x, mask_ratio=None):
        """
        Forward pass with optional dynamic mask_ratio.
        :param x: Input tensor of shape [B, T, D]
        :param mask_ratio: Optional mask ratio for dynamic masking (overrides self.mask_ratio).
        :return: Class output after encoding.
        """
        B, T, _ = x.shape
        pos_embed = self.pos_embed[:, 1:T+1, :] 
        x = x + pos_embed
        x = self.random_masking(x, mask_ratio)
        cls_token = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, 0:1, :]
        x = torch.cat([cls_token, x], dim=1)

        x = self.blocks(x)
        x = self.norm(x)

        # Use the class token and apply a dynamic scaling factor
        cls_out = x[:, 0] + torch.sigmoid(self.alpha) * x[:, 1:].max(dim=1)[0] + (1 - torch.sigmoid(self.alpha)) * x[:, 1:].mean(dim=1)
        return cls_out
    