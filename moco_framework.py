import torch
import torch.nn as nn
from PacthEmbedding import *
from encoder import FeatureEncoder

class Moco_v3(nn.Module):
    def __init__(self, embed_dim=256, encoder_depth=6, mask_ratio=0.2, num_heads=8, mlp_dim=512, num_channels=6):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim

        self.patch_embed = ResnetPatchEmbed(num_channels=num_channels, embed_dim=embed_dim)
        
        self.base_encoder = FeatureEncoder(embed_dim, num_heads, encoder_depth, mask_ratio)
        self.momentum_encoder = FeatureEncoder(embed_dim, num_heads, encoder_depth, mask_ratio)
        
        self._build_projector_and_predictor_mlps(embed_dim, mlp_dim)
        self.momentum_projector = self._build_mlp(3, embed_dim, mlp_dim, embed_dim)
        
        self.initialize_weights()
        self._init_momentum_encoder()
        
        self.momentum_update_rate = 0.996

    def initialize_weights(self):
        if hasattr(self.patch_embed, 'frame_proj') and isinstance(self.patch_embed.frame_proj, nn.Linear):
            nn.init.xavier_uniform_(self.patch_embed.frame_proj.weight)
            if self.patch_embed.frame_proj.bias is not None:
                nn.init.constant_(self.patch_embed.frame_proj.bias, 0)
        elif hasattr(self.patch_embed, 'conv') and isinstance(self.patch_embed.conv, nn.Conv2d):
            nn.init.kaiming_normal_(self.patch_embed.conv.weight, mode='fan_out', nonlinearity='relu')
            if self.patch_embed.conv.bias is not None:
                nn.init.constant_(self.patch_embed.conv.bias, 0)
        if hasattr(self.base_encoder, 'alpha'):
            nn.init.uniform_(self.base_encoder.alpha, 0.3, 0.7)
        if hasattr(self.base_encoder, 'cls_token'):
            nn.init.trunc_normal_(self.base_encoder.cls_token, std=0.02)
        if hasattr(self.base_encoder, 'pos_embed'):
            nn.init.trunc_normal_(self.base_encoder.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
            if m.weight is not None:
                nn.init.constant_(m.weight.float(), 1)
            if m.bias is not None:
                nn.init.constant_(m.bias.float(), 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # Initialize the momentum encoder with the same weights as the base encoder
    def _init_momentum_encoder(self):
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)
            param_m.requires_grad = False

        for param_b, param_m in zip(self.projector.parameters(), self.momentum_projector.parameters()):
            param_m.data.copy_(param_b.data)
            param_m.requires_grad = False

    # Build a multi-layer perceptron (MLP) for the projector and predictor
    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        layers = []
        for l in range(num_layers):
            in_dim = input_dim if l == 0 else mlp_dim
            out_dim = output_dim if l == num_layers - 1 else mlp_dim

            layers.append(nn.Linear(in_dim, out_dim, bias=False))
            if l < num_layers - 1:
                layers.extend([
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(inplace=True)
                ])
            elif last_bn:
                layers.append(nn.BatchNorm1d(out_dim, affine=False))
        return nn.Sequential(*layers)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        self.projector = self._build_mlp(3, dim, mlp_dim, dim)
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, last_bn=False)

    @torch.no_grad()
    def _update_momentum_encoder(self, m=0.99):
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = m * param_m.data + (1 - m) * param_b.data
        for param_b, param_m in zip(self.projector.parameters(), self.momentum_projector.parameters()):
            param_m.data = m * param_m.data + (1 - m) * param_b.data

    def extract_embedding(self, x):
        x = self.patch_embed(x)
        return F.normalize(self.base_encoder(x, mask_ratio=0), dim=1)

    
    def forward_query(self, x):
        x = self.patch_embed(x)
        cls_out = self.base_encoder(x)
        projected = self.projector(cls_out)
        predicted = self.predictor(projected)
        return predicted

    def forward_key(self, x):
        x = self.patch_embed(x)
        cls_out = self.momentum_encoder(x)
        projected = self.momentum_projector(cls_out)
        return projected

    def update_momentum_encoder(self, m=None):
        if m is None:
            m = self.momentum_update_rate
        self._update_momentum_encoder(m)

    
    def forward(self, x1, x2, m=0.996):
        x_cat = torch.cat([self.patch_embed(x1), self.patch_embed(x2)], dim=0)
        cls_out_q = self.base_encoder(x_cat)
        cls_out1_q, cls_out2_q = torch.chunk(cls_out_q, 2)
        q1 = self.predictor(self.projector(cls_out1_q))
        q2 = self.predictor(self.projector(cls_out2_q))
        
        with torch.no_grad():
            self._update_momentum_encoder(m)
            cls_out_k = self.momentum_encoder(x_cat)
            cls_out1_k, cls_out2_k = torch.chunk(cls_out_k, 2)
            k1 = self.momentum_projector(cls_out1_k)
            k2 = self.momentum_projector(cls_out2_k)

        return contrastive_loss(torch.cat([q1, q2]), torch.cat([k2, k1]))

# This is a MoCo v3 model for contrastive learning with momentum encoders and projectors.
class QueryEncoder(nn.Module):
    def __init__(self, moco_model):
        super().__init__()
        self.moco_model = moco_model
        
    def forward(self, x):
        return self.moco_model.forward_query(x)

# This is a KeyEncoder for the MoCo v3 model, which uses the momentum encoder to process input features.
class KeyEncoder(nn.Module):
    def __init__(self, moco_model):
        super().__init__()
        self.moco_model = moco_model
        
    def forward(self, x):
        return self.moco_model.forward_key(x)

# This function computes the contrastive loss between query and key representations.
def contrastive_loss(q, k, T=0.1):
    q, k = F.normalize(q, dim=1), F.normalize(k, dim=1)
    logits = torch.matmul(q, k.T) / T
    labels = torch.arange(q.size(0), device=q.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

# This class implements the MoCo contrastive loss function.
class MoCoContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, q_reps, k_reps, **kwargs):
        batch_size = q_reps.size(0) // 2
        q1, q2 = q_reps[:batch_size], q_reps[batch_size:]
        k1, k2 = k_reps[:batch_size], k_reps[batch_size:]
        
        q = torch.cat([q1, q2], dim=0)  
        k = torch.cat([k2, k1], dim=0)  
        return contrastive_loss(q, k, self.temperature)
