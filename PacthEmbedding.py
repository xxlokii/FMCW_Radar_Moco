import torch.nn as nn
import torch.nn.functional as F
import timm

# # Resnet10 based patchembedding
class ResnetPatchEmbed(nn.Module):
    def __init__(self, num_channels = 6, feature_dim = 256, embed_dim = 256, model_name = 'resnet10t.c3_in1k', pretrained = False):
        super().__init__()
        self.frame_cnn = timm.create_model(model_name, num_classes = feature_dim, in_chans = num_channels, pretrained= pretrained)
        self.frame_proj = nn.Linear(feature_dim, embed_dim)

        # Prvenet the feature too small
        self.frame_cnn.maxpool = nn.Identity()
        self.frame_cnn.conv1.stride = (1, 1)
    
    def forward(self, x):
        b, t, c ,h ,w = x.shape
        x_frames = x.view(b * t, c, h, w)
        features = self.frame_cnn(x_frames)
        features = self.frame_proj(features)
        features = features.view(b, t, -1)
        return features