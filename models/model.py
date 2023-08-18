import torch
import torch.nn as nn
from torchvision.models import vit_b_16_224_in21k as vit_b_16

class GCDModel(nn.Module):
    def __init__(self, num_classes):
        super(GCDModel, self).__init__()
        self.num_classes = num_classes
        self.backbone = vit_b_16(pretrained=True)
        self.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        feature_backbone = self.model(x)
        feature_head = self.head(feature_backbone)
        return feature_backbone, feature_head
    
    def freeze(self, num_layers_to_freeze=None):
        if num_layers_to_freeze is None:
            num_layers_to_freeze = len(self.model.encoder.block) - 1
        
        for i, block in enumerate(self.backbone.blocks):
            if i < num_layers_to_freeze:
                for param in block.parameters():
                    param.requires_grad = False
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))