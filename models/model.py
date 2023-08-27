import torch
import torch.nn as nn
from .vision_transformer import vit_base
import torch.nn.functional as F
# from torchvision.models import vit_base_patch14_224 as vit_base_14 


class GCDModel(nn.Module):
    def __init__(self, num_classes, pretrained_type="dino"):
        super(GCDModel, self).__init__()
        self.num_classes = num_classes

        self.backbone = self._get_backbone(pretrained_type)
        in_features = self.backbone.blocks[-1].norm1.weight.shape[0]

        self.head = MLPHead(in_features, hidden_features=2048, out_features=256)
        self.prototype = Prototypes(in_features, num_classes)
        self.normalize_prototypes()

    def _get_backbone(self, pretrained_type):
        if pretrained_type == "dino":
            backbone = vit_base()
            weights_path = "./pretrained/DINO/dino_vitbase16_pretrain.pth"
        # elif pretrained_type == "dinov2":
        #     self.backbone = vit_base_14(pretrained=False)
        #     weights_path = "path_to_directory/dinov2_vitb14_pretrain.pth"
        else:
            raise ValueError("Invalid pretrained_type. Only 'dino' is supported for now.")
        
        state_dict = torch.load(weights_path)
        backbone.load_state_dict(state_dict)
        return backbone
    
    def forward(self, x):
        feature_backbone = self.backbone(x)

        feature_proj = self.head(feature_backbone)
        feature_proj = F.normalize(feature_proj, dim=-1, p=2)

        feature_norm = F.normalize(feature_backbone, dim=-1, p=2)
        logits = self.prototype(feature_norm)
        return feature_proj, logits
    
    def freeze(self, num_layers_to_freeze=None):
        if num_layers_to_freeze is None:
            # If not provided, freeze all layers
            num_layers_to_freeze = float('inf')

        # Initially freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Go through parameters and decide which ones to unfreeze
        for name, param in self.backbone.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])  
                if block_num >= num_layers_to_freeze:
                    param.requires_grad = True
                    print(f'Finetuning layer {name}')
    
    def save(self, path, save_dict):
        torch.save(save_dict, path)

    def load(self, path):
        checkpoint = torch.load(path)
        
        # Load model state
        self.load_state_dict(checkpoint['model'])
        
        # Return optimizer state and epoch 
        optimizer_state = checkpoint['optimizer']
        epoch = checkpoint['epoch']

        return optimizer_state, epoch
    
    @torch.no_grad()
    def normalize_prototypes(self):
        self.prototype.normalize_prototypes()


# hidden_features = 2048
class MLPHead(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout_prob=0.5):
        super(MLPHead, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu1 = nn.GELU()
        # self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.relu2 = nn.GELU()
        self.fc3 = nn.Linear(hidden_features, out_features)

        # Initialize weights
        self.apply(self._init_weights)


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Prototypes(nn.Module):
    def __init__(self, output_dim, num_prototypes):
        super().__init__()

        self.prototypes = nn.Linear(output_dim, num_prototypes, bias=False)

    @torch.no_grad()
    def normalize_prototypes(self):
        self.prototypes = nn.utils.weight_norm(self.prototypes)
        self.prototypes.weight_g.data.fill_(1)
        self.prototypes.weight_g.requires_grad = False
    def forward(self, x):
        return self.prototypes(x)
    