import torch
import torch.nn as nn
from torchvision.models import vit_b_16
# from torchvision.models import vit_base_patch14_224 as vit_base_14 

class GCDModel(nn.Module):
    def __init__(self, num_classes, pretrained_type="dino"):
        super(GCDModel, self).__init__()
        self.num_classes = num_classes
        
        # 根据指定的预训练类型选择骨干网络
        if pretrained_type == "dino":
            self.backbone = vit_b_16(pretrained=False)
            weights_path = "../pretrained/DINO/dino_vitbase16_pretrain.pth"
        # elif pretrained_type == "dinov2":
        #     self.backbone = vit_base_14(pretrained=False)
        #     weights_path = "path_to_directory/dinov2_vitb14_pretrain.pth"
        else:
            raise ValueError("Invalid pretrained_type. Choose either 'dino' or 'dinov2'.")
        
        # 加载预训练权重
        state_dict = torch.load(weights_path)
        self.backbone.load_state_dict(state_dict, strict=False)
        
        # 创建分类头
        self.head = nn.Linear(self.backbone.head.in_features, num_classes)
        # # Define the MLP head
        # in_features = self.backbone.head.in_features
        # self.head = MLPHead(in_features, hidden_features=512, out_features=num_classes)


    def forward(self, x):
        feature_backbone = self.backbone(x)
        feature_head = self.head(feature_backbone)
        return feature_backbone, feature_head
    
    def freeze(self, num_layers_to_freeze=None):
        if num_layers_to_freeze is None:
            num_layers_to_freeze = len(self.backbone.blocks) - 1
        
        for i, block in enumerate(self.backbone.blocks):
            if i < num_layers_to_freeze:
                for param in block.parameters():
                    param.requires_grad = False
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))



class MLPHead(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout_prob=0.5):
        super(MLPHead, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
