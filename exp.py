import torch
import torchvision.models as models

# 加载预训练的 VGG16 模型
vgg16 = models.vgg16(pretrained=False)

# 计算总参数量
total_params = sum(p.numel() for p in vgg16.parameters())
print(f"VGG16 的总参数量: {total_params}")
