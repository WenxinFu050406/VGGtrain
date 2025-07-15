import torch
import torch.nn as nn
from torchvision.models import vgg11, vgg16, vgg19

# -------------------- 配置参数 --------------------
BATCH_SIZE = 256
NUM_EPOCHS = 100
NUM_CLASSES = 10
VAL_RATIO = 0.2  # 20%训练集作为验证集
FREEZE_LAYERS = 10  # 冻结前10层卷积层
INIT_LR = 1e-4
MIN_LR = 1e-6

# -------------------- 优化的VGG模型 --------------------
class OptimizedVGG(nn.Module):
    def __init__(self, model_type):
        super().__init__()
        if model_type == 'vgg11':
            vgg = vgg11(pretrained=True)
        elif model_type == 'vgg16':
            vgg = vgg16(pretrained=True)
        elif model_type == 'vgg19':
            vgg = vgg19(pretrained=True)
        else:
            raise ValueError("Unsupported model type. Choose from 'vgg11', 'vgg16', or 'vgg19'.")

        # 冻结指定层
        for idx, layer in enumerate(vgg.features.children()):
            if idx < FREEZE_LAYERS:
                for param in layer.parameters():
                    param.requires_grad = False

        self.features = vgg.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # 优化分类头
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, NUM_CLASSES)
        )

        # 初始化分类头权重
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 定义去除批量归一化层后的新模型类
class OptimizedVGGWithoutBN(nn.Module):
    def __init__(self, model_type):
        super().__init__()
        if model_type == 'vgg11':
            vgg = vgg11(pretrained=True)
        elif model_type == 'vgg16':
            vgg = vgg16(pretrained=True)
        elif model_type == 'vgg19':
            vgg = vgg19(pretrained=True)
        else:
            raise ValueError("Unsupported model type. Choose from 'vgg11', 'vgg16', or 'vgg19'.")

        # 冻结指定层
        for idx, layer in enumerate(vgg.features.children()):
            if idx < FREEZE_LAYERS:
                for param in layer.parameters():
                    param.requires_grad = False

        self.features = vgg.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # 优化分类头，去除批量归一化层
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, NUM_CLASSES)
        )

        # 初始化分类头权重
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 加载现成的模型
model_type = 'vgg16'
model_path = f"{model_type}_best_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
original_model = OptimizedVGG(model_type).to(device)
original_model.load_state_dict(torch.load(model_path))

# 创建去除批量归一化层的新模型实例
new_model = OptimizedVGGWithoutBN(model_type).to(device)

# 复制除批量归一化层外的参数
state_dict = original_model.state_dict()
new_state_dict = new_model.state_dict()
for name, param in state_dict.items():
    if 'classifier.1' not in name:  # 跳过批量归一化层
        if 'classifier' in name:
            # 调整分类头层的索引
            parts = name.split('.')
            if int(parts[1]) > 1:
                parts[1] = str(int(parts[1]) - 1)
            new_name = '.'.join(parts)
            new_state_dict[new_name] = param
        else:
            new_state_dict[name] = param

# 将新的状态字典加载到新模型中
new_model.load_state_dict(new_state_dict)

# 保存去除批量归一化层后的新模型
new_model_path = f"{model_type}_best_model_without_bn.pth"
torch.save(new_model.state_dict(), new_model_path)
print(f"已保存去除批量归一化层后的模型到 {new_model_path}")
