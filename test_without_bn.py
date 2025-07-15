import torch
import torch.nn as nn
from torchvision.models import vgg16, vgg19, vgg11
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# -------------------- 配置参数 --------------------
BATCH_SIZE = 256
NUM_EPOCHS = 100
NUM_CLASSES = 10
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
VAL_RATIO = 0.2  # 20%训练集作为验证集
FREEZE_LAYERS = 13  # 冻结前10层卷积层
INIT_LR = 1e-4
MIN_LR = 1e-6

# -------------------- 数据增强 --------------------
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------- 数据加载 --------------------
print("⌛ 加载CIFAR-10数据集...")
test_dataset = datasets.CIFAR10(root='./data', train=False,
                                download=True, transform=eval_transform)


# 创建数据加载器
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=4, pin_memory=True)

print(f"📊测试集: {len(test_dataset)}张")


# -------------------- 优化的VGG16模型 --------------------
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


# -------------------- 评估函数 --------------------
def evaluate(model, data_loader, criterion, device):
    model.eval()
    loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return loss / len(data_loader), 100 * correct / total


# -------------------- 测试函数 --------------------
def test_model(model_path, test_loader, device, model_name="模型", model_type='vgg19'):
    model = OptimizedVGGWithoutBN(model_type).to(device)
    model.load_state_dict(torch.load(model_path))

    test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"\n🔍 {model_name} behaves:")
    print(f"loss: {test_loss:.4f} | accuracy: {test_acc:.2f}%")
    return test_acc


# -------------------- 主函数 --------------------
def main():
    model_type = 'vgg11'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️ device: {device}")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("✅ 已启用cuDNN自动优化")

    print("\n🧪 begin testing...")
    final_acc = test_model("./vgg11_best_model_without_bn.pth", test_loader, device, "VGG11_without_bn", model_type)
    print(f"Model accuracy: {final_acc:.2f}%")

    print("\n🎉 Finished!")


if __name__ == "__main__":
    main()
