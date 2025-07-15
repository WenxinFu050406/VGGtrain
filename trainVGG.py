import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16, vgg19, vgg11
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from tqdm import tqdm
import time

# -------------------- 配置参数 --------------------
BATCH_SIZE = 256
NUM_EPOCHS = 100
NUM_CLASSES = 10
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
VAL_RATIO = 0.2  # 20%训练集作为验证集
FREEZE_LAYERS = 10  # 冻结前10层卷积层
INIT_LR = 1e-4
MIN_LR = 1e-6

# -------------------- 数据增强 --------------------
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------- 数据加载 --------------------
print("⌛ 加载CIFAR-10数据集...")
train_val_dataset = datasets.CIFAR10(root='./data', train=True,
                                     download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root='./data', train=False,
                                download=True, transform=eval_transform)

# 分割训练集为训练和验证 (80:20)
train_size = int((1 - VAL_RATIO) * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
train_dataset, val_dataset = random_split(
    train_val_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=4, pin_memory=True)

print(f"\n📊 数据集划分:")
print(f"训练集: {len(train_dataset)}张")
print(f"验证集: {len(val_dataset)}张")
print(f"测试集: {len(test_dataset)}张")


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


# -------------------- 训练函数 --------------------
def train_model(model, train_loader, val_loader, device, model_name):
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=INIT_LR,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        # 训练阶段
        for images, labels in tqdm(train_loader, desc=f"{model_name} Epoch {epoch + 1}/{NUM_EPOCHS}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            train_loss += loss.item()

        # 计算训练指标
        train_loss /= len(train_loader)
        train_acc = 100 * correct_train / total_train

        # 验证阶段
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # 学习率调整
        scheduler.step(val_acc)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{model_name}_best_model.pth")
            print(f"\n💾 保存 {model_name} 最佳模型 | 验证准确率: {val_acc:.2f}%")

        # 打印日志
        print(f"\n{model_name} Epoch {epoch + 1}/{NUM_EPOCHS} 结果:")
        print(f"训练集: 损失={train_loss:.4f} | 准确率={train_acc:.2f}%")
        print(f"验证集: 损失={val_loss:.4f} | 准确率={val_acc:.2f}%")
        print(f"当前学习率: {optimizer.param_groups[0]['lr']:.2e}")

    # 训练结束时保存最终模型
    torch.save(model.state_dict(), f"{model_name}_final_model.pth")
    print(f"\n💾 保存 {model_name} 最终模型")


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
def test_model(model_path, test_loader, device, model_name="模型"):
    model = OptimizedVGG(model_name.split("_")[0]).to(device)
    model.load_state_dict(torch.load(model_path))

    test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"\n🔍 {model_name}在测试集上的表现:")
    print(f"测试损失: {test_loss:.4f} | 测试准确率: {test_acc:.2f}%")
    return test_acc


# -------------------- 主函数 --------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️ 使用设备: {device}")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("✅ 已启用cuDNN自动优化")

    model_types = [ 'vgg19']
    test_results = {}

    for model_type in model_types:
        # 初始化模型
        print(f"\n🔄 初始化 {model_type} 模型...")
        model = OptimizedVGG(model_type).to(device)

        # 打印参数信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数: {total_params:,} | 可训练参数: {trainable_params:,}")

        # 开始训练
        print(f"\n🔥 开始训练 {model_type}")
        start_time = time.time()
        train_model(model, train_loader, val_loader, device, model_type)
        print(f"\n⏱️ {model_type} 总训练时间: {(time.time() - start_time) / 60:.1f}分钟")

        # 测试两个模型
        print(f"\n🧪 开始测试 {model_type} 阶段...")
        best_acc = test_model(f"{model_type}_best_model.pth", test_loader, device, f"{model_type} 最佳模型")
        final_acc = test_model(f"{model_type}_final_model.pth", test_loader, device, f"{model_type} 最终模型")

        test_results[model_type] = (best_acc, final_acc)

    # 打印所有模型的测试结果对比
    print("\n📊 所有模型测试结果对比:")
    for model_type, (best_acc, final_acc) in test_results.items():
        print(f"{model_type} 最佳模型测试准确率: {best_acc:.2f}%")
        print(f"{model_type} 最终模型测试准确率: {final_acc:.2f}%")

    # 保存结果到文件
    with open("test_results.txt", "w") as f:
        for model_type, (best_acc, final_acc) in test_results.items():
            f.write(f"{model_type} 最佳模型测试准确率: {best_acc:.2f}%\n")
            f.write(f"{model_type} 最终模型测试准确率: {final_acc:.2f}%\n")

    print("\n🎉 所有流程完成!")


if __name__ == "__main__":
    main()
