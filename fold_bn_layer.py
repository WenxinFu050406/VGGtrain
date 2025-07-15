import torch
import torch.nn as nn
import copy

def bn_folding_model(model):
    new_model = copy.deepcopy(model)
    module_names = list(new_model._modules)

    for k, name in enumerate(module_names):
        if len(list(new_model._modules[name]._modules)) > 0:
            new_model._modules[name] = bn_folding_model(new_model._modules[name])
        else:
            if isinstance(new_model._modules[name], (nn.BatchNorm2d, nn.BatchNorm1d)):
                prev_module = new_model._modules[module_names[k - 1]]
                if isinstance(prev_module, nn.Conv2d):
                    folded_layer = fold_conv_bn_eval(prev_module, new_model._modules[name])
                elif isinstance(prev_module, nn.Linear):
                    folded_layer = fold_linear_bn_eval(prev_module, new_model._modules[name])
                else:
                    continue
                new_model._modules.pop(name)
                new_model._modules[module_names[k - 1]] = folded_layer

    return new_model

def bn_folding(weight, bias, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    if bias is None:
        bias = bn_rm.new_zeros(bn_rm.shape)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    if len(weight.shape) == 4:  # 卷积层
        w_fold = weight * (bn_w * bn_var_rsqrt).view(-1, 1, 1, 1)
    elif len(weight.shape) == 2:  # 全连接层
        w_fold = weight * (bn_w * bn_var_rsqrt).view(-1, 1)
    b_fold = (bias - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return torch.nn.Parameter(w_fold), torch.nn.Parameter(b_fold)

def fold_conv_bn_eval(conv, bn):
    assert (not (conv.training or bn.training)), "Fusion only for eval!"
    fused_conv = copy.deepcopy(conv)

    fused_conv.weight, fused_conv.bias = bn_folding(fused_conv.weight, fused_conv.bias,
                                                    bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)

    return fused_conv

def fold_linear_bn_eval(linear, bn):
    assert (not (linear.training or bn.training)), "Fusion only for eval!"
    fused_linear = copy.deepcopy(linear)

    fused_linear.weight, fused_linear.bias = bn_folding(fused_linear.weight, fused_linear.bias,
                                                        bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)

    return fused_linear

# 加载你的VGG模型
def load_vgg_model(model_path):
    # 假设你有一个OptimizedVGG类定义
    from trainVGG import OptimizedVGG
    model = OptimizedVGG('vgg16')  # 根据你的模型类型调整
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # 确保模型处于评估模式
    return model

# 合并BN层并保存新模型
def merge_bn_and_save(original_model_path, merged_model_path):
    # 加载原始模型
    model = load_vgg_model(original_model_path)  # 原始模型

    # 合并BN层
    merged_model = bn_folding_model(model)

    # 修改全连接层的键名
    state_dict = merged_model.state_dict()
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'classifier.1' not in key:  # 过滤掉批量归一化层的键
            if 'classifier.2' in key:
                new_key = key.replace('classifier.2', 'classifier.1')
            elif 'classifier.3' in key:
                new_key = key.replace('classifier.3', 'classifier.2')
            elif 'classifier.4' in key:
                new_key = key.replace('classifier.4', 'classifier.3')
            else:
                new_key = key
            new_state_dict[new_key] = value

    # 保存合并后的模型
    torch.save(new_state_dict, merged_model_path)
    print(f"已保存合并BN层后的模型到: {merged_model_path}")

    return model, merged_model  # 返回原始模型和合并模型

# 使用示例
if __name__ == "__main__":
    original_model_path = "vgg16_best_model.pth"
    merged_model_path = "vgg16_merged_bn_model.pth"

    # 获取原始模型和合并模型
    original_model, merged_model = merge_bn_and_save(original_model_path, merged_model_path)

    # 验证合并后的模型
    test_input = torch.randn(1, 3, 32, 32)  # 输入尺寸需与模型匹配（如CIFAR-10的32x32）
    with torch.no_grad():
        original_output = original_model(test_input)  # 使用原始模型
        merged_output = merged_model(test_input)     # 使用合并模型

    # 检查输出差异（应接近0）
    max_diff = torch.max(torch.abs(original_output - merged_output)).item()
    print(f"原始模型与合并后模型的最大输出差异: {max_diff:.10f}")