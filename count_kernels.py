import torch.nn as nn
from vgg_without_bn import OptimizedVGGWithoutBN

def count_conv_kernels(model):
    """
    计算模型中卷积层的卷积核总元素数量
    
    参数:
        model: 输入的PyTorch模型
    
    返回:
        total_kernel_elements: 所有卷积核的总元素数量
        kernel_elements_per_layer: 每个卷积层的卷积核元素数量
    """
    total_kernel_elements = 0
    kernel_elements_per_layer = {}
    
    # 遍历模型的所有模块
    for name, module in model.named_modules():
        # 检查是否为卷积层
        if isinstance(module, nn.Conv2d):
            # 获取卷积层的参数
            out_channels = module.out_channels
            in_channels = module.in_channels
            kernel_size = module.kernel_size
            
            # 计算当前卷积层的卷积核总元素数量
            if isinstance(kernel_size, tuple):
                elements_per_kernel = in_channels * kernel_size[0] * kernel_size[1]
            else:  # 如果kernel_size是单个整数
                elements_per_kernel = in_channels * kernel_size * kernel_size
                
            layer_elements = elements_per_kernel * out_channels
            kernel_elements_per_layer[name] = layer_elements
            total_kernel_elements += layer_elements
    
    return total_kernel_elements, kernel_elements_per_layer

def count_total_parameters(model):
    """
    计算模型的总参数数量
    
    参数:
        model: 输入的PyTorch模型
    
    返回:
        total_params: 模型的总参数数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def count_3x3_kernels(model):
    """
    计算模型中所有3×3卷积核的总个数（每个输出通道对应一组cin个卷积核）
    
    参数:
        model: 输入的PyTorch模型
    
    返回:
        total_3x3_kernels: 3×3卷积核的总个数（按cin*cout计算）
        kernels_per_layer: 每层3×3卷积核的个数
    """
    total_3x3_kernels = 0
    kernels_per_layer = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # 获取卷积核尺寸
            if isinstance(module.kernel_size, tuple):
                kh, kw = module.kernel_size
            else:
                kh = kw = module.kernel_size
            
            # 只统计3×3卷积核
            if kh == 3 and kw == 3:
                in_channels = module.in_channels
                out_channels = module.out_channels
                layer_kernels = in_channels * out_channels
                kernels_per_layer[name] = layer_kernels
                total_3x3_kernels += layer_kernels
    
    return total_3x3_kernels, kernels_per_layer

def count_kernels_by_size(model):
    """按尺寸统计卷积核数量（每个输出通道对应一组cin个卷积核）"""
    kernel_stats = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # 获取卷积核尺寸
            if isinstance(module.kernel_size, tuple):
                kh, kw = module.kernel_size
            else:
                kh = kw = module.kernel_size
            
            kernel_size_str = f"{kh}x{kw}"
            
            # 计算该层卷积核数量（cin*cout）
            in_channels = module.in_channels
            out_channels = module.out_channels
            layer_kernels = in_channels * out_channels
            
            # 初始化或累加对应尺寸的卷积核数量
            if kernel_size_str not in kernel_stats:
                kernel_stats[kernel_size_str] = 0
            kernel_stats[kernel_size_str] += layer_kernels
    
    return kernel_stats
# 使用示例
if __name__ == "__main__":
    # 创建模型实例
    model = OptimizedVGGWithoutBN(model_type='vgg19')
    
    # 计算卷积核元素总量
    total_kernel_elements, kernel_elements_per_layer = count_conv_kernels(model)
    
    # 计算总参数数量
    total_params = count_total_parameters(model)
    
    # 计算3×3卷积核个数
    total_3x3, kernels_per_layer_3x3 = count_3x3_kernels(model)
    
    # 按尺寸统计卷积核
    kernel_stats_by_size = count_kernels_by_size(model)
    
    # 打印结果
    print(f"模型中卷积核的总元素数量: {total_kernel_elements}")
    print(f"模型中3×3卷积核的总个数: {total_3x3}")
    print(f"模型的总参数数量: {total_params}")
    

    


