import torch

# 函数形式，方便计算重建图像的峰值信噪比
def PSNR_function(prediction, label): # 输出的是tensor类型的一个标量
    diff = torch.clamp(prediction, 0, 1) - torch.clamp(label, 0, 1)
    rmse = (diff ** 2).mean().sqrt()
    psnr = 20 * torch.log10(1 / rmse)
    return psnr
    
