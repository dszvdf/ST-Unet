import time
import torch
import glob
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.dataset import BasicDataset
from utils.data_vis import show_while_predicting
from utils.utils import process_common
from utils.SSIM import SSIM_function
from utils.NPCC import PCC_function
from utils.PSNR import PSNR_function
from utils.SSIM import SSIM_function
from torch.nn.functional import mse_loss as MSE_function
from utils.plot import paper_plot_direct

# 单个推理
def predict_single(model,
                   device,
                   corr2,
                   original_image,
                   experiment_image,
                   data_type=torch.float32):

    corr2 = process_common(corr2, data_type)['corr2']
    corr2 = corr2.unsqueeze(0)
    corr2 = corr2.to(device=device, dtype=data_type)

    start = time.time()  
    with torch.no_grad():
        image_pred = model(corr2)
        image_pred = image_pred.squeeze().cpu().numpy()
    print("重建图像耗时：", time.time()-start)
    show_while_predicting(corr2.cpu().squeeze().numpy(), image_pred, original_image, experiment_image, 1, 128, False)
    return image_pred, experiment_image


# 批量推理
def predict_batch( model,
                   device,
                   dataset_type,
                   experiment_read_dir,
                   label_read_dir,
                   experiment_save_dir,
                   corr_save_dir,
                   image_pred_save_dir,
                   label_save_dir,
                   cut_size, 
                   experiment_ext,
                   label_ext,
                   data_type=torch.float32):
    batch_size = 1
    experiment_name_list = glob.glob(experiment_read_dir + "*" + experiment_ext)

    DataSet = BasicDataset(dataset_type, experiment_name_list, label_read_dir, label_ext, cut_size, data_type)
    loader = DataLoader(DataSet,
                        shuffle=False,
                        batch_size=batch_size,
                        num_workers=0)
    n = len(DataSet)
    pcc, psnr, ssim, mse = 0, 0, 0, 0
    with tqdm(total=n, position=0) as pbar:
        for batch in loader:
            corr, label = batch["corr"], batch["label"]
            corr = corr.to(device=device, dtype=data_type)
            label = label.to(device=device, dtype=data_type)

            rebuild_start = time.time()
            with torch.no_grad():
                image_pred = model(corr)
                pcc += PCC_function(label, image_pred).item()
                psnr += PSNR_function(label, image_pred).item()
                ssim += SSIM_function(label, image_pred).item()
                mse += MSE_function(label, image_pred).item()
            rebuild_time = time.time() - rebuild_start

            save_start = time.time()
            # 保存图像
            # 实验散斑图案名称
            experiment_image_name = batch["experiment_image_name"][0]
            # 自相关图案名称
            corr_name = experiment_image_name
            # 重建图像名称
            image_pred_name = experiment_image_name
            # label名称
            label_name = experiment_image_name

            # 保存实验散斑
            paper_plot_direct(batch["experiment_image"].squeeze(), experiment_save_dir, experiment_image_name, '.png', 128, True)
            # 保存自相关
            paper_plot_direct(corr.cpu().squeeze().numpy(), corr_save_dir, corr_name, '.png', 128, True)
            # 保存重建图像
            paper_plot_direct(255-image_pred.squeeze().cpu().numpy(), image_pred_save_dir, image_pred_name, '.png', 128, True)
            # 保存label
            paper_plot_direct(label.squeeze().cpu().numpy(), label_save_dir, label_name, '.png', 128, True)
            save_time = time.time() - save_start

            pbar.update(batch_size)
            pbar.set_postfix(**{"rebuild_time:": rebuild_time, "save_time:": save_time})
            
    # 输出图像重建质量评价参数
    print("Average PCC: ", pcc/n)
    print("Average PSNR: ", psnr/n)
    print("Average SSIM: ", ssim/n)
    print("Average MSE: ", mse/n)





