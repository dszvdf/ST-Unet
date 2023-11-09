from skimage import io
import numpy as np
import torch
import os
from torch.utils.data import Dataset
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.color import rgb2gray
from skimage.transform import rescale

class BasicDataset(Dataset):
    def __init__(self, 
                 dataset_type, 
                 experiment_name_list, 
                 label_read_dir,
                 label_ext, 
                 cut_size, 
                 data_type=torch.float32):
        # 数据集类型
        self.dataset_type = dataset_type
        # 实验散斑图案路径列表
        self.experiment_name_list = experiment_name_list
        # label读取目录
        self.label_read_dir = label_read_dir
        # label文件后缀
        self.label_ext = label_ext
        # 自相关截取大小
        self.cut_size = cut_size
        # 计算数据精度
        self.data_type = data_type

    def __len__(self):
        return len(self.experiment_name_list)
    @staticmethod
    def padding(image, N):
        pad_width = int(N / 2)
        image = np.pad(image, ((pad_width), (pad_width)), 'constant', constant_values=0.0)
        height, width = image.shape
        hl, hr, wu, wd = int((height-N)/2), int((height+N)/2), int((width - N)/2), int((width + N)/2)
        return image[hl: hr, wu: wd]
    @staticmethod
    def correlation(image1, image2=None):
        if (image2 is None):
            Freq = abs(np.fft.fft2(np.fft.fftshift(image1))) ** 2
            corr = abs(np.fft.ifftshift(np.fft.ifft2(Freq)))
            corr = corr - corr.min()
            return corr
        else:
            Freq_1 = abs(np.fft.fft2(np.fft.fftshift(image1)))
            Freq_2 = abs(np.fft.fft2(np.fft.fftshift(image2)))
            Freq = Freq_1 * Freq_2
            corr = abs(np.fft.ifftshift(np.fft.ifft2(Freq)))
            corr = corr - corr.min()
            return corr

    def process_common(self, corr, label, experiment_image, experiment_image_name, data_type=torch.float32):
        if len(corr.shape) == 2:
            corr = np.expand_dims(corr, axis=2)
        if len(label.shape) == 2:
            label = np.expand_dims(label, axis=2)
        # 自相关和原图的预处理
        corr = corr / np.max(corr)
        if (self.dataset_type != "letters"):
            label = label / np.max(label)
        corr = (corr - corr.min()) / (corr.max() - corr.min())
        corr = corr.transpose((2, 0, 1))
        label = label.transpose((2, 0, 1))
        batch  = {"corr": torch.from_numpy(corr).type(data_type),
                  "label": torch.from_numpy(label).type(data_type),
                  "experiment_image": experiment_image,
                  "experiment_image_name": experiment_image_name.split(os.sep)[-1].split(".")[0]}
        return batch
    
    def process_mnist(self, experiment_image_name):
        # 读取散斑图案
        experiment_image = io.imread(experiment_image_name)

        # 计算散斑自相关
        corr = self.correlation(experiment_image)
        h, w = corr.shape
        corr = corr[int((h-self.cut_size)/2): int((h+self.cut_size)/2), int((w-self.cut_size)/2): int((w+self.cut_size)/2)]
        corr = rescale(corr, scale=(128/self.cut_size), mode='constant')

        # label
        # original_image_name = "TestImage_" + experiment_image_name.split(os.sep)[-1].split(".")[0] + self.label_ext
        original_image_name = experiment_image_name.split(os.sep)[-1].split(".")[0] + self.label_ext
        original_image = io.imread(self.label_read_dir + original_image_name)
        original_image = rescale(255-original_image, scale=(128//max(original_image.shape)), mode='constant')
        original_image = self.padding(original_image, 128)

        # 散斑图案
        h, w = experiment_image.shape
        experiment_image = experiment_image[int((h-1024)/2): int((h+1024)/2), int((w-1024)/2): int((w+1024)/2)]
        experiment_image = rescale(experiment_image, scale=(128/1024), mode='constant')

        batch = self.process_common(corr, original_image, experiment_image, experiment_image_name, self.data_type)
        return batch
    

    def process_mnist_combination(self, experiment_image_name):
        # 读取散斑图案
        experiment_image = io.imread(experiment_image_name)

        # 计算散斑自相关
        corr = self.correlation(experiment_image)
        h, w = corr.shape
        corr = corr[int((h-self.cut_size)/2): int((h+self.cut_size)/2), int((w-self.cut_size)/2): int((w+self.cut_size)/2)]
        corr = rescale(corr, scale=(128/self.cut_size), mode='constant')

        # label
        # original_image_name = "TestImage_" + experiment_image_name.split(os.sep)[-1].split(".")[0] + self.label_ext
        original_image_name = experiment_image_name.split(os.sep)[-1].split(".")[0] + self.label_ext
        original_image = io.imread(self.label_read_dir + original_image_name)
        h, w = original_image.shape
        original_image = original_image[int((h-150)/2): int((h+150)/2), int((w-150)/2): int((w+150)/2)]
        original_image = rescale(original_image, scale=(128/max(original_image.shape)), mode='constant')

        # 散斑图案
        h, w = experiment_image.shape
        experiment_image = experiment_image[int((h-1024)/2): int((h+1024)/2), int((w-1024)/2): int((w+1024)/2)]
        experiment_image = rescale(experiment_image, scale=(128/1024), mode='constant')
        
        batch = self.process_common(corr, original_image, experiment_image, experiment_image_name, self.data_type)
        return batch
    
    def process_letters(self, experiment_image_name):
        # 读取散斑图案
        experiment_image = io.imread(experiment_image_name)
        h, w = experiment_image.shape
        experiment_image = experiment_image[int((h-512)/2): int((h+512)/2), int((w-512)/2): int((w+512)/2)]
        # 计算散斑自相关
        corr = self.correlation(experiment_image)
        h, w = corr.shape
        corr = corr[int((h-self.cut_size)/2): int((h+self.cut_size)/2), int((w-self.cut_size)/2): int((w+self.cut_size)/2)]
        corr = rescale(corr, scale=(128/self.cut_size), order=2, mode='reflect')

        # label
        original_image_name = experiment_image_name.split(os.sep)[-1].split(".")[0] + self.label_ext
        original_image = io.imread(self.label_read_dir + original_image_name)
        h, w = original_image.shape
        original_image = original_image[int((h-70)/2): int((h+70)/2), int((w-70)/2): int((w+70)/2)]
        original_image = rescale(original_image, scale=(128/max(original_image.shape)), order=1, mode='constant')

        # 散斑图案
        h, w = experiment_image.shape
        experiment_image = experiment_image[int((h-1024)/2): int((h+1024)/2), int((w-1024)/2): int((w+1024)/2)]
        experiment_image = rescale(experiment_image, scale=(128/1024), order=2, mode='reflect')

        batch = self.process_common(corr, original_image, experiment_image, experiment_image_name, self.data_type)
        return batch


    def __getitem__(self, index):
        experiment_image_name = self.experiment_name_list[index]
        if (self.dataset_type == "mnist"):
            batch = self.process_mnist(experiment_image_name)
        elif (self.dataset_type == "mnist_combination"):
            batch = self.process_mnist_combination(experiment_image_name)
        elif (self.dataset_type == "letters"):
            batch = self.process_letters(experiment_image_name)
        elif (self.dataset_type == "rotation"): 
            batch = self.process_letters(experiment_image_name)
        else:
            pass
        return batch