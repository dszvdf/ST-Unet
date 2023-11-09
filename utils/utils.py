import numpy as np
import cv2
from skimage.transform import rescale
import torch

def padding(image, N):
    pad_width = int(N / 2)
    image = np.pad(image, ((pad_width), (pad_width)), 'constant', constant_values=0.0)
    height, width = image.shape
    hl, hr, wu, wd = int((height-N)/2), int((height+N)/2), int((width - N)/2), int((width + N)/2)
    return image[hl: hr, wu: wd]

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

def fspecial(shape=(4, 4), sigma=0.5):
    return np.multiply(cv2.getGaussianKernel(shape[0], sigma), cv2.getGaussianKernel(shape[1], sigma).T)

def imfilter(image, filter):
    return cv2.filter2D(image.astype('float32'), -1, filter, borderType=cv2.BORDER_CONSTANT)

def filter(image, filter_len):
    image = np.double(image)
    h, w = image.shape

    freq = np.fft.fftshift(np.fft.fft2(image))

    T = np.zeros((h, w))
    ceil_h, ceil_w = int(np.ceil(h/2)), int(np.ceil(w/2))
    T[ceil_h-filter_len: ceil_h+filter_len, ceil_w-filter_len: ceil_w+filter_len] = 1

    freq = freq * T

    freq = abs(np.fft.ifft2(np.fft.ifftshift(freq)))
    freq_new = image / freq

    filter = fspecial([4, 4], 0.5)
    after_filter_image = imfilter(freq_new, filter)

    return after_filter_image

def PSF(kernel_size):
    sigma = 0.003
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                        np.linspace(-1, 1, kernel_size))
    dst = x**2+y**2
    gauss = np.exp(-(dst / (2.0 * sigma**2)))
    ret = gauss/gauss.max()
    return ret

def process_common(corr, data_type=torch.float32):
    if len(corr.shape) == 2:
        corr = np.expand_dims(corr, axis=2)
    # 自相关和原图的预处理
    corr = corr / np.max(corr)
    corr = (corr - corr.min()) / (corr.max() - corr.min())
    corr = corr.transpose((2, 0, 1))
    batch  = {"corr2": torch.from_numpy(corr).type(data_type)}
    return batch


def preprocess_mnist(original_image, experiment_image, Size):
    h, w = experiment_image.shape
    experiment_image = experiment_image[int((h-512)/2): int((h+512)/2), int((w-512)/2): int((w+512)/2)]

    corr = correlation(experiment_image)
    # corr = correlation(corr, PSF(512))

    h, w = corr.shape
    corr = corr[int((h-Size)/2): int((h+Size)/2), int((w-Size)/2): int((w+Size)/2)]
    corr = rescale(corr, scale=(128/Size), mode='constant')
    corr = 255 * corr

    original_image = rescale(255-original_image, scale=(128/28), mode='constant')

    # h, w = experiment_image.shape
    # experiment_image = experiment_image[int((h-1024)/2): int((h+1024)/2), int((w-1024)/2): int((w+1024)/2)]
    experiment_image = rescale(experiment_image, scale=(128/512), mode='constant')
    return [original_image, corr, experiment_image]


def preprocess_mnist_combination(original_image, experiment_image, Size):
    corr2 = correlation(experiment_image)

    h, w = corr2.shape
    corr2 = corr2[int((h-Size)/2): int((h+Size)/2), int((w-Size)/2): int((w+Size)/2)]
    corr2 = rescale(corr2, scale=(128/Size), mode='constant')

    h, w = original_image.shape
    original_image = original_image[int((h-150)/2): int((h+150)/2), int((w-150)/2): int((w+150)/2)]
    original_image = rescale(original_image, scale=(128/max(original_image.shape)), mode='constant')

    h, w = experiment_image.shape
    experiment_image = experiment_image[int((h-1024)/2): int((h+1024)/2), int((w-1024)/2): int((w+1024)/2)]
    experiment_image = rescale(experiment_image, scale=(128/1024), mode='constant')
    return [original_image, corr2, experiment_image]


def preprocess_letters(original_image, experiment_image, Size):
    corr2 = correlation(experiment_image)

    h, w = corr2.shape
    corr2 = corr2[int((h-Size)/2): int((h+Size)/2), int((w-Size)/2): int((w+Size)/2)]
    corr2 = rescale(corr2, scale=(128/Size), mode='constant')

    h, w = original_image.shape
    original_image = original_image[int((h-70)/2): int((h+70)/2), int((w-70)/2): int((w+70)/2)]
    original_image = rescale(original_image, scale=(128/max(original_image.shape)), mode='constant')

    h, w = experiment_image.shape
    experiment_image = experiment_image[int((h-1024)/2): int((h+1024)/2), int((w-1024)/2): int((w+1024)/2)]
    experiment_image = rescale(experiment_image, scale=(128/1024), mode='constant')
    return [original_image, corr2, experiment_image]