import matplotlib.pyplot as plt
import skimage.io as io
import glob
import os
from skimage.transform import rescale


def paper_plot(read_dir, save_dir, read_ext, save_ext, output_size, ifr=False):
    if ifr:
        cmap = plt.cm.inferno_r
    else:
        cmap = plt.cm.inferno

    image_namelist = glob.glob(read_dir + "*" + read_ext)

    for image_name in image_namelist:
        image = io.imread(image_name)
        h, w = image.shape
        image = rescale(255-image, scale=(output_size/max(h, w)), mode='constant')
        plt.rcParams['figure.figsize'] = (10.24, 10.24) 
        plt.rcParams['savefig.dpi'] = 100
        plt.axis('off')
        plt.margins(0, 0)
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.imshow(image, 
                   cmap=cmap,
                   interpolation='nearest')
        image_name = image_name.split(os.sep)[-1].split(".")[0]
        plt.savefig(save_dir + image_name + save_ext)


def paper_plot_direct(image, save_dir, save_name, save_ext, output_size, ifr=False):
    if ifr:
        cmap = plt.cm.inferno_r
    else:
        cmap = plt.cm.inferno
    h, w = image.shape
    image = rescale(255-image, scale=(output_size/max(h, w)), mode='constant')
    plt.rcParams['figure.figsize'] = (10.24, 10.24) 
    plt.rcParams['savefig.dpi'] = 100
    plt.axis('off')
    plt.margins(0, 0)
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.imshow(image, 
               cmap=cmap,
               interpolation='nearest')

    plt.savefig(save_dir + save_name + save_ext)
    plt.clf()