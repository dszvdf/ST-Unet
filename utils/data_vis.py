import matplotlib.pyplot as plt

def show_while_predicting(corr2s, reconstructions, images, expris, show_nums, output_size, batch=False):
    plt.rcParams['figure.figsize'] = (16, 16) 
    plt.rcParams['savefig.dpi'] = 100
    cmap = plt.cm.gray#inferno
    if not batch:
        fig = plt.figure(num=1)
        ax1 = fig.add_subplot(221)
        ax1.imshow(expris.reshape((output_size, output_size)), cmap=cmap, interpolation='nearest')
        ax1.set_title("expris")
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        
        ax2 = fig.add_subplot(222)
        ax2.imshow(corr2s.reshape((output_size, output_size)), cmap=cmap, interpolation='nearest')
        ax2.set_title("corr2s")
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        
        ax3 = fig.add_subplot(223)
        ax3.imshow(reconstructions.reshape((output_size, output_size)), cmap=cmap, interpolation='nearest')
        ax3.set_title("reconstructions")
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)
        
        ax4 = fig.add_subplot(224)
        ax4.imshow(images.reshape((output_size, output_size)), cmap=cmap, interpolation='nearest')
        ax4.set_title("images")
        ax4.get_xaxis().set_visible(False)
        ax4.get_yaxis().set_visible(False)

    else:
        _, figs1 = plt.subplots(1, show_nums, figsize=(12, 12)) 
        for f, img in zip(figs1, corr2s):
            f.imshow(img.reshape((output_size, output_size)))
            f.axes.get_xaxis().set_visible(False)
            f.axes.get_yaxis().set_visible(False)

        _, figs2 = plt.subplots(1, show_nums, figsize=(12, 12)) 
        for f, img in zip(figs2, reconstructions):
            f.imshow(img.reshape((output_size, output_size)))
            f.axes.get_xaxis().set_visible(False)
            f.axes.get_yaxis().set_visible(False)

        _, figs3 = plt.subplots(1, show_nums, figsize=(12, 12)) 
        for f, img in zip(figs3, images):
            f.imshow(img.reshape((output_size, output_size)))
            f.axes.get_xaxis().set_visible(False)
            f.axes.get_yaxis().set_visible(False)

        _, figs4 = plt.subplots(1, show_nums, figsize=(12, 12)) 
        for f, img in zip(figs4, expris):
            f.imshow(img.reshape((output_size, output_size)))
            f.axes.get_xaxis().set_visible(False)
            f.axes.get_yaxis().set_visible(False)

    plt.show()



def show_while_training(imgs, reconstructions, show_nums, output_size):
    _, figs1 = plt.subplots(1, show_nums, figsize=(12, 12)) 
    for f, img in zip(figs1, imgs):
        f.imshow(img.reshape((output_size, output_size)))
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)

    _, figs2 = plt.subplots(1, show_nums, figsize=(12, 12)) 
    for f, img in zip(figs2, reconstructions):
        f.imshow(img.reshape((output_size, output_size)))
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)

    plt.show()
