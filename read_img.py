import numpy as np
import matplotlib.pyplot as plt

#choose how many plots (1) = 1, 4 = 4
num_plots = 1


imgs = np.load("/Users/kalebhuneau/Desktop/MTHE 493/GAN-Group6/AlphaGAN/mnist/alpha-d3.0-g3.0/v2/img/predictions1.npy")


if num_plots == 1:
    # one image
    print(np.shape(imgs))
    plt.imshow(imgs[60], cmap = 'binary')
    plt.show()

elif num_plots == 4:
    # 4 x 4 grid of images
    fig, axs = plt.subplots(2,2)
    axs[0][0].imshow(imgs[4], cmap = 'binary')
    #axs[0][0]("first")
    axs[0][1].imshow(imgs[5], cmap='binary')
    axs[1][0].imshow(imgs[6], cmap='binary')
    axs[1][1].imshow(imgs[7], cmap='binary')
    plt.show()