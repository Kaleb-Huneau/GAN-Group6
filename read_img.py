import numpy as np
import matplotlib.pyplot as plt

img = np.load("/Users/kalebhuneau/Desktop/MTHE 493/GAN-Group6/AlphaGAN/mnist/alpha-d3.0-g3.0/v2/img/predictions1.npy")

#print(img)

plt.imshow(img[6])
plt.show()