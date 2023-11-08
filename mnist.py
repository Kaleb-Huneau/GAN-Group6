#from keras.datasets import mnist
import alphagan
import gc
import argparse

class Option():
    """
    Empty class to hold the gan options
    """
    def __init__(self, gan_type, alpha, seed, c_type, n_epochs, dataset, loss_type, lambda_d, lambda_c, num_images, gp, gen_lr, dis_lr, q_lr, gp_coef, alpha_d, alpha_g, k, shifted, l1):
        self.gan_type = gan_type
        self.alpha = alpha
        self.seed = seed
        self.c_type = c_type
        self.n_epochs = n_epochs
        self.dataset = dataset
        self.loss_type = loss_type
        self.lambda_d = lambda_d
        self.lambda_c = lambda_c
        self.num_images = num_images
        self.gp = gp
        self.gen_lr = gen_lr
        self.dis_lr = dis_lr
        self.q_lr = q_lr
        self.gp_coef = gp_coef
        self.alpha_d = alpha_d
        self.alpha_g = alpha_g
        self.k = k
        self.shifted = shifted
        self.l1 = l1
        return
#set up options
opts = Option(gan_type='alpha', alpha=3.0, seed=42, c_type='discrete', n_epochs=10, dataset='mnist', loss_type='vanilla', lambda_d=1.0, lambda_c=0.1, num_images=100, gp=False, gen_lr=0.0002, dis_lr=0.0002, q_lr=0.0002, gp_coef=5.0, alpha_d=3.0, alpha_g=3.0, k=2.0, shifted=False, l1=False)

# Define an alphagan to test
gan = alphagan.AlphaGAN(opts)

# sets data to mnist and configures it for the gan
gan.dataset = 'mnist'
gan.get_data()


#build the generative network
gan.build_gan()

gan.train()

gan.save_generated_images(1)