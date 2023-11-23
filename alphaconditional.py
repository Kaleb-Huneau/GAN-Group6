from alphagan import AlphaGAN

class Alphaconditional(AlphaGAN):

    def __init__(self, opt):
        '''
        Keeps the same parameter options as regular AlphaGAN
        :param opt: Options
        '''
        self.opt = opt
        self.batch_size = 100
        self.noise_dim = 28*28
        self.epsilon = 1e-8
        self.alpha_d = float(opt.alpha_d)
        self.alpha_g = float(opt.alpha_g)
        self.seed = opt.seed
        self.loss_type = opt.loss_type
        self.dataset = opt.dataset
        self.n_epochs = opt.n_epochs
        self.gp = opt.gp
        self.scores = np.zeros(self.n_epochs)
        self.num_images = opt.num_images
        self.gp_coef = opt.gp_coef
        if self.dataset != 'cifar10':
            self.num_images = 10
        self.d_opt = Adam(2e-4, beta_1 = 0.5)
        self.g_opt = Adam(2e-4, beta_1 = 0.5)
        if self.dataset == 'cifar10':
            self.noise_dim = 100
        self.l1 = opt.l1
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        return
