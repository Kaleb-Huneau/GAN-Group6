#from keras.datasets import mnist
import alphagan

# Define an alphagan to test
gan = alphagan.AlphaGAN()

# sets data to mnist and configures it for the gan
gan.dataset = 'mnist'
gan.get_data()


#build the generative network
gan.build_gan()

gan.train()