import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, BatchNormalization, \
    LeakyReLU, Conv2DTranspose, Conv2D, Dropout, \
        Flatten, Reshape
from keras.initializers import RandomNormal
import argparse
from tensorflow.keras.utils import plot_model

with(tf.device("cpu")):

    parser = argparse.ArgumentParser(description="Run GAN model with specific alpha values")
    parser.add_argument("--alpha_d", type=float, required=True, help="alpha_d value")
    parser.add_argument("--alpha_g", type=float, required=True, help="alpha_g value")
    parser.add_argument("--type", type=str, required=True, help="tumor or no tumor")
    args = parser.parse_args()

    IMG_SIZE = 128  # Assuming image size is 128x128
    noise_dim = IMG_SIZE * IMG_SIZE  # Assuming noise dimension matches image size

    def build_generator():
        model = Sequential()
        model.add(Dense(8 * 8 * 256, use_bias=False, kernel_initializer=
        RandomNormal(mean=0.0, stddev=0.01), input_shape=(noise_dim,)))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Reshape((8, 8, 256)))
        # Add more layers here...
        model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='tanh', use_bias=False,
                                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01)))
        return model

    def build_discriminator():
        model = Sequential()
        # Add layers here...
        model.add(Conv2D(64, (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
        model.add(LeakyReLU(0.2))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        return model

    # Build and compile the generator
    generator = build_generator()
    discriminator = build_discriminator()

    # Print model summaries
    print("\nGenerator Summary:")
    generator.summary()
    plot_model(generator, to_file='/data/user1/AlphaGANMRI/v3/model_simple_gen.png', show_shapes=False, show_layer_names=False)
    print("\nDiscriminator Summary:")
    discriminator.summary()
    plot_model(discriminator, to_file='/data/user1/AlphaGANMRI/v3/model_simple_disc.png', show_shapes=False, show_layer_names=False)

