import os
import tensorflow as tf
from scipy.linalg import sqrtm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

IMG_SIZE = 128
BATCH_SIZE = 32  # Adjust based on your GPU memory

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use GPU 0

def load_model(model_path):
    return tf.saved_model.load(model_path)

def generate_noise(num_samples, noise_dim):
    return tf.random.normal([num_samples, noise_dim])

def load_and_preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = (image / 127.5) - 1  # Normalize images to [-1, 1]
    return image

def load_images_in_batches(directory):
    image_paths = [os.path.join(directory, fname) for fname in os.listdir(directory) if fname.endswith('.jpg')]
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    image_ds = image_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return image_ds

def generate_images(model_path, num_images, noise_dim):
    loaded_model = load_model(model_path)
    generator_signature = loaded_model.signatures['serving_default']
    
    # Assuming the model expects a noise vector for each image to generate
    batches = num_images // BATCH_SIZE
    images = []
    for _ in range(batches):
        noise = generate_noise(BATCH_SIZE, noise_dim)
        if(IMG_SIZE == 64):
            generated_images = generator_signature(dense_2_input=noise)['conv2d_transpose_7']
        elif(IMG_SIZE == 128):
            generated_images = generator_signature(dense_2_input=noise)['conv2d_transpose_9']
        images.append(generated_images)
    
    # If num_images is not a multiple of BATCH_SIZE, handle the last batch separately
    if num_images % BATCH_SIZE > 0:
        noise = generate_noise(num_images % BATCH_SIZE, noise_dim)
        if(IMG_SIZE == 64):
            generated_images = generator_signature(dense_2_input=noise)['conv2d_transpose_7']
        elif(IMG_SIZE == 128):
            generated_images = generator_signature(dense_2_input=noise)['conv2d_transpose_9']
        images.append(generated_images)
    
    images = tf.concat(images, axis=0)
    images = (images.numpy() + 1.0) / 2.0  # Denormalize images
    return images

def compute_fid(fake_images, real_images):
    fake_images_flat = fake_images.reshape(fake_images.shape[0], -1)
    real_images_flat = real_images.reshape(real_images.shape[0], -1)
    fake_images_flat = real_images_flat
    
    fake_mu = np.mean(fake_images_flat, axis=0)
    fake_sigma = np.cov(fake_images_flat, rowvar=False)
    
    real_mu = np.mean(real_images_flat, axis=0)
    real_sigma = np.cov(real_images_flat, rowvar=False)
    
    covSqrt = sqrtm(fake_sigma.dot(real_sigma))
    if np.iscomplexobj(covSqrt):
        covSqrt = covSqrt.real
    
    fidScore = np.sum((fake_mu - real_mu)**2) + np.trace(fake_sigma + real_sigma - 2*covSqrt)
    return fidScore

if __name__ == "__main__":
    real_images_dir = '/home/user4/Desktop/AlphaGANMRI/cleaned128by128/Training/onlytumors'
    model_path = '/home/user4/Desktop/AlphaGANMRI/AlphaGAN/mri/alpha-d1.0-g5.0/onlytumors_128x128_alpha_d=1_alpha_g=5_500_epoch/models/generator250/'
    num_images = 10000  # Make sure this is a multiple of BATCH_SIZE for simplicity
    noise_dim = IMG_SIZE * IMG_SIZE

    real_image_ds = load_images_in_batches(real_images_dir)
    real_images = np.vstack([images.numpy() for images in real_image_ds])

    fake_images = generate_images(model_path, num_images, noise_dim)
    fid_score = compute_fid(fake_images, real_images)
    print("FID score:", fid_score)
