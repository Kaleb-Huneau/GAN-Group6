from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from scipy.linalg import sqrtm
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
with tf.device('/gpu:1'): 
    #class fid_calculate()
    import numpy as np
    import os
    IMG_SIZE = 64

    def load_model(model_path):
        return tf.saved_model.load(model_path)

    def generate_noise(num_samples, noise_dim):
        return tf.random.normal([num_samples, noise_dim])

    def save_images(images, folder_path,add_to_folder):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        images_uint8 = tf.image.convert_image_dtype(images, tf.uint8, saturate=True)
        
        for i, img in enumerate(images_uint8):
            # Generate a new filename if the file already exists
            file_path = os.path.join(folder_path, f'generated_image_{i}.jpeg')
            if(add_to_folder == 1):
                while os.path.exists(file_path):
                    i += 1  # Increment the counter if file exists
                    file_path = os.path.join(folder_path, f'generated_image_{i}.jpeg')  # Update file path

            encoded_img = tf.image.encode_jpeg(img)
            tf.io.write_file(file_path, encoded_img)

    def generate_imgs(model_path, num_images, noise_dim):
        # Load the model
        loaded_model = load_model(model_path)
        print(loaded_model.signatures['serving_default'])
        generator_signature = loaded_model.signatures['serving_default']
        
        noise = generate_noise(num_images, noise_dim)
        if(IMG_SIZE == 64):
            generated_images = generator_signature(dense_2_input=noise)['conv2d_transpose_7']
        elif(IMG_SIZE == 128):
            generated_images = generator_signature(dense_2_input=noise)['conv2d_transpose_9']
        
        images = (generated_images.numpy() + 1.0) / 2.0
        
        return(images)

        # Save images
        # save_images(images, output_folder,add_to_folder)
        
    # def compute_fid(fake_images,real_images):
    #         #real_images = real_images.numpy()
    #         #ake_images = fake_images.numpy()
            
    #         print(fake_images.shape)
           
    #         #fake_images = fake_images.reshape(10000, IMG_SIZE*IMG_SIZE)
          
    #         # fake_mu = fake_images.mean(axis=0)
    #         # fake_sigma = np.cov(np.transpose(fake_images))
    #         # real_mu = real_images.mean(axis=0)
    #         # real_sigma = np.cov(np.transpose(real_images))
            
    #         num_fake_images = fake_images.shape[0]
    #         fake_images_flat = fake_images.reshape((num_fake_images, -1))  # Flatten each image
            
    #         num_real_images = real_images.shape[0]
    #         real_images_flat = real_images.reshape((num_real_images, -1))  # Flatten each image

    #         # Now you can compute the mean and covariance of the flattened images
    #         fake_mu = np.mean(fake_images_flat, axis=0)
    #         fake_sigma = np.cov(fake_images_flat, rowvar=False)  # rowvar=False: treat each column as a variable

    #         real_mu = np.mean(real_images_flat, axis=0)
    #         real_sigma = np.cov(real_images_flat, rowvar=False)
    #         print("1")
            
    #         covSqrt = sqrtm(np.matmul(fake_sigma, real_sigma))
    #         print("4")
    #         if np.iscomplexobj(covSqrt):
    #             print("2")
    #             covSqrt = covSqrt.real
    #         print("5")
    #         fidScore = np.linalg.norm(real_mu - fake_mu) + np.trace(real_sigma + fake_sigma - 2 * covSqrt)
    #         print("3")
    #         return fidScore

    def compute_fid(fake_images, real_images):
        # Check input shapes
        print("Fake images shape:", fake_images.shape)
        print("Real images shape:", real_images.shape)
        
        num_fake_images = fake_images.shape[0]
        fake_images_flat = fake_images.reshape((num_fake_images, -1))  # Flatten each image
            
        num_real_images = real_images.shape[0]
        real_images_flat = real_images.reshape((num_real_images, -1))  # Flatten each image

        # Now you can compute the mean and covariance of the flattened images
        fake_mu = np.mean(fake_images_flat, axis=0)
        fake_sigma = np.cov(fake_images_flat, rowvar=False)  # rowvar=False: treat each column as a variable

        real_mu = np.mean(real_images_flat, axis=0)
        real_sigma = np.cov(real_images_flat, rowvar=False)
        
        print("Fake mu shape:", fake_mu.shape)
        print("Real mu shape:", real_mu.shape)
        print("Fake sigma shape:", fake_sigma.shape)
        print("Real sigma shape:", real_sigma.shape)
        
        # one that is working
        covSqrt = sqrtm(np.matmul(fake_sigma, real_sigma))
        print("CovSqrt shape:", covSqrt.shape)
        # Compute eigenvalue decomposition
        # fake_sigma[fake_sigma < 0] = 0
        # real_sigma[real_sigma < 0] = 0
        # eigenvalues, eigenvectors = np.linalg.eigh(np.matmul(fake_sigma, real_sigma))
        # sqrt_eigenvalues = np.sqrt(eigenvalues)
        # covSqrt = np.dot(np.dot(eigenvectors, np.diag(sqrt_eigenvalues)), eigenvectors.T)

        if np.iscomplexobj(covSqrt):
            print("CovSqrt is complex!")
            covSqrt = covSqrt.real

        fidScore = np.linalg.norm(real_mu - fake_mu) + np.trace(real_sigma + fake_sigma - 2 * covSqrt)
        return fidScore


    def load_real_images(dir):
        tumor_training_images = []
        for filename in os.listdir(dir):
            if filename.endswith('.jpg'):  # Assuming images are in JPG format
                image_path = os.path.join(dir, filename)
                image = plt.imread(image_path, format='jpg')
                tumor_training_images.append(image)

        # Convert the list of images to a NumPy array
        return np.asarray(tumor_training_images)
        
    #v38 is 128x128 images

    real_images_path = '/data/user1/AlphaGANMRI/cleaned64by64/Training/notumor'
    model_path = '/data/user1/AlphaGANMRI/v4/AlphaGAN_first/mri/alpha-d2.0-g2.0/v11/models/generator250'
    num_images = 600  #  number of images to generate
    noise_dim = IMG_SIZE*IMG_SIZE  # dimension of the noise vector
    output_folder = '/data/user1/AlphaGANMRI/v3/generated_imgs/128x128testOnlyTumors'  # Specify the folder to save to


    fake_images = generate_imgs(model_path, num_images, noise_dim)
    real_images = load_real_images(real_images_path)
    # expand the dim to (4117, 128, 128, 1) from (4117, 128, 128)
    real_images = np.expand_dims(real_images, axis=-1)
    real_images = real_images/255
    real_images = real_images[:800]

    print(np.min(real_images))
    print(np.max(real_images))
    print(np.min(fake_images))
    print(np.max(fake_images))
    #fake_images = real_images
    #real_images = fake_images
    
    
    fid_score = compute_fid(fake_images,real_images)
    
    print(fid_score)
    