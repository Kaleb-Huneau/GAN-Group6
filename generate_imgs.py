import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
with tf.device('/gpu:0'): 
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

    def main(model_path, num_images, noise_dim, output_folder, add_to_folder):
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

        # Save images
        save_images(images, output_folder,add_to_folder)
        
    #v38 is 128x128 images

    if __name__ == "__main__":
        model_path = '/data/user1/AlphaGANMRI/v4/AlphaGAN_alternatives/mri/alpha-d0.1-g0.5/v2/models/generator100'
        num_images = 100  #  number of images to generate
        noise_dim = IMG_SIZE*IMG_SIZE  # dimension of the noise vector
        output_folder = '/data/user1/AlphaGANMRI/v3/Alternative_tests'
        #output_folder = '/home/user4/Desktop/AlphaGANMRI/AlphaGAN/cleaned128by128augmented2/Training/glioma'  # Specify the folder to save to
        add_to_folder = 1 #specifies whether or not you want to add the photos to the folder and keep the old ones or replace the ones that already exist

        main(model_path, num_images, noise_dim, output_folder,add_to_folder)
