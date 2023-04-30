import tensorflow as tf
from tensorflow.keras import layers
import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
# Define hyperparameters
latent_dim = 100
image_shape = (128, 128, 3)
batch_size = 32
epochs = 10000
data_path = './ships-aerial-images/images/'

# Load images into memory
image_files = glob.glob(data_path + "*")
num_batches = len(image_files) // batch_size
num_images = len(image_files)
print("Number of images found: ", num_images)

images = []
for i, file_path in enumerate(image_files):
    print(f"Loading image {i+1}/{num_images}", end="\r")
    image = cv2.imread(file_path)
    if len(image.shape) == 2:
        image_file = cv2.cvtColor(image_file, cv2.COLOR_GRAY2BGR)
    image = cv2.resize(image, (image_shape[0], image_shape[1]))
    image = image / 255.0  # Normalize
    images.append(image)
images = np.array(images)

# Create generator model
generator = tf.keras.Sequential()
generator.add(layers.Dense(32 * 32 * 256, input_shape=(latent_dim,)))
generator.add(layers.Reshape((32, 32, 256)))
generator.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same'))
generator.add(layers.BatchNormalization())
generator.add(layers.LeakyReLU(alpha=0.2))
generator.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'))
generator.add(layers.BatchNormalization())
generator.add(layers.LeakyReLU(alpha=0.2))
generator.add(layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', activation='tanh'))

# Create discriminator model
discriminator = tf.keras.Sequential()
discriminator.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=(128, 128, 3)))
discriminator.add(layers.LeakyReLU(alpha=0.2))
discriminator.add(layers.Dropout(0.3))
discriminator.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
discriminator.add(layers.LeakyReLU(alpha=0.2))
discriminator.add(layers.Dropout(0.3))
discriminator.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
discriminator.add(layers.LeakyReLU(alpha=0.2))
discriminator.add(layers.Dropout(0.3))
discriminator.add(layers.Flatten())
discriminator.add(layers.Dense(1, activation='sigmoid'))

# Compile discriminator
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer, metrics=['accuracy'])
discriminator.trainable = False
discriminator.verbose = 0

# Combine generator and discriminator into a GAN model
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
gan = tf.keras.Sequential([generator, discriminator])
gan.compile(loss='binary_crossentropy', optimizer=generator_optimizer)
generator.verbose = 0

# Train model
for epoch in range(epochs):
    # Sample random noise
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    
    # Train num_batches
    for i in tqdm(range(num_batches)):
        print(f"Batch {i+1}/{num_batches}", end="\r")

        # Get real and fake images
        real_images = images[i * batch_size: (i + 1) * batch_size]
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_images = generator.predict(noise)
    
        # Train discriminator
        d_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        y = np.ones((batch_size, 1))
        gan.verbose = 0
        generator_loss = gan.train_on_batch(noise, y)
    
    # Print progress
    print(f"Epoch {epoch + 1}/{epochs}")
    print("\nDiscriminator Loss:", d_loss)
    print("\nGenerator Loss:", generator_loss)

    # Save generated images
    if (epoch + 1) % 50 == 0:
        noise = np.random.normal(0, 1, (5, latent_dim))
        generated_images = generator.predict(noise)
        generated_images = (generated_images + 1) / 2
        #generated_images = np.array(generated_images*255).astype('uint8')
        for i in range(5):
            cv2.imwrite(f"./generated/generated_images_{epoch + 1}_{i + 1}.jpg", generated_images[i] * 255.0)
