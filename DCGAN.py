
import os
import numpy as np
import cv2
from glob import glob
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
w_init = tf.keras.initializers.GlorotNormal()

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_png(img)
    img = tf.image.resize_with_crop_or_pad(img, IMG_H, IMG_W)
    img = tf.cast(img, tf.float32)
    img = (img - 127.5) / 127.5
    return img

def tf_dataset(images_path, batch_size):
    #print(images_path)
    dataset = tf.data.Dataset.from_tensor_slices(images_path)
    dataset = dataset.shuffle(buffer_size=10240)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def deconv_block(inputs, num_filters, kernel_size, strides, bn=True):
    x = layers.Conv2DTranspose(
        filters=num_filters,
        kernel_size=kernel_size,
        kernel_initializer=w_init,
        padding="same",
        strides=strides,
        use_bias=False
        )(inputs)

    if bn:
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
    return x


def conv_block(inputs, num_filters, kernel_size, padding="same", strides=2, activation=True):
    x = layers.Conv2D(
        filters=num_filters,
        kernel_size=kernel_size,
        kernel_initializer=w_init,
        padding=padding,
        strides=strides,
    )(inputs)

    if activation:
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.3)(x)
    return x

def build_generator(latent_dim):
    f = [2**i for i in range(5)][::-1]
    print(f)
    filters = 32
    output_strides = 16
    h_output = IMG_H // output_strides
    w_output = IMG_W // output_strides

    noise = layers.Input(shape=(latent_dim,), name="generator_noise_input")

    x = layers.Dense(f[0] * filters * h_output * w_output, use_bias=False)(noise)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Reshape((h_output, w_output, 16 * filters))(x)

    for i in range(1, 5):
        x = deconv_block(x,
            num_filters=f[i] * filters,
            kernel_size=5,
            strides=2,
            bn=True
        )
        print(f[i] * filters)
    
    #raise
    x = conv_block(x,
        num_filters=IMG_C,
        kernel_size=5,
        strides=1,
        activation=False
    )
    fake_output = layers.Activation("tanh")(x)

    return Model(noise, fake_output, name="generator")

def build_discriminator():
    f = [2**i for i in range(4)]
    image_input = layers.Input(shape=(IMG_H, IMG_W, IMG_C))
    x = image_input
    filters = 64
    output_strides = 16
    h_output = IMG_H // output_strides
    w_output = IMG_W // output_strides

    for i in range(0, 4):
        x = conv_block(x, num_filters=f[i] * filters, kernel_size=5, strides=2)

    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)

    return Model(image_input, x, name="discriminator")

def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)

def gradient_penalty(samples, output):
    gradients = tf.gradients(output, samples)
    gradients_sqr = tf.square(gradients)
    gradients_sqr_sum = tf.reduce_sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = tf.sqrt(gradients_sqr_sum)
    gradient_penalty = tf.reduce_mean(tf.square(gradient_l2_norm - 1))
    return gradient_penalty

class GAN(Model):
    def __init__(self, discriminator, generator, latent_dim, n_critic=1):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.n_critic = n_critic
        #self.epoch_loss = {"d_loss": [], "g_loss": []}
        
    def compile(self, d_optimizer, g_optimizer):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
    
    @tf.function
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        for i in range(self.n_critic):
            ## Train the discriminator
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            generated_images = self.generator(random_latent_vectors)

            with tf.GradientTape() as ftape, tf.GradientTape() as rtape:
                real_predictions = self.discriminator(real_images)
                fake_predictions = self.discriminator(generated_images)
                d1_loss = tf.reduce_mean(fake_predictions) - tf.reduce_mean(real_predictions)
                alpha = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
                interpolated_images = real_images * alpha + generated_images * (1 - alpha)
                interpolated_predictions = self.discriminator(interpolated_images)
                gradients = tf.gradients(interpolated_predictions, interpolated_images)[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
                gradient_penalty = tf.reduce_mean(tf.square(slopes - 1))
                d_loss = d1_loss + 10 * gradient_penalty

            d_grads = ftape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))

        ## Train the generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as gtape:
            generated_images = self.generator(random_latent_vectors)
            fake_predictions = self.discriminator(generated_images)
            g_loss = -tf.reduce_mean(fake_predictions)

        g_grads = gtape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_weights))


        return {"d_loss": d_loss, "g_loss": g_loss}

def save_plot(examples, epoch, n):
    examples = (examples + 1) / 2.0
    for i in range(n * n):
        plt.subplot(n, n, i+1)
        plt.axis("off")
        plt.imshow(examples[i], cmap="gray")
    filename = f"./grid_search/{test_id}/samples/generated_plot_epoch-{epoch+1}.png"
    plt.savefig(filename, )
    plt.close()

if __name__ == "__main__":
    ## Hyperparameters
    img_size = 256
    IMG_H = img_size
    IMG_W = img_size
    IMG_C = 1
    
    latent_dim = img_size
    
    batch_size = 10
    num_epochs = 500
    
    generator_learning_rate = 0.0001
    #generator_decay = 0.00001
    generator_decay = None
    generator_beta_1 = 0.5
    generator_amsgrad = True
    
    test_id = 2
    
    #csv_logger = tf.keras.callbacks.CSVLogger(f"./grid_search/{test_id}/training.log")
    os.makedirs(f"./grid_search/{test_id}/")
    os.makedirs(f"./grid_search/{test_id}/samples/")
    os.makedirs(f"./grid_search/{test_id}/saved_model/")
    h_data = str(
        f"""
        img_size = {img_size}
        latent_dim = {latent_dim}\n
        batch_size = {batch_size}\n
        num_epochs = {num_epochs}\n
        generator_learning_rate = {generator_learning_rate}\n
        generator_decay = {generator_decay}\n
        generator_beta_1 = {generator_beta_1}\n
        generator_amsgrad = {generator_amsgrad}\n
        """
    )
    
    with open(f"./grid_search/{test_id}/hyperparams.txt", "w") as outfile:
        outfile.write(h_data)
    
    d_model = build_discriminator()
    g_model = build_generator(latent_dim)
    
    LOAD = False
    
    load_id = 2
    if LOAD == True:
        d_model.load_weights(f"./grid_search/{load_id}/saved_model/d_model.h5")
        g_model.load_weights(f"./grid_search/{load_id}/saved_model/g_model.h5")

    #d_model.summary()
    #g_model.summary()

    gan = GAN(d_model, g_model, latent_dim)

    d_optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0001,
        #decay = 0.00001,
        beta_1=0.5,
        amsgrad = True
    )
    g_optimizer = tf.keras.optimizers.Adam(
        learning_rate=generator_learning_rate,
        #decay = generator_decay,
        beta_1 = generator_beta_1,
        amsgrad = True,
    )
    gan.compile(d_optimizer, g_optimizer)

    #images_path = glob("./datasets/SAI/8/**")
    #images_path = images_path + glob("./datasets/SAI/8/**")
    
    #images_path = glob("./datasets/SAI/7/**")
    #images_path = images_path + glob("./datasets/SAI/7/**")
    
    #images_path = glob("./datasets/SAI/6/**")
    #images_path = images_path + glob("./datasets/SAI/6/**")
    
    #images_path = glob("./datasets/SAI/5/**")
    #images_path = images_path + glob("./datasets/SAI/5/**")
    
    #images_path = glob("./datasets/SAI/4/**")
    #images_path = images_path + glob("./datasets/SAI/4/**")
    
    #images_path = glob("./datasets/SAI/3/**")
    #images_path = images_path + glob("./datasets/SAI/3/**")
    
    #images_path = glob("./datasets/SAI/2/**")
    #images_path = images_path + glob("./datasets/SAI/2/**")
    
    #images_path = glob("./datasets/SAI/1/**")
    #images_path = images_path + glob("./datasets/SAI/1/**")
    
    images_path = glob("./datasets/SAI/0/**")
    #images_path = images_path + glob("./datasets/SAI/0/**")
    
    images_dataset = tf_dataset(images_path, batch_size)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs}")
        gan.fit(images_dataset, epochs=1)

        n_samples = 32
        noise = np.random.normal(size=(n_samples, latent_dim))
        examples = g_model.predict(noise)

        save_plot(examples, epoch, int(np.sqrt(n_samples)))
        if epoch % 10 == 0:
            g_model.save(f"./grid_search/{test_id}/saved_model/g_model.h5")
            d_model.save(f"./grid_search/{test_id}/saved_model/d_model.h5")