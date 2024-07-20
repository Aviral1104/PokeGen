import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os


tf.random.set_seed(42)
np.random.seed(42)

# parameters
latent_dim = 100
height = 64
width = 64
channels = 3
batch_size = 64
epochs = 20000
save_interval = 500

save_dir = 'D:\\Python Project\\generated_images'
os.makedirs(save_dir, exist_ok=True)

def load_and_preprocess_data(data_dir):
    image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.png') or f.endswith('.jpg')]
    images = []
    for path in image_paths:
        img = keras.preprocessing.image.load_img(path, target_size=(height, width))
        img_array = keras.preprocessing.image.img_to_array(img)
        images.append(img_array)
    
    images = np.array(images)
    images = (images - 127.5) / 127.5  # Normalize to [-1, 1]
    return images

data_dir = 'D:\\Python Project\\images\\images'
images = load_and_preprocess_data(data_dir)


def build_generator():
    model = keras.Sequential([
        keras.layers.Dense(4 * 4 * 256, input_shape=(latent_dim,)),
        keras.layers.Reshape((4, 4, 256)),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.2),
        
        keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.2),
        
        keras.layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.2),
        
        keras.layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.2),
        
        keras.layers.Conv2DTranspose(channels, kernel_size=4, strides=2, padding='same', activation='tanh')
    ])
    return model


def build_discriminator():
    model = keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=4, strides=2, padding='same', input_shape=(height, width, channels)),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Dropout(0.3),
        
        keras.layers.Conv2D(64, kernel_size=4, strides=2, padding='same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Dropout(0.3),
        
        keras.layers.Conv2D(128, kernel_size=4, strides=2, padding='same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Dropout(0.3),
        
        keras.layers.Conv2D(256, kernel_size=4, strides=2, padding='same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Dropout(0.3),
        
        keras.layers.Flatten(),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


generator = build_generator()
discriminator = build_discriminator()


discriminator.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])


discriminator.trainable = False
gan = keras.Sequential([generator, discriminator])
gan.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy')

# Training loop
def train_gan(generator, discriminator, gan, images, epochs, batch_size, latent_dim, save_interval):
    batches_per_epoch = images.shape[0] // batch_size
    
    for epoch in range(epochs):
        for _ in range(batches_per_epoch):

            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            generated_images = generator.predict(noise)
            
            real_images = images[np.random.randint(0, images.shape[0], batch_size)]
            
            discriminator_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)) * 0.9)  # Label smoothing
            discriminator_loss_fake = discriminator.train_on_batch(generated_images, np.zeros((batch_size, 1)))
            discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)
            
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            gan_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
        
        if epoch % save_interval == 0:
            print(f"Epoch {epoch}, Discriminator Loss: {discriminator_loss[0]}, Generator Loss: {gan_loss}")
            generate_and_save_images(generator, epoch, noise)

def generate_and_save_images(model, epoch, test_input):
    predictions = model.predict(test_input)
    predictions = (predictions + 1) / 2.0  
    
    fig = plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i])
        plt.axis('off')
    
    plt.savefig(os.path.join(save_dir, f'generated_pokemon_epoch_{epoch}.png'))
    plt.close()

physical_devices = tf.config.list_physical_devices('GPU') #using my gpu lol, it takes ages to run w/o it
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

train_gan(generator, discriminator, gan, images, epochs, batch_size, latent_dim, save_interval)

noise = np.random.normal(0, 1, (16, latent_dim))
generate_and_save_images(generator, epochs, noise)
