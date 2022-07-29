import json
import matplotlib.pyplot as plt

import tensorflow as tf

num_epochs = 1000
gen_lr = 0.001
discrim_lr = 0.001
batch_size = 32
latent_dim = 100
fixed_noise = tf.random.normal([batch_size, latent_dim])

real_imgs = tf.keras.utils.image_dataset_from_directory("imgs",
                                                        label_mode=None,
                                                        batch_size=batch_size,
                                                        image_size=(128,128))


gen = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32*32*256,use_bias=False,input_shape=(latent_dim,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    
    tf.keras.layers.Reshape((32,32,256)),
    tf.keras.layers.Conv2DTranspose(128,(5,5),strides=(1,1),padding="same",use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    
    tf.keras.layers.Conv2DTranspose(64,(5,5),strides=(2,2),padding="same",use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    
    tf.keras.layers.Conv2DTranspose(1,(5,5),strides=(2,2),padding="same",use_bias=False)
])

discrim = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64,(5,5),strides=(2,2),padding="same",input_shape=(128,128,1)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Conv2D(128,(5,5),strides=(2,2),padding="same"),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1)
])

loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
gen_optim = tf.keras.optimizers.Adam(gen_lr)
discrim_optim = tf.keras.optimizers.Adam(discrim_lr)

def discrim_loss(real, fake):
    real_loss = loss(tf.ones_like(real),real)
    fake_loss = loss(tf.zeros_like(fake),fake)
    total_loss = real_loss + fake_loss
    return total_loss

def gen_loss(fake):
    return loss(tf.ones_like(fake),fake)

def progress(current,total,**kwargs):
    done_token, current_token = ("=", ">")
    token_arr = []
    token_arr.extend([done_token]*current)
    if (total-current): token_arr.extend([current_token])
    attrs = json.dumps(kwargs).replace('"',"")[1:-1]
    final = f"{current}/{total} [{''.join(token_arr)}{' '*max(0,total-current-1)}] - {attrs}"
    print(final,end=("\r","\n\n")[current==total])

@tf.function
def train_step(imgs):
    noise = tf.random.normal([batch_size, latent_dim])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as discrim_tape:
        gen_imgs = gen(noise, training=True)
        
        real_output = discrim(imgs, training=True)
        fake_output = discrim(gen_imgs, training=True)

        gen_loss_val = gen_loss(fake_output)
        discrim_loss_val = discrim_loss(real_output, fake_output)

    gen_grad = gen_tape.gradient(gen_loss_val, gen.trainable_variables)
    discrim_grad = discrim_tape.gradient(discrim_loss_val, discrim.trainable_variables)

    gen_optim.apply_gradients(zip(gen_grad, gen.trainable_variables))
    discrim_optim.apply_gradients(zip(discrim_grad, discrim.trainable_variables))
    
    return tf.reduce_mean(gen_loss_val), tf.reduce_mean(discrim_loss_val)

def train(dataset, epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for idx, batch in enumerate(dataset):
            gen_loss_mean_val, discrim_loss_mean_val = train_step(batch)
            progress(idx+1,len(dataset),gen_loss=round(gen_loss_mean_val.numpy().astype(float),2),discrim_loss=round(discrim_loss_mean_val.numpy().astype(float),2))
        
        if (epoch % 200) == 0 or epoch + 1 == epochs:
            plt.imshow(gen(fixed_noise)[0].numpy().astype("uint8"))
            plt.show()

train(real_imgs, num_epochs)

gen.save("models/generator.h5")
discrim.save("models/discriminator.h5")