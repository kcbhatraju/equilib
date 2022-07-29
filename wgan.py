import tensorflow as tf
import matplotlib.pyplot as plt
import math, json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

num_epochs = 15
gen_lr = 0.01
discrim_lr = 0.03
batch_size = 32
latent_dim = 100
fixed_noise = tf.random.normal([batch_size, latent_dim])

real_imgs = tf.keras.utils.image_dataset_from_directory("imgs",
                                                        label_mode=None,
                                                        batch_size=batch_size,
                                                        image_size=(128,128))


class ResBlock(tf.keras.Model):
    def __init__(self, filters,channels,kernel_size):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters,kernel_size,padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        
        self.conv2 = tf.keras.layers.Conv2D(channels,kernel_size,padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        
        self.act = tf.keras.layers.Activation("relu")
        self.add = tf.keras.layers.Add()
    
    def call(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.act(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        x = self.add([x,input])
        x = self.act(x)
        
        return x


class Generator(tf.keras.Model):
    def __init__(self,output_shape=(128,128),channels=3):
        super().__init__()
        self.fc = tf.keras.layers.Dense(math.prod(output_shape)*channels)
        self.reshape = tf.keras.layers.Reshape((*output_shape,channels))
        
        self.rb1 = ResBlock(32,channels,(3,3))
        self.rb2 = ResBlock(64,channels,(3,3))
        self.rb3 = ResBlock(128,channels,(3,3))
        self.rb4 = ResBlock(256,channels,(3,3))
        
        self.conv = tf.keras.layers.Conv2D(3,(3,3),padding="same")
    
    def call(self,input):
        x = self.fc(input)
        x = self.reshape(x)
        
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        
        x = self.conv(x)
        
        return x


class Discriminator(tf.keras.Model):
    def __init__(self,channels=3):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(channels,(3,3),padding="same")
        
        self.rb1 = ResBlock(256,channels,(3,3))
        self.rb2 = ResBlock(128,channels,(3,3))
        self.rb3 = ResBlock(64,channels,(3,3))
        self.rb4 = ResBlock(32,channels,(3,3))
        
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(1)
    
    def call(self,input):
        x = self.conv(input)
        
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        
        x = self.pool(x)
        x = self.fc(x)
        
        return x

gen = Generator()
discrim = Discriminator()

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
        
        if (epoch % 5) == 0 or epoch + 1 == epochs:
            plt.imshow(gen(fixed_noise)[0].numpy().astype("uint8"))
            plt.show()

train(real_imgs, num_epochs)

gen.save("models/generator.h5")
discrim.save("models/discriminator.h5")
