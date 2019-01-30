from keras.layers import Input, Dense, Reshape, Add, Activation, Lambda, Conv1D
from keras import layers
from keras.models import Model
from keras.optimizers import RMSprop
import keras.backend as K
import numpy as np


def stop_grad(ys):
    return K.stop_gradient(ys[0] - ys[1]) + ys[1]


class WGAN(object):
    def __init__(self, input_shape, latent_dim, tau=0.75, gumbel=False, hard_gumbel=False):
        self.channels = 1
        self.inputShape_ = input_shape
        self.latentDim_ = latent_dim
        self.gumbel_ = gumbel
        self.hardGumbel_ = hard_gumbel

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        self.tau_ = tau
        optimizer = RMSprop()

        # Build and compile the critic
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss,
                            optimizer=optimizer,
                            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        gen_ins = self.generator.inputs
        img = self.generator(gen_ins)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(gen_ins, valid)
        self.combined.compile(loss=self.wasserstein_loss,
                              optimizer=optimizer,
                              metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):
        L = self.inputShape_[0]
        z = Input(shape=(self.latentDim_,))
        x = Dense(100 * self.inputShape_[0])(z)
        x = Reshape((L, 100,))(x)

        # res block 1:
        res_in = x
        x = layers.Activation('relu')(x)
        x = Conv1D(100, 5, padding='same')(x)
        x = layers.Activation('relu')(x)
        x = Conv1D(100, 5, padding='same')(x)
        x = Lambda(lambda z: z * 0.3)(x)
        x = Add()([res_in, x])

        # res block 2:
        res_in = x
        x = layers.Activation('relu')(x)
        x = Conv1D(100, 5, padding='same')(x)
        x = layers.Activation('relu')(x)
        x = Conv1D(100, 5, padding='same')(x)
        x = Lambda(lambda z: z * 0.3)(x)
        x = Add()([res_in, x])

        # res block 3:
        res_in = x
        x = layers.Activation('relu')(x)
        x = Conv1D(100, 5, padding='same')(x)
        x = layers.Activation('relu')(x)
        x = Conv1D(100, 5, padding='same')(x)
        x = Lambda(lambda z: z * 0.3)(x)
        x = Add()([res_in, x])

        # res block 4:
        res_in = x
        x = layers.Activation('relu')(x)
        x = Conv1D(100, 5, padding='same')(x)
        x = layers.Activation('relu')(x)
        x = Conv1D(100, 5, padding='same')(x)
        x = Lambda(lambda z: z * 0.3)(x)
        x = Add()([res_in, x])

        # res block 5:
        res_in = x
        x = layers.Activation('relu')(x)
        x = Conv1D(100, 5, padding='same')(x)
        x = layers.Activation('relu')(x)
        x = Conv1D(100, 5, padding='same')(x)
        x = Lambda(lambda z: z * 0.3)(x)
        x = Add()([res_in, x])

        x = Conv1D(self.inputShape_[-1], 1, padding='same')(x)

        logits = x
        if self.gumbel_:
            #             U = Input(tensor=K.random_uniform(K.shape(logits), 0, 1))
            eps = 1e-20
            g = Lambda(
                lambda y: 1. / (self.tau_) * (y - K.log(-K.log(K.random_uniform(K.shape(logits), 0, 1) + eps) + eps)))(
                logits)
            out = layers.Activation('softmax')(g)
            if self.hardGumbel_:
                k = K.shape(logits)[-1]
                out_hard = Lambda(lambda y: K.tf.cast(K.tf.equal(y, K.tf.reduce_max(y, 1, keepdims=True)), y.dtype))(
                    out)
                out = Lambda(stop_grad)([out_hard, out])
            model = Model(inputs=z, outputs=out)
        else:
            out = Activation('softmax')(logits)
            model = Model(inputs=z, outputs=out)

        return model

    def build_critic(self):
        L = self.inputShape_[0]
        x = Input(shape=self.inputShape_)
        y = Conv1D(100, 1, padding='same')(x)

        # res block 1:
        res_in = y
        y = layers.Activation('relu')(y)
        y = Conv1D(100, 5, padding='same')(y)
        y = layers.Activation('relu')(y)
        y = Conv1D(100, 5, padding='same')(y)
        y = Lambda(lambda z: z * 0.3)(y)
        y = Add()([res_in, y])

        # res block 2:
        res_in = y
        y = layers.Activation('relu')(y)
        y = Conv1D(100, 5, padding='same')(y)
        y = layers.Activation('relu')(y)
        y = Conv1D(100, 5, padding='same')(y)
        y = Lambda(lambda z: z * 0.3)(y)
        y = Add()([res_in, y])

        # res block 3:
        res_in = y
        y = layers.Activation('relu')(y)
        y = Conv1D(100, 5, padding='same')(y)
        y = layers.Activation('relu')(y)
        y = Conv1D(100, 5, padding='same')(y)
        y = Lambda(lambda z: z * 0.3)(y)
        y = Add()([res_in, y])

        # res block 4:
        res_in = y
        y = layers.Activation('relu')(y)
        y = Conv1D(100, 5, padding='same')(y)
        y = layers.Activation('relu')(y)
        y = Conv1D(100, 5, padding='same')(y)
        y = Lambda(lambda z: z * 0.3)(y)
        y = Add()([res_in, y])

        # res block 5:
        res_in = y
        y = layers.Activation('relu')(y)
        y = Conv1D(100, 5, padding='same')(y)
        y = layers.Activation('relu')(y)
        y = Conv1D(100, 5, padding='same')(y)
        y = Lambda(lambda z: z * 0.3)(y)
        y = Add()([res_in, y])
        y = Reshape((L * 100,))(y)
        out = Dense(1)(y)

        model = Model(inputs=x, outputs=out)
        return model

    def train(self, X_train, epochs, batch_size=128, verbose=0):
        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]

                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.latentDim_))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the critic
                d_loss_real = self.critic.train_on_batch(imgs, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

                # ---------------------
                #  Train Generator
                # ---------------------
            noise = np.random.normal(0, 1, (batch_size, self.latentDim_))
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            if verbose:
                print("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

    def sample(self, noise):
        sampled_x = self.generator.predict(noise)
        return sampled_x

