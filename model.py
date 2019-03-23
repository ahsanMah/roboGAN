import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow.keras.layers import Input, Dense

def generator(name,hidden_nodes = [10,10], input_dim = 2, output_dim = 2,
                initializer = tf.keras.initializers.he_normal()):
    
    with tf.variable_scope(name):
        # Tells model to expect batches of input_dim features
        # inputs = Input(shape=(input_dim,))

        gen = tf.keras.Sequential([
            Dense(units = hidden_nodes[0], activation="elu", input_dim=input_dim, kernel_initializer=initializer),
            Dense(units = hidden_nodes[1], activation="elu",kernel_initializer=initializer),
            Dense(units = output_dim, activation="elu", kernel_initializer=initializer)
            ])
        # print(gen.summary())

    return gen

def discriminator(name,hidden_nodes = [10,10], input_dim = 2, output_dim = 2,
                initializer = tf.keras.initializers.he_normal()):
    
    with tf.variable_scope(name):
        disc = tf.keras.Sequential([
            Dense(units = hidden_nodes[0], activation="elu", input_dim=input_dim, kernel_initializer=initializer),
            Dense(units = hidden_nodes[1], activation="elu",kernel_initializer=initializer),
            Dense(units = output_dim, activation="elu", kernel_initializer=initializer)
            ])
        # print(disc.summary())

    return disc

def cycle_consistency_loss(self, G, F, x, y):
    """ cycle consistency loss (L1 norm)
    """
    forward_loss = tf.reduce_mean(tf.abs(F(G(x))-x))
    backward_loss = tf.reduce_mean(tf.abs(G(F(y))-y))
    loss = self.lambda1*forward_loss + self.lambda2*backward_loss
    return loss
