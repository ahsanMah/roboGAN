import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow.keras.layers import Input, Dense


# """Description 

# Parameters
# ----------
# file_loc : str
#     The file location of the spreadsheet
# print_cols : bool, optional
#     A flag used to print the columns to the console (default is
#     False)

# Returns
# -------
# list
#     a list of strings used that are the header columns
# """


class RoboGAN:
    
    def __init__(self,nDimX, nDimY, \
                lambda1=10,lambda2=10, \
                learning_rate=2e-4,beta1=0.5):      
        """
        Parameters
        ----------
            lambda1: integer, weight for forward cycle loss (X->Y->X)
            lambda2: integer, weight for backward cycle loss (Y->X->Y)
            learning_rate: float, initial learning rate for Adam
            beta1: float, momentum term of Adam 
        """
        
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.learning_rate = learning_rate
        self.beta1 = beta1

        self.BCE = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        
        # Generating forward GAN = G: X -> Y
        X = tf.placeholder(tf.float32, shape=(None,nDimX), name = "X")
        self.G = self.generator(name="G")
        self.D_Y = self.discriminator(name="D_Y")

        # Generating backward GAN = F: Y -> X
        Y = tf.placeholder(tf.float32, shape=(None, nDimY), name="Y")
        self.F = self.generator(name="F")
        self.D_X = self.discriminator(name="D_X")

        return


    # def create_model(nDimX, nDimY):
    #     # Generating forward GAN = G: X -> Y
    #     X = tf.placeholder(tf.float32, shape=(None,nDimX), name = "X")
    #     self.G = self.generator(name="G")
    #     D_Y = GAN.discriminator(name="D_Y")

    #     # Generating backward GAN = F: Y -> X
    #     Y = tf.placeholder(tf.float32, shape=(None, nDimY), name="Y")
    #     F = GAN.generator(name="F")
    #     D_X = GAN.discriminator(name="D_X")



    def generator(self, name, hidden_nodes = [10,10], input_dim = 2, output_dim = 2, initializer = tf.keras.initializers.he_normal() ):
        
        with tf.variable_scope(name):
            # Tells model to expect batches of input_dim features
            # inputs = Input(shape=(input_dim,))

            gen = tf.keras.Sequential([
                Dense(units = hidden_nodes[0], activation="elu", input_dim=input_dim, kernel_initializer=initializer),
                Dense(units = hidden_nodes[1], activation="tanh",kernel_initializer=initializer),
                Dense(units = output_dim, kernel_initializer=initializer)
                ])

        return gen

    def discriminator(self,name,hidden_nodes = [10,10], input_dim = 2,
                    initializer = tf.keras.initializers.he_normal()):
        """
        Discriminator will output a single real value which will be 
        interpreted as the probability that the given sample is 'real'
        """
        with tf.variable_scope(name):
            disc = tf.keras.Sequential([
                Dense(units = hidden_nodes[0], activation="elu", input_dim=input_dim, kernel_initializer=initializer),
                Dense(units = hidden_nodes[1], activation="elu",kernel_initializer=initializer),
                Dense(units = 1,kernel_initializer=initializer)
                ])
            # print(disc.summary())

        return disc


    # def get_optimizers():




    def cycle_consistency_loss(self, G_Fx, F_Gy, x, y):
        """ Cycle consistency loss (L1 norm)
        """
        forward_loss = tf.reduce_mean(tf.abs(F_Gy-x))
        backward_loss = tf.reduce_mean(tf.abs(G_Fx-y))
        loss = self.lambda1*forward_loss + self.lambda2*backward_loss
        return loss

    
    # NOTE: Both losses assume non-sigmoid logits as input

    def generator_loss(self, fake_output, heuristic=True):
        """
        Either use Binary Cross Entropy to approximate E[ D (G(x)) ]
        or the heuristic suggested in the original Goodfellow
        """

        if heuristic:
            return -0.5 * tf.reduce_mean(self.safe_log(fake_output))

        return self.BCE(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output):

        real_loss = self.BCE(tf.ones_like(real_output), real_output)
        fake_loss = self.BCE(tf.zeros_like(fake_output), fake_output)
        total_loss = 0.5 * (real_loss + fake_loss)

        return total_loss

    def safe_log(self, X, epsilon = 1e-24):
        return tf.log(X+epsilon)