import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow.keras.layers import Input, Dense

class RoboGAN:
    
    """
    A simple CycleGAN implementation that is based on fully connected networks.
    Gives access to generators and discriminators and their related fucntions.

    Attributes
    ----------
    G : The forward Generator that will transform X into Y
    D_Y : The Discriminator associated with G to distinguish real Y from fake Y samples
    X : Placeholder node for G's input (should be instantiated inside feed_dict)

    F : The backward Generator that will transform Y into X
    D_X : The Discriminator associated with F to distinguish real X from fake X samples
    Y : Placeholder node for F's input (should be instantiated inside feed_dict)

    """
    
    def __init__(self,nDimX, nDimY, \
                lambda1=10,lambda2=10, \
                learning_rate=2e-4,beta1=0.5, endposDiscriminator = False, endposGenerator = False, armlengthG=2, armlengthF=2):      
        """
        Parameters
        ----------
            nDimX, nDimY: int
                Dimensions for the two domains X and Y that the CycleGAN will transform between
            lambda1: integer
                weight for forward cycle loss (X->Y->X)
            lambda2: integer
                weight for backward cycle loss (Y->X->Y)
            learning_rate: float
                initial learning rate for Adam
            beta1: float
                momentum term of Adam 
        """
        self.endposGenerator = endposGenerator
        self.endposDiscriminator = endposDiscriminator
        self.nrLinksX = round(nDimX/5)
        self.nrLinksY = round(nDimY/5)
        self.armlengthG = armlengthG
        self.armlengthF = armlengthF
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.learning_rate = learning_rate
        self.beta1 = beta1

        self.BCE = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        
        self.optimizers_init = False

        # Generating forward GAN = G: X -> Y
        self.X = tf.placeholder(tf.float32, shape=(None,nDimX), name= "X")
        self.G = self.generator(name= "G", input_dim= nDimX, output_dim= nDimY)
        
        if(self.endposDiscriminator):
            dimY = nDimY+2
            dimX = nDimX+2
        else:
            dimY = nDimY
            dimX = nDimX
        self.D_Y = self.discriminator(name= "D_Y", input_dim= dimY)

        # Generating backward GAN = F: Y -> X
        self.Y = tf.placeholder(tf.float32, shape=(None, nDimY), name="Y")
        self.F = self.generator(name= "F", input_dim= nDimY, output_dim= nDimX)
        self.D_X = self.discriminator(name= "D_X", input_dim= dimX)

        return

    def make_optimizers(self):
        """
        Initializes the Adam optimizers with class learning parameters
        Four optimizers are created, one for each network
        """

        self.opt_G = tf.train.AdamOptimizer(learning_rate= self.learning_rate, beta1= self.beta1)
        self.opt_F = tf.train.AdamOptimizer(learning_rate= self.learning_rate, beta1= self.beta1)
        self.opt_D_Y = tf.train.AdamOptimizer(learning_rate= self.learning_rate, beta1= self.beta1)
        self.opt_D_X = tf.train.AdamOptimizer(learning_rate= self.learning_rate, beta1= self.beta1)
        
        self.optimizers_init = True
        return 

    def optimize(self, gradients):
        """
        Builds the graph nodes for running an optimization step
        
        Parameters
        ----------
            gradients: Tensorflow nodes array
                List of gradients expected in the order used to unpack below

        Returns
        -------
            trainers: Tensorflow nodes array
                Nodes that represent an optimization step for each network in the CycleGAN
        """
        if self.optimizers_init == False:
            self.make_optimizers()
          
        G_gradients, F_gradients, D_Y_gradients, D_X_gradients = gradients

        train_G = self.opt_G.apply_gradients(zip(G_gradients, self.G.trainable_variables))
        train_F = self.opt_F.apply_gradients(zip(F_gradients, self.F.trainable_variables))

        train_D_Y = self.opt_D_Y.apply_gradients(zip(D_Y_gradients, self.D_Y.trainable_variables))
        train_D_X = self.opt_D_X.apply_gradients(zip(D_X_gradients, self.D_X.trainable_variables))

        trainers = [train_G, train_F, train_D_Y, train_D_X] 

        return trainers

    # TODO: Automatically generate layers using hidden_nodes array. (Using a looping condition)
    
    def genSliceTanh(self, input_dim):
        nrVals = round(input_dim/5)
        def sliceTanh(x):
            y = tf.concat([tf.tanh(x[:,0:2*nrVals]),x[:,2*nrVals:]],1)
            return y
        return sliceTanh
    
    def generator(self, name, hidden_nodes = [50, 50, 50], input_dim = 2, output_dim = 2, initializer = tf.keras.initializers.he_normal() ):
        """
        Generator will transform the input data into the output domain
        
        Parameters
        ----------
            name : str
                Namespace to be used for the model - makes layer names mroe readable
            hidden_nodes : int array 
                Number of nodes in each hidden layer (in order)
            input_dim, output_dim: int
                Dimensions of the input and output domains for this generator (G: X->Y)
            initializer: Method from keras.initializers.*, optional
                A kernel initializer to be used in each layer
        
        Returns
        -------
            gen : A tf node representing a Keras Sequential model
        """
        sliceTanh=self.genSliceTanh(input_dim)
        with tf.variable_scope(name):

            gen = tf.keras.Sequential([
                Dense(units = hidden_nodes[0], activation="elu", input_dim=input_dim, kernel_initializer=initializer),
                Dense(units = hidden_nodes[1], activation="elu", kernel_initializer=initializer),
                Dense(units = hidden_nodes[2], activation="elu", kernel_initializer=initializer),
                Dense(units = output_dim, activation=sliceTanh, kernel_initializer=initializer)
                ])

        return gen

    def discriminator(self,name,hidden_nodes = [50,50, 50], input_dim = 2,
                    initializer = tf.keras.initializers.he_normal()):
        """
        Discriminator will output a single real value which will be 
        interpreted as the probability that the given sample is 'real'
        
        Parameters
        ----------
            name : str
                Namespace to be used for the model - makes layer names mroe readable
            hidden_nodes : int array 
                Number of nodes in each hidden layer (in order)
            input_dim: int
                Dimensions of the input this discriminator has to analyze (D(X))
            initializer: Method from keras.initializers.*, optional
                A kernel initializer to be used in each layer
        
        Returns
        -------
            disc : A tf node representing a Keras Sequential model
        """
        with tf.variable_scope(name):
            disc = tf.keras.Sequential([
                Dense(units = hidden_nodes[0], activation="elu", input_dim=input_dim, kernel_initializer=initializer),
                Dense(units = hidden_nodes[1], activation="elu",kernel_initializer=initializer),
                #Dense(units = hidden_nodes[2], activation="elu",kernel_initializer=initializer),
                Dense(units = 1,kernel_initializer=initializer)
                ])
            # print(disc.summary())

        return disc

    def cycle_consistency_loss(self, G_Fx, F_Gy, x, y):
        """ 
        Cycle consistency loss (L1 norm)
        """
        forward_loss = tf.reduce_mean(tf.abs(F_Gy-x))
        backward_loss = tf.reduce_mean(tf.abs(G_Fx-y))
        loss = self.lambda1*forward_loss + self.lambda2*backward_loss
        return loss

    
    def end_effector_loss(self,inputData, outputData, maxLength):
        inputLinks = round(inputData.shape[1].value/5)
        outputLinks = round(outputData.shape[1].value/5)
       
        endeff1 = inputData[:,inputLinks*4-2:inputLinks*4]
        endeff2 = outputData[:,outputLinks*4-2:outputLinks*4]
        norms = tf.norm(endeff1-endeff2, axis=1)
        avg = tf.reduce_mean(norms)/maxLength
        return avg
            
    
    # NOTE: Both losses assume non-sigmoid logits as input

    def generator_loss(self, fake_output, heuristic=True):
        """
        Either use Binary Cross Entropy to approximate E[ D (G(x)) ]
        or the heuristic suggested in the original Goodfellow
        """

        if heuristic:
            P_X = tf.keras.backend.softmax(fake_output)
            print(fake_output)
            return -0.5 * tf.reduce_mean(self.safe_log(P_X))

        return self.BCE(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output):
        """
        Discriminator will learn to correctly recognize real outputs (all 1s)
        while simultaneously recognizing fake samples (all zeros)
        """
        
        real_loss = self.BCE(tf.ones_like(real_output), real_output)
        fake_loss = self.BCE(tf.zeros_like(fake_output), fake_output)
        total_loss = 0.5 * (real_loss + fake_loss)

        return total_loss

    def safe_log(self, X, epsilon = 1e-24):
        """
        To prevent runtime errors from log(0)
        """
        return tf.log(X+epsilon)