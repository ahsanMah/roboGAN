import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow.keras.layers import Input, Dense
#from robot import compareInternalPositions

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
                learning_rate=2e-4,beta1=0.5, endposDiscriminator = False, endposGenerator = False, allPosGenerator = False, lengthsX=[2,1], lengthsY=[2,1], conditional_disc=False):      
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
        self.allPosGenerator = allPosGenerator
        self.endposDiscriminator = endposDiscriminator
        self.nrLinksX = round(nDimX/5)
        self.nrLinksY = round(nDimY/5)
        self.armlengthY = sum(lengthsY)
        self.armlengthX = sum(lengthsX)
        self.lengthsX = lengthsX
        self.lengthsY = lengthsY
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.conditional_disc = conditional_disc

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
#         self.opt_D_X = tf.train.MomentumOptimizer(learning_rate=self.learning_rate*10, momentum=0.9, use_nesterov=True)
#         self.opt_D_Y = tf.train.MomentumOptimizer(learning_rate=self.learning_rate*10, momentum=0.9, use_nesterov=True)
        
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
                Dense(units = hidden_nodes[2], activation="elu", kernel_initializer=initializer),
                Dense(units = output_dim, activation=sliceTanh, kernel_initializer=initializer)
                ])

        return gen

    def discriminator(self,name,hidden_nodes = [50,50,50], input_dim = 2,
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
        
        dimensions = input_dim
        
        if self.conditional_disc:
            dimensions += 1
        
        with tf.variable_scope(name):
            disc = tf.keras.Sequential([
                Dense(units = hidden_nodes[0], activation="elu", input_dim=dimensions, kernel_initializer=initializer),
                Dense(units = hidden_nodes[1], activation="elu",kernel_initializer=initializer),
                Dense(units = hidden_nodes[2], activation="elu",kernel_initializer=initializer),
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
    
    
    
    
    def computeY(self,alphas, lengths):
        print("Alphas: {}".format(alphas))
        print("Lengths: {}".format(lengths))
        nrSamples = alphas.shape[0].value
        nrLinks = alphas.shape[1].value
        
        
        y = [] #tf.zeros((nrSamples,nrLinks * 3))
        for i in range(nrSamples): #nrSamples
            positions = [[]]
            norms = [[]]
            pose = tf.eye(3)#forwardR(alphas[i,nrLinks - 1],l[nrLinks - 1])
            for j in range(nrLinks):
                pose = tf.matmul(pose, self.R(alphas[i,j],lengths[j]))
                #print(pose)
                #print(pose)
#                 y[i,2*j:2*(j+1)]=pose[0:2,2] #position
                _pose = [[pose[0,2], pose[1,2]]]
                positions=tf.concat((positions,_pose),axis=1) 
#                 positions.extend(pose[0:2,2])
                #if j>0:
#                 y[i,(nrLinks*2) + j] = tf.norm(pose[0:2,2],axis=0) #dist from origin
#                 _norm = [[tf.norm(pose[0:2,2])]]

#                 norms = tf.concat([norms,_norm], axis=1)

#             y.append(tf.concat((positions,norms),axis=1))
            y.append(positions)
        
#         print(y)
        y = tf.squeeze(tf.stack(y))
        
        return y


    def R(self, alpha, l ):
        return tf.convert_to_tensor([[tf.math.cos(alpha),-tf.math.sin(alpha), tf.math.cos(alpha)*l],[tf.math.sin(alpha), tf.math.cos(alpha), tf.math.sin(alpha)*l],[0, 0, 1]])

#     def getOriginalAngles(self,s,c):
#         if tf.math.greater(c, 0):
#             ang1 = tf.math.asin(s)
#             if s >= 0:
#                 ang2 = tf.math.acos(c)
#             else:
#                 ang2 = -tf.math.acos(c)
#         else:
#             if s >= 0:
#                 ang1 = tf.pi - tf.math.asin(s)
#                 ang2 = tf.math.acos(c)
#             else:
#                 ang1 = -tf.pi - tf.math.asin(s)
#                 ang2 = -tf.math.acos(c)

#         return (ang1 + ang2)/2
    
    # Rewriting function for tensorflow
    def getOriginalAngles(self,s,c):
        
        condition = tf.math.greater(c, 0)
        s_cond = tf.math.less(s,0)
        
        cond_true = lambda: tf.math.asin(s)
        cond_false = lambda: tf.cond(s_cond, 
                                     lambda: -np.pi - tf.math.asin(s),
                                     lambda: (np.pi - tf.math.asin(s)))
        
        
        s_cond_a2_true = lambda: -tf.math.acos(c)
        s_cond_a2_false = lambda: tf.math.acos(c)
        
        
        ang1 = tf.cond(condition, cond_true, cond_false)
        ang2 = tf.cond(s_cond, s_cond_a2_true, s_cond_a2_false)

        return (ang1 + ang2)/2
    
    
    
#     def getAvgAngle(self,dataRow, nrLinks):
#         print('nrLinks')
#         print(nrLinks)
#         angles = tf.zeros([1,nrLinks*2])
#         for j in range(nrLinks):
#                 print('sine value')
#                 print(tf.math.greater(dataRow[2*j], 0))
#                 angles[0,j] = self.getOriginalAngles(dataRow[2*j], dataRow[2*j+1])
#         #a1,a2 = self.getOriginalAngles(s,c)
#         return angles

    def getAvgAngle(self,dataRow, nrLinks):
        
        angles = []
        for j in range(nrLinks):
                angles.append(self.getOriginalAngles(dataRow[2*j], dataRow[2*j+1]))
        
        angles  = tf.stack(angles)
        
        return angles

    
#     def positionsFromAngles(self,data, nrLinks, lengths):
#         print(tf.shape(data)[0])
#         angles=tf.zeros_like(data) #[tf.shape(data)[0], nrLinks]
#         for i in range(100): #tf.shape(data)[0] #replace with real batch size
#             angles[i,:] = self.getAvgAngle(data[i,:], nrLinks)
#         return self.computeY(angles, lengths)
    
    def positionsFromAngles(self,data, nrLinks, lengths):
        print(tf.shape(data)[0])
        angles = []
        for i in range(100): #tf.shape(data)[0] #replace with real batch size
            angles.append(self.getAvgAngle(data[i,:], nrLinks))
        
        angles = tf.stack(angles)
        print("Calculated Angles")
        print(angles)
        return self.computeY(angles, lengths)
    
    def compareInternalPositions(self, data, nrLinks, lengths): #pos from angles vs real pos
        anglePos = self.positionsFromAngles(data, nrLinks, lengths) #only positions, not distances
        pos = data[:,nrLinks * 2 : nrLinks * 4]
        norms = [tf.linalg.norm(anglePos - pos, axis=1)]
        norms = tf.transpose(norms)
        print("Norms: {}".format(norms))
        return norms #tf.transpose(tf.reduce_mean(norms))
    
    def valid_config_loss(self, data, nrLinks, lengths):
        return self.compareInternalPositions(data, nrLinks, lengths)

    
    def end_effector_loss(self,inputData, outputData, maxLength):
        inputLinks = round(inputData.shape[1].value/5)
        outputLinks = round(outputData.shape[1].value/5)
       
        endeff1 = inputData[:,inputLinks*4-2:inputLinks*4]
        endeff2 = outputData[:,outputLinks*4-2:outputLinks*4]
        norms = tf.norm(endeff1-endeff2, axis=1)
        avg = tf.reduce_mean(norms) #/maxLength

        return avg
    
    #ToDo fix!
    def all_positions_loss(self, inputData, outputData, maxLength):
        inputLinks = round(inputData.shape[1].value/5)
        outputLinks = round(outputData.shape[1].value/5)
       
        #ToDo generalize for different numbers of links:
        loss = 0
        for i in range(inputLinks): #inputLinks
            pos1 = inputData[:,(inputLinks*2)+ (2 * i) :(inputLinks*2) + (2*i) + 1]
            pos2 = outputData[:,(outputLinks*2) + (2 * i) : (outputLinks*2 + 2*i) + 1]
            norms = tf.norm(pos1-pos2, axis=1)
            avg = tf.reduce_mean(norms)  #/maxLength
            loss += avg
        loss = loss/inputLinks
        return loss
            
    
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