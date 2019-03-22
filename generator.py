import tensorflow as tf 


class Generator:

    def __init__(self, name, hidden_nodes = [10,10], output_dim = 2):
        self.name = name
        self.reuse = False
        self.hidden_nodes = hidden_nodes
        self.output_dim = output_dim
        # self.layer_initializer = 

    def __call__(self, X):
        '''
            X : input data -> batch_size x features
            Returns: Generated X with same dimensions
        '''
        with tf.variable_scope(self.name, reuse=self.reuse):

            hidden1 = tf.layers.dense(X, self.hidden_nodes[0],name="g_hidden1",
                                    activation=tf.nn.elu)
            hidden2 = tf.layers.dense(hidden1, self.hidden_nodes[1],name="g_hidden2",
                                    activation=tf.nn.tanh)
            logits = tf.layers.dense(hidden2, self.output_dim, name = "g_logits" )

            self.reuse = True
    
        return logits