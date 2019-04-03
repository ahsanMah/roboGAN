import tensorflow as tf
from model import RoboGAN
from datetime import datetime
import os
import logging

def train_step(GAN):
     """
     Simulates one training step of a Cycle GAN
     Generates fake forward and backward samples and uses discriminators to evaluate them
     Next it uses real data batches to evaluate gradients for discriminators

     Parameters
     ----------
          GAN: The Cycle GAN model which will be used to generate data

     Returns
     ----------
          losses: List of all the losses in order
          gradients: List of all calculated gradients in order 
     """
     X,Y = GAN.X, GAN.Y
     G,F = GAN.G, GAN.F
     D_X, D_Y = GAN.D_X, GAN.D_Y

     losses = []
     gradients = []

     with tf.GradientTape() as G_tape,   \
     tf.GradientTape() as D_Y_tape, \
     tf.GradientTape() as F_tape,   \
     tf.GradientTape() as D_X_tape:
     
          # Generating forward samples X -> Y
          fake_Y = G(X)
          D_Gy_logits = D_Y(fake_Y)
          D_Y_logits = D_Y(Y)
          
          # Generating backward samples X <- Y
          fake_X = F(Y)
          D_Fx_logits = D_X(fake_X) # D_Fx = D_X ( F(Y) )
          D_X_logits = D_X(X)
          
          G_Fx = G(fake_X)
          F_Gy = F(fake_Y)
          
          cycle_loss = GAN.cycle_consistency_loss(G_Fx, F_Gy, X, Y)
          
          G_loss = GAN.generator_loss(D_Gy_logits, heuristic=False) + cycle_loss
          D_Y_loss = GAN.discriminator_loss(real_output= D_Y_logits, fake_output= D_Gy_logits)
          
          F_loss = GAN.generator_loss(D_Fx_logits, heuristic=False) + cycle_loss
          D_X_loss = GAN.discriminator_loss(real_output= D_X_logits, fake_output= D_Fx_logits)
          
          G_gradients = G_tape.gradient(G_loss, G.trainable_variables)
          D_Y_gradients = D_Y_tape.gradient(D_Y_loss, D_Y.trainable_variables)
          
          F_gradients = F_tape.gradient(F_loss, F.trainable_variables)
          D_X_gradients = D_X_tape.gradient(D_X_loss, D_X.trainable_variables)

          losses.extend([G_loss, F_loss, D_Y_loss, D_X_loss])
          gradients.extend([G_gradients, F_gradients, D_Y_gradients, D_X_gradients])
          
          traienrs = GAN.optimize(gradients)

     return losses, gradients, trainers