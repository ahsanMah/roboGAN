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
     nrLinksX = GAN.nrLinksX
     nrLinksY = GAN.nrLinksY
     print(GAN.endposDiscriminator)
     print(GAN.endposGenerator)
     print(GAN.allPosGenerator)

     losses = []
     gradients = []
     trainers = []

     with tf.GradientTape() as G_tape,   \
     tf.GradientTape() as D_Y_tape, \
     tf.GradientTape() as F_tape,   \
     tf.GradientTape() as D_X_tape:
     
          # Generating forward samples X -> Y
          fake_Y = G(X)
          fake_X = F(Y)
          
          if(GAN.endposDiscriminator):
              print('additional discriminator loss')
              D_Gy_logits = D_Y(tf.concat([fake_Y,X[:,nrLinksX*4-2:nrLinksX*4]], axis = 1))
              D_Y_logits = D_Y(tf.concat([Y,Y[:,nrLinksY*4-2:nrLinksY*4]], axis = 1))
            
              D_Fx_logits = D_X(tf.concat([fake_X,Y[:,nrLinksY*4-2:nrLinksY*4]],axis = 1))
              D_X_logits = D_X(tf.concat([X,X[:,nrLinksX*4-2:nrLinksX*4]],axis = 1))
          else:
              print('no additional discriminator loss')
              valid_Y_loss = GAN.valid_config_loss(fake_Y, nrLinksY,GAN.lengthsY)
              zero_loss = tf.zeros_like(valid_Y_loss)
              D_Gy_logits = D_Y(tf.concat([fake_Y, valid_Y_loss], axis = 1))
              D_Y_logits = D_Y(tf.concat([Y,zero_loss], axis = 1))
            
              valid_X_loss = GAN.valid_config_loss(fake_X, nrLinksX, GAN.lengthsX)
             
              D_Fx_logits = D_X(tf.concat([fake_X, valid_X_loss], axis = 1)) # D_Fx = D_X ( F(Y) )
              D_X_logits = D_X(tf.concat([X,zero_loss], axis = 1))
            
          print('Discriminator done')
          G_Fx = G(fake_X)
          F_Gy = F(fake_Y)
          print('Cycle done')
          
          
          # Generating backward samples X <- Y
          
          
           # D_Fx = D_X ( F(Y) )
          #
          
          cycle_loss = GAN.cycle_consistency_loss(G_Fx, F_Gy, X, Y)
          
          
          
          if(GAN.endposGenerator):
              # Forward Loss
              print('end effector generator')
              G_dist_loss = GAN.end_effector_loss(X, fake_Y, 3)  
              G_loss = GAN.generator_loss(D_Gy_logits, heuristic=False) + cycle_loss + G_dist_loss
                
              # Backward Loss
              F_dist_loss = GAN.end_effector_loss(Y, fake_X, 3) 
              F_loss = GAN.generator_loss(D_Fx_logits, heuristic=False) + cycle_loss + F_dist_loss
            
          else: #ToDo automate maxLength
              if(GAN.allPosGenerator):
                  print('all pos generator')
                  # Forward Loss
                  #G_dist_loss = GAN.end_effector_loss(X, fake_Y, 3)  
                  G_all_pos_loss = GAN.all_positions_loss(X, fake_Y, 3)  
                  print(G_all_pos_loss)
                  G_loss = GAN.generator_loss(D_Gy_logits, heuristic=False) + cycle_loss + G_all_pos_loss 
                
                  # Backward Loss
                  #F_dist_loss = GAN.end_effector_loss(Y, fake_X, 3) 
                  F_all_pos_loss = GAN.all_positions_loss(Y, fake_X, 3) 
                  print(F_all_pos_loss)
                  F_loss = GAN.generator_loss(D_Fx_logits, heuristic=False) + cycle_loss + F_all_pos_loss 
              else:
                  print('standard loss generator')
                  G_loss = GAN.generator_loss(D_Gy_logits, heuristic=False) + cycle_loss
                  F_loss = GAN.generator_loss(D_Fx_logits, heuristic=False) + cycle_loss
      
          D_Y_loss = GAN.discriminator_loss(real_output= D_Y_logits, fake_output= D_Gy_logits)     
          D_X_loss = GAN.discriminator_loss(real_output= D_X_logits, fake_output= D_Fx_logits)
          
          # Gradient Computations
#           G_gradients = G_tape.gradient(G_loss, G.trainable_variables)
          G_gradients = tf.gradients(G_loss, G.trainable_variables)
          D_Y_gradients = tf.gradients(D_Y_loss, D_Y.trainable_variables)
          
          F_gradients = tf.gradients(F_loss, F.trainable_variables)
          D_X_gradients = tf.gradients(D_X_loss, D_X.trainable_variables)

          losses.extend([G_loss, F_loss, D_Y_loss, D_X_loss])
          gradients.extend([G_gradients, F_gradients, D_Y_gradients, D_X_gradients])
          
          trainers = GAN.optimize(gradients)

          # Building summaries
          tf.summary.histogram('D_Y/true', D_Y_logits)
          tf.summary.histogram('D_Y/fake', D_Gy_logits)
          tf.summary.histogram('D_X/true', D_X_logits)
          tf.summary.histogram('D_X/fake', D_Fx_logits)

          tf.summary.scalar('loss/G', G_loss)
          tf.summary.scalar('loss/D_Y', D_Y_loss)
          tf.summary.scalar('loss/F', F_loss)
          tf.summary.scalar('loss/D_X', D_X_loss)
          tf.summary.scalar('loss/cycle', cycle_loss)

          summary_op = tf.summary.merge_all()

     return losses, gradients, trainers, summary_op