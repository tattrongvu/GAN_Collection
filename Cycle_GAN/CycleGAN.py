import sys, os
sys.path.append(os.getcwd())
#sys.path.append("/notebooks/dev1/robotic_gym")

import numpy as np
import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from datetime import datetime

from models import *
from pre_process import *
from helpers import *

from tensorboardX import SummaryWriter

class CycleGAN:
    def __init__(self, args, device):
        self.device = device
        self.args = args

        #Init Generator
        self.G_XtoY = CycleGenerator(self.args.g_conv_dim,self.args.g_num_res_blocks).to(self.device)
        self.G_YtoX = CycleGenerator(self.args.g_conv_dim,self.args.g_num_res_blocks).to(self.device)

        # Instantiate discriminators
        self.D_X = Discriminator(self.args.d_conv_dim).to(self.device)
        self.D_Y = Discriminator(self.args.d_conv_dim).to(self.device)

        #Optimizer
        self.g_params = list(self.G_XtoY.parameters()) + list(self.G_YtoX.parameters())  # Get generator parameters

        self.g_optimizer = optim.Adam(self.g_params, self.args.g_lr, [self.args.beta1, self.args.beta2])
        self.d_x_optimizer = optim.Adam(self.D_X.parameters(), self.args.d_lr, [self.args.beta1, self.args.beta2])
        self.d_y_optimizer = optim.Adam(self.D_Y.parameters(), self.args.d_lr, [self.args.beta1, self.args.beta2])


        #Dataset
        self.dataloader_X, self.test_dataloader_X = get_data_loader(image_type=self.args.domain_X, image_dir=self.args.datasets_dir, \
            image_size=self.args.image_size, batch_size=self.args.batch_size)

        self.dataloader_Y, self.test_dataloader_Y = get_data_loader(image_type=self.args.domain_Y, image_dir=self.args.datasets_dir, \
            image_size=self.args.image_size, batch_size=self.args.batch_size)

        
        print("Dataset {} contains: {} samples".format(self.args.domain_X, len(self.dataloader_X.dataset)))
        print("Dataset {} contains: {} samples".format(self.args.domain_Y, len(self.dataloader_Y.dataset)))


        self.writer = SummaryWriter(logdir=self.args.log_dir+self.args.dataset_name+"/"+self.args.model_name)
        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
            # path to save the model
        self.model_path = os.path.join(self.args.save_dir, self.args.model_name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        
        if not os.path.exists(self.args.samples_dir):
            os.mkdir(self.args.samples_dir)


    def learn(self):

        test_iter_X = iter(self.test_dataloader_X)
        test_iter_Y = iter(self.test_dataloader_Y)

        # Get some fixed data from domains X and Y for sampling. These are images that are held
        # constant throughout training, that allow us to inspect the model's performance.
        fixed_X = test_iter_X.next()[0]
        fixed_Y = test_iter_Y.next()[0]
        fixed_X = scale(fixed_X) # make sure to scale to a range -1 to 1
        fixed_Y = scale(fixed_Y)

        # batches per epoch
        iter_X = iter(self.dataloader_X)
        iter_Y = iter(self.dataloader_Y)

        batches_per_epoch = min(len(iter_X), len(iter_Y))
        print("Batches per epoch:",batches_per_epoch)

        for epoch in range(1, self.args.n_epochs+1):

            # Reset iterators for each epoch
            if epoch % batches_per_epoch == 0:
                iter_X = iter(self.dataloader_X)
                iter_Y = iter(self.dataloader_Y)

            d_x_loss, d_y_loss, g_total_loss = self.update(iter_X, iter_Y)

            # Print the log info
            if epoch % 10 == 0:
                print('[{}] Epoch [{:5d}/{:5d}] | d_X_loss: {:6.4f} | d_Y_loss: {:6.4f} | g_total_loss: {:6.4f}'.format(
                        datetime.now(), epoch, self.args.n_epochs, d_x_loss, d_y_loss, g_total_loss), end="\r")

            
            # Save the generated samples
            if epoch % self.args.sample_interval == 0:
                self.G_YtoX.eval() # set generators to eval mode for sample generation
                self.G_XtoY.eval()
                save_samples(epoch, fixed_Y, fixed_X, self.G_YtoX, self.G_XtoY, batch_size=self.args.batch_size, sample_dir=self.args.sample_dir)
                self.G_YtoX.train()
                self.G_XtoY.train()
                view_samples(epoch, sample_dir=self.args.sample_dir)

            self.writer.add_scalar("Loss/dx_loss", d_x_loss, epoch)
            self.writer.add_scalar("Loss/dy_loss", d_y_loss, epoch)
            self.writer.add_scalar("Loss/g_total_loss", g_total_loss, epoch)

            # Save the model parameters
            if epoch % self.args.save_interval == 0:
                checkpoint(epoch, self.G_XtoY, self.G_YtoX, self.D_X, self.D_Y, checkpoint_dir=self.args.save_dir)

    def update(self, iter_X, iter_Y):
        # Reset iterators for each epoch
    
        images_X, _ = iter_X.next()
        images_X = scale(images_X) # make sure to scale to a range -1 to 1

        images_Y, _ = iter_Y.next()
        images_Y = scale(images_Y)
        
        # move images to GPU if available (otherwise stay on CPU)
        images_X = images_X.to(self.device)
        images_Y = images_Y.to(self.device)


        # ============================================
        #            TRAIN THE DISCRIMINATORS
        # ============================================

        ##   First: D_X, real and fake loss components   ##

        # 1. Compute the discriminator losses on real images
        self.d_x_optimizer.zero_grad()
        real_D_loss = real_mse_loss(self.D_X(images_X))
        # 3. Compute the fake loss for D_X
        fake_D_loss = fake_mse_loss(self.D_X(self.G_YtoX(images_Y)))
        # 4. Compute the total loss and perform backprop
        d_x_loss = real_D_loss + fake_D_loss
        d_x_loss.backward()
        self.d_x_optimizer.step()
        
        ##   Second: D_Y, real and fake loss components   ##
        self.d_y_optimizer.zero_grad()
        real_D_y_loss = real_mse_loss(self.D_Y(images_Y))
        
        fake_D_y_loss = fake_mse_loss(self.D_Y(self.G_XtoY(images_X)))
        
        d_y_loss = real_D_y_loss + fake_D_y_loss
        d_y_loss.backward()
        self.d_y_optimizer.step()


        # =========================================
        #            TRAIN THE GENERATORS
        # =========================================

        ##    First: generate fake X images and reconstructed Y images    ##
        self.g_optimizer.zero_grad()
        # 1. Generate fake images that look like domain X based on real images in domain Y
        out_1 = self.G_YtoX(images_Y)
        # 2. Compute the generator loss based on domain X
        loss_1 = real_mse_loss(self.D_X(out_1))
        # 3. Create a reconstructed y
        out_2 = self.G_XtoY(out_1)
        # 4. Compute the cycle consistency loss (the reconstruction loss)
        loss_2 = cycle_consistency_loss(real_im = images_Y, reconstructed_im = out_2, lambda_weight=self.args.lambda_weight)

        ##    Second: generate fake Y images and reconstructed X images    ##
        out_3 = self.G_XtoY(images_X)
        # 5. Add up all generator and reconstructed losses and perform backprop
        loss_3 = real_mse_loss(self.D_Y(out_3))
        out_4 = self.G_YtoX(out_3)
        loss_4 =  cycle_consistency_loss(real_im = images_X, reconstructed_im = out_4, lambda_weight=self.args.lambda_weight)

        g_total_loss = loss_1 + loss_2 + loss_3 + loss_4
        g_total_loss.backward()
        self.g_optimizer.step()
        

        return [d_x_loss.item(), d_y_loss.item(), g_total_loss.item()]