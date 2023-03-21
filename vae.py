import torch.nn as nn
import torch

import sys
sys.path.insert(0, '/home/kruzhilov/genetics')
from utils.model import DownsamplingBlock, DNAgenerator
from train import import_data
from utils.train_val import input_part
from utils.datasets import Dataset, DNATokenizer
from utils.model import PixelShuffle1D

class DNAencoder(nn.Module):
    def __init__(self, latent_size):
        super(DNAencoder, self).__init__()
        # Dropout definition
        self.activation = nn.Mish()
        self.seq_len = 512
        # Output size for each convolution
        self.out_size = 4
        self.embedding_size = 4
        self.normalization = nn.BatchNorm1d(4*self.out_size)
        # Number of strides for each convolution
        #self.stride = 2
        # CNN parameters definition
        # Kernel sizes
        self.kernel_1 = 2
        self.kernel_2 = 3
        self.kernel_3 = 4
        self.kernel_4 = 5

        # Convolution layers definition
        self.conv_1 = nn.Conv1d(self.embedding_size, self.out_size, self.kernel_1, padding="same")
        self.conv_2 = nn.Conv1d(self.embedding_size, self.out_size, self.kernel_2, padding="same")
        self.conv_3 = nn.Conv1d(self.embedding_size, self.out_size, self.kernel_3, padding="same")
        self.conv_4 = nn.Conv1d(self.embedding_size, self.out_size, self.kernel_4, padding="same")

        self.down1 = DownsamplingBlock(4*self.out_size, 8*self.out_size, 512)
        self.down2 = DownsamplingBlock(8*self.out_size, 4*self.out_size, 256)
        self.down3 = DownsamplingBlock(4*self.out_size, 2*self.out_size, 128)
        self.down4 = DownsamplingBlock(2*self.out_size, 4, 64)

        self.fc1 = nn.Linear(64*2, 64*2)
        self.fc2 = nn.Linear(64*2, latent_size)
        self.normalization_fc = nn.BatchNorm1d(128)

    def forward(self, x):
      #x = self.embedding(x)
      #x = torch.transpose(x, 1, 2) 
      # Convolution layer 1 is applied
      x1 = self.conv_1(x)
      x1 = self.activation(x1)
      
      # Convolution layer 2 is applied
      x2 = self.conv_2(x)
      x2 = self.activation(x2)
   
      # Convolution layer 3 is applied
      x3 = self.conv_3(x)
      x3 = self.activation(x3)
      
      # Convolution layer 4 is applied
      x4 = self.conv_4(x)
      x4 = self.activation(x4)
      
      union = torch.cat((x1, x2, x3, x4), 1)
      union = self.normalization(union)
      x = self.down1(union)
      x = self.down2(x)
      x = self.down3(x)
      x = self.down4(x)
      
      union = torch.flatten(x, start_dim=1)
      union = self.fc1(union)
      union = self.normalization_fc(union)
      union = self.activation(union)
    #   union = self.fc2(union)
      return union


class DNAdecoder(nn.Module):
    def __init__(self, n_latent):
        super(DNAdecoder, self).__init__()
        self.activation = nn.LeakyReLU()
        self.init_fc = nn.Sequential(
                    nn.Linear(n_latent, 2048),
                    nn.BatchNorm1d(2048),
                    self.activation,
                    nn.Linear(2048, 4*512),
                    nn.BatchNorm1d(4*512),
                    self.activation,
                    )
        self.conv_init = nn.Sequential(
                         nn.Conv1d(16,32,5, padding="same"),
                         self.activation,
                         nn.Conv1d(32,16,5, padding="same"),
                         nn.LayerNorm(128),
                         self.activation
                        )
        self.pixel_shuffle = PixelShuffle1D(4)
        self.conv_up1 = nn.Sequential(
                         nn.Conv1d(4,32,5, padding="same"),
                         self.activation,
                         nn.Conv1d(32,4,5, padding="same"),
                         nn.LayerNorm(512),
                         self.activation
                        )
        self.res_conv = nn.Conv1d(4,4,1)
    
    def forward(self, x):
        x = self.init_fc(x)
        x = torch.reshape(x, [-1, 16, 128])
        x = self.conv_init(x) + x
        x = self.pixel_shuffle(x)
        x = self.conv_up1(x) + x
        self.res_conv(x)
        return x 





class DNA_VAE(nn.Module):
    def __init__(self, latent_size):
        super(DNA_VAE, self).__init__()
        self.encoder = DNAencoder(latent_size)
        self.decoder = DNAdecoder(latent_size)
        self.mu = nn.Linear(64*2, latent_size)
        self.sigma_log = nn.Linear(64*2, latent_size)

    def kl_divergence(self, mu, std):
        kl = torch.square(mu) + torch.square(std) - 2*torch.log(std + 10**-8) - 1
        kl = kl.mean()
        return kl

    def forward(self, x, label):
        latent = self.encoder(x)
        mu = self.mu(latent)
        sigma_log = self.sigma_log(latent)
        sigma = torch.exp(sigma_log)
        z = mu + sigma*torch.randn_like(mu)
        x_rec = self.decoder(z)
        return mu, sigma, x_rec


if __name__ == "__main__":
    device = "cuda:1"
    epoch_number = 150
    batch_size = 45

    MODEL_NUMBER = 0
    train_loader, test_loader = import_data(MODEL_NUMBER, batch_size)

    vae = DNA_VAE(64)
    vae.to(device)
    optim = torch.optim.Adam(vae.parameters(), lr=0.0001, betas=(0.5, 0.99))
    rec_loss_func = nn.BCEWithLogitsLoss()

    kl_list = []
    rec_list = []

    for epoch in range(epoch_number):
        for idx, batch in enumerate(train_loader):
            input, labels = input_part(batch, kmer=False, device=device)
            label = torch.zeros(1).to(device)
            mu, sigma, x_rec = vae(input, label)
            rec_loss = rec_loss_func(x_rec, input)
            kl_div = vae.kl_divergence(mu, sigma)
            rec_list.append(rec_loss.item())
            kl_list.append(kl_div.item())
            optim.zero_grad()
            loss = rec_loss + 0.001*kl_div
            loss.backward()
            optim.step()
            
            if idx % 100 == 0:
                rec = sum(rec_list) / len(rec_list)
                kl = sum(kl_list) / len(kl_list)
                formated_str = "{epoch} rec:{rec:1.3f}, KL:{kl:1.3f}".format(epoch=epoch, rec=rec, kl=kl)
                print(formated_str)
                kl_list = []
                rec_list = []
        print(torch.softmax(x_rec, dim=1))
