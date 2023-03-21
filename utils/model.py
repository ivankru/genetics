import torch.nn as nn
import torch
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, vector_size, dim, hidden_dim=8):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.to_qkv = nn.Conv1d(dim, 3*hidden_dim, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)
        self.attention = nn.MultiheadAttention(vector_size, num_heads=8, batch_first=True)
        self.normalization = nn.LayerNorm(vector_size)

    def forward(self, x):
        x_norm = self.normalization(x)
        qkv = self.to_qkv(x_norm)
        q = qkv[:,:self.hidden_dim, :]
        k = qkv[:,self.hidden_dim:2*self.hidden_dim, :]
        v = qkv[:,2*self.hidden_dim:, :]
        h, _ = self.attention(q, k, v)
        h = self.to_out(h)
        h = h + x
        return h


class DownsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, dim, kernel_size=3):
        n_shortcut = int((n_inputs + n_outputs) // 2) + 1
        super(DownsamplingBlock, self).__init__()
        self.kernel_size = kernel_size
        # CONV 1
        self.pre_shortcut_convs = nn.Conv1d(n_inputs, n_shortcut, self.kernel_size, padding="same")# padding="same"
        self.shortcut_convs = nn.Conv1d(n_shortcut, n_shortcut, 1, padding="same")#padding="same"
        self.post_shortcut_convs = nn.Conv1d(n_shortcut, n_outputs, self.kernel_size, padding="same")#, padding="same"
        self.down = nn.Conv1d(n_outputs, n_outputs, 3, 2, padding=1)#nn.MaxPool1d(2)
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.res_conv = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else nn.Identity()
        self.attention = SelfAttention(dim, n_shortcut)

    def forward(self, x):
        # PREPARING SHORTCUT FEATURES
        shortcut = self.pre_shortcut_convs(x)
        shortcut = self.layer_norm1(shortcut)
        shortcut = F.mish(shortcut)
        shortcut = self.shortcut_convs(shortcut)
        shortcut = self.attention(shortcut)
        # shortcut = torch.cat([h, shortcut], dim=1)
        # shortcut = self.shortcut_convs(shortcut)
        # shortcut = F.mish(shortcut)
        # PREPARING FOR DOWNSAMPLING
        shortcut = self.post_shortcut_convs(shortcut)
        shortcut = self.layer_norm2(shortcut)
        h = F.mish(shortcut)
        h = h + self.res_conv(x)
        # DOWNSAMPLING
        out = self.down(h)
        return out


class DNAclassificator(nn.Module):
    def __init__(self):
        super(DNAclassificator, self).__init__()
        # Dropout definition
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.Mish()

        self.embedding_size = 4
        self.seq_len = 512
        self.embedding = nn.Embedding(4101, self.embedding_size)
        # Output size for each convolution
        self.out_size = 4
        # Number of strides for each convolution
        #self.stride = 2
        # CNN parameters definition
        # Kernel sizes
        self.kernel_1 = 1
        self.kernel_2 = 2
        self.kernel_3 = 3
        self.kernel_4 = 4

        # Convolution layers definition
        self.conv_1 = nn.Conv1d(self.embedding_size, self.out_size, self.kernel_1, padding="same")
        self.conv_2 = nn.Conv1d(self.embedding_size, self.out_size, self.kernel_2, padding="same")
        self.conv_3 = nn.Conv1d(self.embedding_size, self.out_size, self.kernel_3, padding="same")
        self.conv_4 = nn.Conv1d(self.embedding_size, self.out_size, self.kernel_4, padding="same")

        self.down1 = DownsamplingBlock(4*self.out_size, 8*self.out_size, 512)
        self.down2 = DownsamplingBlock(8*self.out_size, 8*self.out_size, 256)
        self.down3 = DownsamplingBlock(8*self.out_size, 4*self.out_size, 128)
        self.down4 = DownsamplingBlock(4*self.out_size, 4, 64)

        self.attention_full = SelfAttention(vector_size=self.seq_len // 2, dim=2)

        self.conv_2_1 = nn.Conv1d(4, 4, 3, padding="same")
        self.conv_2_2 = nn.Conv1d(4, 2, 1)

        self.fc1 = nn.Linear(32*2, 64)
        self.fc2 = nn.Linear(64, 2)

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
      x = self.down1(union)
      x = self.down2(x)
      x = self.down3(x)
      x = self.down4(x)
      x = self.activation(x)
      x = self.conv_2_1(x)
      x = self.activation(x)
      x = self.conv_2_2(x)
      x = self.activation(x)

      union = x.reshape(x.size(0), -1)

      union = self.dropout(union)
      union = self.fc1(union)
      union = self.activation(union)
      union = self.dropout(union)
      union = self.fc2(union)
      return union


class DNAdescriminator(nn.Module):
    def __init__(self):
        super(DNAdescriminator, self).__init__()
        # Dropout definition
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.Mish()

        self.embedding_size = 4
        self.seq_len = 512
        self.cond_embedding = ConditionalEmbedding(128)
        self.layer_norm = nn.LayerNorm(self.seq_len)
        # Output size for each convolution
        self.out_size = 4
        # Number of strides for each convolution
        #self.stride = 2
        # CNN parameters definition
        # Kernel sizes
        self.kernel_1 = 1
        self.kernel_2 = 2
        self.kernel_3 = 3
        self.kernel_4 = 4

        # Convolution layers definition
        self.conv_1 = nn.Conv1d(self.embedding_size, self.out_size, self.kernel_1, padding="same")
        self.conv_2 = nn.Conv1d(self.embedding_size, self.out_size, self.kernel_2, padding="same")
        self.conv_3 = nn.Conv1d(self.embedding_size, self.out_size, self.kernel_3, padding="same")
        self.conv_4 = nn.Conv1d(self.embedding_size, self.out_size, self.kernel_4, padding="same")

        self.down1 = DownsamplingBlock(4*self.out_size, 8*self.out_size, 512)
        self.down2 = DownsamplingBlock(8*self.out_size, 8*self.out_size, 256)
        self.down3 = DownsamplingBlock(8*self.out_size, 4*self.out_size, 128)
        self.down4 = DownsamplingBlock(4*self.out_size, 4, 64)

        self.attention_full = SelfAttention(vector_size=self.seq_len // 2, dim=2)

        self.conv_2_1 = nn.Conv1d(4, 4, 3, padding="same")
        self.conv_2_2 = nn.Conv1d(4, 2, 1)

        self.fc1 = nn.Linear(32*2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x, label):
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
      union = self.layer_norm(union)
      x = self.down1(union)
      x = self.down2(x)
      x = self.down3(x)
      x = self.down4(x)

      cond_embedding = self.cond_embedding(label)
      cond_embedding = cond_embedding.reshape(x.shape)
      x = x + cond_embedding

      x = self.activation(x)
      x = self.conv_2_1(x)
      x = self.activation(x)
      x = self.conv_2_2(x)
      x = self.activation(x)

      union = x.reshape(x.size(0), -1)

      #union = self.dropout(union)
      union = self.fc1(union)
      union = self.activation(union)
      #union = self.dropout(union)
      union = self.fc2(union)
      return union


class ConditionalEmbedding(nn.Module):
    def __init__(self, dim):
        super(ConditionalEmbedding, self).__init__()
        self.fc1 = nn.Linear(2, 32)
        self.layer_norm1 = nn.LayerNorm(32)
        self.fc2 = nn.Linear(32, dim)
        self.layer_norm2 = nn.LayerNorm(dim)

    def const(self, cond):
        cond = cond.repeat(2, 1).transpose(1,0)
        device = cond.device
        cond = cond.int()
        cond1 = torch.FloatTensor([-0.5, 0.5]).to(device)
        cond1 = cond1[None,:].repeat(cond.shape[0],1)
        cond0 = torch.FloatTensor([0.5,-0.5]).to(device)
        cond0 = cond0[None,:].repeat(cond.shape[0],1)
        const_tensor = torch.zeros_like(cond1).to(device)
        nonzer = torch.nonzero(cond)
        const_tensor[cond==0] = cond0[cond==0]
        const_tensor[cond==1] = cond1[cond==1]
        return const_tensor
      
    def forward(self, cond):
        x = self.const(cond).to(cond.device)
        x = self.fc1(x)
        x = self.layer_norm1(x)
        x = F.mish(x)
        x = self.fc2(x)
        #x = self.layer_norm2(x)
        return x


class PixelShuffle1D(torch.nn.Module):
    """
    1D pixel shuffler. https://arxiv.org/pdf/1609.05158.pdf
    Upscales sample length, downscales channel length
    "short" is input, "long" is output
    """
    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)

        return x


class UpsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, dim, kernel_size=5, up_dim=None):
        n_shortcut = int((n_inputs + n_outputs) // 2)
        super(UpsamplingBlock, self).__init__()
        self.kernel_size = kernel_size

        self.pixel_shuffle = PixelShuffle1D(2)
        # CONV 1
        self.pre_shortcut_convs = nn.Conv1d(n_inputs, n_shortcut, self.kernel_size, padding="same")#padding="same"
        self.shortcut_convs = nn.Conv1d(n_shortcut, n_shortcut, self.kernel_size, padding="same")
        self.post_shortcut_convs = nn.Conv1d(n_shortcut, n_outputs, self.kernel_size, padding="same")#padding="same"
        #self.up = nn.Upsample(scale_factor=2)
        if up_dim is None:
            self.up = nn.ConvTranspose1d(n_inputs, n_inputs, kernel_size=4, stride=2, padding=1)#padding=1
        else:
            self.up = nn.ConvTranspose1d(up_dim, up_dim, kernel_size=4, stride=2, padding=1)
        self.layer_norm1 = nn.LayerNorm(2*dim)
        self.layer_norm2 = nn.LayerNorm(2*dim)
        self.res_conv = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else nn.Identity()
        self.attention = SelfAttention(2*dim, n_shortcut)
        self.cond_emb = ConditionalEmbedding(2*dim)

    def forward(self, x, t):
        initial_x = self.pixel_shuffle(x)
        x = self.up(x) 
        #t = self.cond_emb(t)[:,None,:].repeat(1, x.shape[1], 1)
        #x = x + t
        shortcut = x
        # PREPARING SHORTCUT FEATURES
        shortcut = self.pre_shortcut_convs(shortcut)
        shortcut = self.layer_norm1(shortcut)
        shortcut = F.mish(shortcut)
        shortcut = self.shortcut_convs(shortcut)
        shortcut = self.attention(shortcut)
        # PREPARING FOR UPSAMPLING
        out = self.post_shortcut_convs(shortcut)
        out = self.layer_norm2(out)
        out = F.mish(out)
        out = out + initial_x
        return out


class DNAgenerator(nn.Module):
    def __init__(self, z_dim, output_size=512, softmax=True):
        super(DNAgenerator, self).__init__()
        self.softmax = True
        intermediate_level_size = 1024
        self.dim_levels = [64, 128, 256, 512]
        self.channel_levels = [64, 32, 16, 8]
        self.output_channels = 4

        self.init_layer = nn.Sequential(nn.Linear(z_dim, intermediate_level_size), \
            nn.Mish(), nn.Linear(intermediate_level_size, self.channel_levels[0]*self.dim_levels[0]), \
            nn.Mish())
        self.cond_emb = ConditionalEmbedding(self.channel_levels[0])
        self.init_conv = nn.Sequential(
                        nn.Conv1d(self.channel_levels[0], self.channel_levels[0], 3, padding="same"),
                        nn.LayerNorm(self.dim_levels[0]), nn.Mish(),
                        nn.Conv1d(self.channel_levels[0], self.channel_levels[0], 1, padding="same"),
                        nn.LayerNorm(self.dim_levels[0]), nn.Mish())
        self.up1 = UpsamplingBlock(self.channel_levels[0], self.channel_levels[1], self.dim_levels[0])
        self.up2 = UpsamplingBlock(self.channel_levels[1], self.channel_levels[2], self.dim_levels[1])
        self.up3 = UpsamplingBlock(self.channel_levels[2], self.channel_levels[3], self.dim_levels[2])
        self.res_conv = nn.Sequential(
                        nn.Conv1d(self.channel_levels[3], self.output_channels, 3, padding="same"),
                        nn.Mish(),
                        nn.Conv1d(self.output_channels, self.output_channels, 1, bias=False))

    def forward(self, z, t):
        x = self.init_layer(z)
        x = torch.reshape(x, [-1, self.channel_levels[0], self.dim_levels[0]])
        condition = self.cond_emb(t)[:,None,:].repeat(1, x.shape[1], 1)
        x = x + condition
        x = self.init_conv(x)

        x = self.up1(x, t)
        x = self.up2(x, t)
        x = self.up3(x, t)
        x = self.res_conv(x)
        if self.softmax:
            x = torch.softmax(x, dim=1)
        return x


class DNAgenerator2(nn.Module):
    def __init__(self, n_latent):
        super(DNAgenerator2, self).__init__()
        self.activation = nn.LeakyReLU()
        self.init_fc = nn.Sequential(
                    nn.Linear(n_latent, 2048),
                    nn.BatchNorm1d(2048),
                    self.activation,
                    nn.Linear(2048, 4*512),
                    nn.BatchNorm1d(4*512),
                    self.activation,
                    )
        self.cond_embedding = ConditionalEmbedding(2048)
        self.conv_init = nn.Sequential(
                         nn.Conv1d(32,32,3, padding="same"),
                         self.activation,
                         nn.Conv1d(32,32,3, padding="same"),
                         nn.LayerNorm(64),
                         self.activation
                        )
        self.pixel_shuffle1 = PixelShuffle1D(4)
        self.conv_up1 = nn.Sequential(
                         nn.Conv1d(8,32,5, padding="same"),
                         self.activation,
                         nn.Conv1d(32,8,5, padding="same"),
                         nn.LayerNorm(256),
                         self.activation
                        )
        self.pixel_shuffle2 = PixelShuffle1D(2)                
        self.conv_up2 = nn.Sequential(
                         nn.Conv1d(4,32,5, padding="same"),
                         self.activation,
                         nn.Conv1d(32,4,5, padding="same"),
                         nn.LayerNorm(512),
                         self.activation
                        )
        self.res_conv = nn.Sequential(nn.Conv1d(4,4,3,padding="same"),
                                      self.activation,
                                      nn.Conv1d(4,4,1, bias=False))
    
    def forward(self, x, label):
        assert x.shape[0] == label.shape[0]
        cond_embd = self.cond_embedding(label)
        x = self.init_fc(x)
        x = x + 0.1*cond_embd
        x = torch.reshape(x, [-1, 32, 64])
        x = self.conv_init(x) + x
        x = self.pixel_shuffle1(x)
        x = self.conv_up1(x) + x
        x = self.pixel_shuffle2(x)
        x = self.conv_up2(x) + x
        x = self.res_conv(x)
        x = torch.softmax(x, dim=1)
        return x 
