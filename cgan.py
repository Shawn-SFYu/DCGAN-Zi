import torch
from torch import nn


class EncoderBlock(nn.Module):
    """Encoder block"""
    def __init__(self, in_channel, out_channel, kernel_size=4, stride=2, padding=1, norm=True):
        super().__init__()
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
        
        self.bn=None
        if norm:
            self.bn = nn.BatchNorm2d(out_channel)
        
    def forward(self, x):
        fx = self.act(x)
        fx = self.conv(fx)
        
        if self.bn is not None:
            fx = self.bn(fx)
        return fx
    

class DecoderBlock(nn.Module):
    """Decoder block"""
    def __init__(self, in_channel, out_channel, kernel_size=4, stride=2, padding=1, dropout=False):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.deconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channel)       
        
        self.dropout=None
        if dropout:
            self.dropout = nn.Dropout2d(p=0.5, inplace=True)
            
    def forward(self, x):
        fx = self.act(x)
        fx = self.deconv(fx)
        fx = self.bn(fx)

        if self.dropout is not None:
            fx = self.dropout(fx)
            
        return fx


  
class UnetGenerator(nn.Module):
    """Unet-like Encoder-Decoder model"""
    def __init__(self, n_z, n_noise, feature, n_channel):
        super().__init__()
        self.n_z = n_z
        self.n_noise = n_noise
        self.encoder1 = nn.Conv2d(n_channel, feature, kernel_size=4, stride=2, padding=1) # size 32
        self.encoder2 = EncoderBlock(feature, feature*2) 
        self.encoder3 = EncoderBlock(feature*2, feature*4)
        self.encoder4 = EncoderBlock(feature*4, feature*8) # size 4
        self.encoder5 = EncoderBlock(feature*8, feature*8, kernel_size=2, stride=2, padding=0) # size 2
        self.encoder6 = EncoderBlock(feature*8, feature*8, kernel_size=2, stride=1, padding=0) # 1
        self.encoder7 = EncoderBlock(feature*8, n_z, kernel_size=1, stride=1, padding=0, norm=False)

        self.decoder7 = DecoderBlock(n_z+n_noise, feature*8, kernel_size=1, stride=1, padding=0, dropout=False) # 1
        self.decoder6 = DecoderBlock(2*feature*8, feature*8, kernel_size=2, stride=1, padding=0, dropout=False) # 2
        self.decoder5 = DecoderBlock(2*feature*8, feature*8, kernel_size=2, stride=2, padding=0) # 4
        self.decoder4 = DecoderBlock(2*feature*8, feature*4) # 8
        self.decoder3 = DecoderBlock(2*feature*4, feature*2) 
        self.decoder2 = DecoderBlock(2*feature*2, feature)
        self.decoder1 = nn.ConvTranspose2d(2*feature, n_channel, kernel_size=4, stride=2, padding=1) # 64
        self.apply(self._weights_init)

    def _weights_init(self, m):
        if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, 0.0, 0.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.trunc_normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

        
    def encoder(self, x):
        enc_dict = dict()
        e1 = self.encoder1(x)
        enc_dict['e1'] = e1
        e2 = self.encoder2(e1)
        enc_dict['e2'] = e2
        e3 = self.encoder3(e2)
        enc_dict['e3'] = e3
        e4 = self.encoder4(e3)
        enc_dict['e4'] = e4
        e5 = self.encoder5(e4)
        enc_dict['e5'] = e5
        e6 = self.encoder6(e5)
        enc_dict['e6'] = e6
        e7 = self.encoder7(e6)
        enc_dict['e7'] = e7

        return e7, enc_dict
    
    def decoder(self, z, enc_dict):
        d7 = self.decoder7(z)
        d7 = torch.cat([d7, enc_dict['e6']], dim=1)
        d6 = self.decoder6(d7)
        d6 = torch.cat([d6, enc_dict['e5']], dim=1)
        d5 = self.decoder5(d6)
        d5 = torch.cat([d5, enc_dict['e4']], dim=1)
        d4 = self.decoder4(d5)
        d4 = torch.cat([d4, enc_dict['e3']], dim=1)
        d3 = self.decoder3(d4)
        d3 = torch.cat([d3, enc_dict['e2']], dim=1)
        d2 = nn.ReLU(True)(self.decoder2(d3))
        d2 = torch.cat([d2, enc_dict['e1']], dim=1)
        d1 = self.decoder1(d2)
        return d1

 
    def forward(self, x):
        # encoder forward

        e7, enc_dict = self.encoder(x)
        # decoder forward + skip connections
        noise = torch.randn(e7.shape[0], self.n_noise, 1, 1)
        z = torch.concatenate((e7, noise), dim=1)
        y = self.decoder(z, enc_dict)
        
        return nn.Tanh()(y)


class AeGenerator(nn.Module):
    def __init__(self, n_z, n_noise, feature, n_channel):
        super().__init__()
        self.n_z = n_z
        self.n_noise = n_noise
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channel, feature, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature),
            nn.ReLU(True),
            nn.Conv2d(feature, feature*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature*2),
            nn.ReLU(True),
            nn.Conv2d(feature*2, feature*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature*4),
            nn.ReLU(True),
            nn.Conv2d(feature*4, feature*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature*8),
            nn.ReLU(True),
            nn.Conv2d(feature*8, n_z, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_z),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(n_z+n_noise, feature * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(feature * 8, feature * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(feature * 4, feature * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(feature * 2,     feature, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    feature,      n_channel, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        self.apply(self._weights_init)

    def _weights_init(self, m):
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.trunc_normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.trunc_normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

                                  
    def forward(self, input):
        z = self.encoder(input)
        noise = torch.randn(z.shape[0], self.n_noise, 1, 1)
        z = torch.concatenate((z, noise), dim=1)
        output = self.decoder(z)
        return output

 
if __name__ == '__main__':
    pass
