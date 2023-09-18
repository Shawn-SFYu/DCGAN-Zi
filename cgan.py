import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, n_z, n_noise, feature, n_channel):
        super(Generator, self).__init__()
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
            nn.Conv2d(feature*8, n_z, kernel_size=4, stride=1, padding=0, bias=False)


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
