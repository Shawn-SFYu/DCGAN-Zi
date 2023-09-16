from torch import nn

class Generator(nn.Module):
    def __init__(self, n_z, feature, n_channel):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(n_z, feature * 8, 4, 1, 0, bias=False),
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
        self.apply(self.weights_init)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.trunc_normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.trunc_normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        
    def forward(self, input):

        output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, n_z, feature, n_channel):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(n_channel, feature, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(feature, feature * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(feature * 2, feature * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(feature * 4, feature * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(feature * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.apply(self.weights_init)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.trunc_normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.trunc_normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, input):

        output = self.main(input)
        return output.view(-1, 1).squeeze(1)
    
if __name__ == '__main__':
    pass
