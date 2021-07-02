import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, 
                 bn: bool = False) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_features, out_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_features) if bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class TransConvBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 stride: int = 2, padding: int = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_features, out_features, 4,
                               stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class Discriminator(nn.Module):
    def __init__(self, n_channels: int,
                 ndf: int = 64,
                 img_size: int = 64) -> None:
        super().__init__()
        self.img_size = img_size
        self.layers = nn.Sequential(
            ConvBlock(n_channels, ndf, bn=False),
            ConvBlock(ndf, ndf * 2),
            ConvBlock(ndf * 2, ndf * 4),
            ConvBlock(ndf * 4, ndf * 8),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Generator(nn.Module):
    def __init__(self, z_dim: int,
                 n_channels: int,
                 ngf: int = 64,
                 img_size: int = 64) -> None:
        super(Generator, self).__init__()
        self.img_size = img_size
        self.layers = nn.Sequential(
            TransConvBlock(z_dim, ngf * 8, stride=1, padding=0),  # 4 x 4
            TransConvBlock(ngf * 8, ngf * 4),  # 8 x 8
            TransConvBlock(ngf * 4, ngf * 2),  # 16 x 16
            TransConvBlock(ngf * 2, ngf * 1),  # 32 x 32
            nn.ConvTranspose2d(ngf, n_channels, 4, 2, 1,
                               bias=False),  # 64 x 64
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ConditionalDiscriminator(Discriminator):
    def __init__(self, n_channels: int,
                 n_classes: int,
                 ndf: int = 64,
                 img_size: int = 64) -> None:
        super().__init__(n_channels + 1, ndf=ndf, img_size=img_size)
        self.embed = nn.Embedding(n_classes, self.img_size ** 2)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = torch.cat([
            x,
            self.embed(labels).view(-1, 1, self.img_size, self.img_size)
        ], dim=1)
        return super().forward(x)


class ConditionalGenerator(Generator):
    def __init__(self,
                 z_dim: int,
                 n_channels: int,
                 n_classes: int,
                 ngf: int = 32) -> None:

        super().__init__(z_dim * 2, n_channels, ngf=ngf)
        self.embed = nn.Embedding(n_classes, z_dim)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        cat = torch.cat([
            x,
            embedding
        ], dim=1)

        return super().forward(cat)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Block') == -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
