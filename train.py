import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.utils
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.transforms.transforms import Resize
from tqdm import tqdm

from models import Discriminator, Generator, weights_init


def main():
    # Hyper parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z_dim = 256
    lr = 1e-4
    
    batch_size = 32
    num_epochs = 200

    n_channels = 3
    
    # Models
    discriminator = Discriminator(n_channels=n_channels).to(device)
    generator = Generator(z_dim=z_dim, n_channels=n_channels).to(device)

    # Initialize weights
    discriminator.apply(weights_init)
    generator.apply(weights_init)

    # Use multi-gpu

    # Pre-processing
    # data_mean = (0.4818, 0.4324, 0.3844)
    # data_std = (0.2601, 0.2518, 0.2538)
    data_mean = tuple(0.5 for _ in range(n_channels))
    data_std = tuple(0.5 for _ in range(n_channels))

    transform = transforms.Compose([
        transforms.Resize((64, 64), interpolation=Image.NEAREST),
        # transforms.CenterCrop((128, 128)),
        # transforms.Resize((64, 64), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        # transforms.Normalize(data_mean, data_std),
    ])

    # Data
    # trainset = dset.ImageFolder(root='cats', transform=transform)
    # trainset = dset.ImageFolder(root='dataset/celeba', transform=transform)
    trainset = datasets.EMNIST('dataset', transform=transform, download=True)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    opt_disc = torch.optim.Adam(discriminator.parameters(), lr=lr)
    opt_gen = torch.optim.Adam(generator.parameters(), lr=lr)
    
    bce_loss = nn.BCELoss()
    writer = SummaryWriter(f"runs/CELEBA")
    
    # Logging and Tensorboard 
    fixed_noise = torch.randn(batch_size * z_dim)
    fixed_noise = fixed_noise.to(device).view(batch_size, z_dim, 1, 1)
    epoch_iterator = tqdm(range(num_epochs))
    
    for epoch in epoch_iterator:
        batch_iterator = tqdm(trainloader, desc=f"Epoch {epoch}")
        for batch_idx, (real, _) in enumerate(batch_iterator):
            real = real.to(device)
            batch_size = real.shape[0]
            
            # Train descriminator: minimize the classification loss
            noise = torch.randn(batch_size * z_dim)
            noise = noise.view(batch_size, z_dim, 1, 1).to(device)
            fake = generator(noise)
            
            disc_real = discriminator(real).view(-1)
            disc_fake = discriminator(fake).view(-1)
            
            loss_real = bce_loss(disc_real, torch.ones_like(disc_real))
            loss_fake = bce_loss(disc_fake, torch.zeros_like(disc_fake))
            
            disc_loss = (loss_real + loss_fake) / 2
            
            discriminator.zero_grad()
            disc_loss.backward(retain_graph=True)
            opt_disc.step()
            
            # Train generator: maximize the classification loss for fake images
            output = discriminator(fake)
            gen_loss = bce_loss(output, torch.ones_like(output))
            generator.zero_grad()
            gen_loss.backward()
            opt_gen.step()
            
            # Tensorboard
            if batch_idx == 0:
                with torch.no_grad():
                    # Reshape to (B, C, H, W)
                    fake = generator(fixed_noise).reshape(-1, n_channels, 64, 64)
                    real = real.reshape(-1, n_channels, 64, 64)
                    
                    img_grid_real = torchvision.utils.make_grid(
                        real, 
                        normalize=True
                    )
                    
                    img_grid_fake = torchvision.utils.make_grid(
                        fake, 
                        normalize=True
                    )
                    
                    writer.add_scalar(
                        "Generator Loss", gen_loss, global_step=epoch
                    )
                    writer.add_scalar(
                        "Descriminator Loss", disc_loss, global_step=epoch
                    )
                    
                    writer.add_image(
                        "CELEBA Fake Image", img_grid_fake, global_step=epoch
                    )
                    
                    writer.add_image(
                        "CELEBA Real Image", img_grid_real, global_step=epoch
                    )
            batch_iterator.set_description(
                f"Epoch {epoch}: Loss D: {disc_loss:.4f}, G: {gen_loss:.4f}. Training"
            )
        epoch_iterator.set_description(
            f"Loss D: {disc_loss:.4f}, G: {gen_loss:.4f}. Training"
        )

if __name__ == "__main__":
    main()
