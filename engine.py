import os
import math
import torch
from torch import nn
import torchvision.utils as vision_utils


from typing import Iterable

REAL_LABEL = 1.0
FAKE_LABEL = 0.0


class GenEvalLogger():
    def __init__(self, device, latent_size, img_num=64, dir="./gen_eval", dump=True) -> None:
        self.fixed_noise = torch.randn(img_num, latent_size, 1, 1).to(device)
        self.img_list = []
        self.dump = dump
        self.dir = dir
        os.makedirs(self.dir, exist_ok=True)
        

    def evalute_on_fixed_noise(self, generator: nn.Module, epoch: int, num: int):
        with torch.no_grad():
            fake_img = generator(self.fixed_noise).detach().cpu()
        if not self.dump:
            self.img_list.append(vision_utils.make_grid(fake_img, padding=2, normalize=True))
        else:
            filepath = os.path.join(self.dir, f"epoch{epoch}-num{num}.png")
            vision_utils.save_image(fake_img, fp=filepath, padding=2, normalize=True)


def train_one_epoch(
        generator: nn.Module,
        discriminator: nn.Module,
        latent_size: int,
        g_optim: torch.optim.Optimizer,
        d_optim: torch.optim.Optimizer,
        criterion: nn.Module,
        dataloader: Iterable,
        device: torch.device,
        epoch: int,
        metric_logger,
        print_freq: int,
        gen_eval_logger,
        gen_eval_freq: int
):
    for i, data in enumerate(dataloader):
        
        # Update D network: maximize log(D(x)) + log(1 - D(G(z)))

        ## Train with all-real batch
        discriminator.train(True)
        d_optim.zero_grad()
        # Format batch
        real_image = data[0].to(device)   # get first image in the batch
        b_size = real_image.size(0)
        label = torch.full((b_size,), REAL_LABEL, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = discriminator(real_image).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, latent_size, 1, 1, device=device)
        # Generate fake image batch with G
        fake_image = generator(noise)
        label.fill_(FAKE_LABEL)
        # Classify all fake batch with D
        output = discriminator(fake_image.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        d_optim.step()

        # Update G network: maximize log(D(G(z)))

        g_optim.zero_grad()
        label.fill_(REAL_LABEL)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = discriminator(fake_image).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        g_optim.step()
        
        # Output training stats
        if i % print_freq == 0:
            print('Epoch %d: [%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        
        if metric_logger is not None:
            metric_logger.update(loss=errG.item(), head="G-loss")
            metric_logger.update(loss=errD.item(), head="D-loss")
            metric_logger.set_step()            


        
        # Check how the generator is doing by saving G's output on fixed_noise
        if (i % gen_eval_freq == 0) or (i == len(dataloader)-1) :
            gen_eval_logger.evalute_on_fixed_noise(generator, epoch, num=math.floor(i // gen_eval_freq))
            



            
            

