import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.utils import save_image

class AnimeDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
    


def save_checkpoint(model_G, model_D, 
                    optimizer_G, optimizer_D, 
                    loss_G, loss_D, 
                    checkpoint_dir,
                    filename="checkpoint.pth.tar"):
    state = {
        'model_G_state_dict': model_G.state_dict(),
        'model_D_state_dict': model_D.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'loss_G': loss_G,
        'loss_D': loss_D,
    }
    torch.save(state, os.path.join(checkpoint_dir, filename))

def load_checkpoint(model_G, model_D, 
                    optimizer_G, optimizer_D,
                    checkpoint_dir,
                    filename="checkpoint.pth.tar"):
    checkpoint = torch.load(os.path.join(checkpoint_dir, filename))
    model_G.load_state_dict(checkpoint['model_G_state_dict'])
    model_D.load_state_dict(checkpoint['model_D_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    loss_G = checkpoint['loss_G']
    loss_D = checkpoint['loss_D']
    return loss_G, loss_D


def compute_gradient_penalty(D, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size(), requires_grad=False, device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train(netD, netG, dataloader, nz,
            device, optimizerD, optimizerG, 
            num_epochs, imgs_dir, n_critic=5):
    lambda_gp = 10
    disc_loss = []
    gene_loss = []

    # Create batch of latent vectors to visualize progress
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    for epoch in range(num_epochs):
        for _, data in enumerate(dataloader, 0):
            # train discriminator more frequently
            for _ in range(n_critic):
                netD.zero_grad()
                real_cpu = data.to(device)
                b_size = real_cpu.size(0)

                # Loss of discriminator
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                fake = netG(noise)

                errD_real = -torch.mean(netD(real_cpu))
                errD_fake = torch.mean(netD(fake.detach()))

                gradient_penalty = compute_gradient_penalty(netD, real_cpu, fake, device=device)
                errD = errD_real + errD_fake + lambda_gp * gradient_penalty
                errD.backward()
                optimizerD.step()

            # Update Generator
            netG.zero_grad()
            noise = torch.randn(b_size, nz, 1, 1, device=device)  # Re-generate noise
            fake = netG(noise)
            errG = -torch.mean(netD(fake)) # Using discriminator result to update generator
            errG.backward()

            optimizerG.step()

        # Loss record and print
        print(f'[{epoch}/{num_epochs}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}')
        disc_loss.append(errD.item())
        gene_loss.append(errG.item())

        # Save generated samples and checkpoints
        if epoch % 500 == 0:
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            save_image(fake, f'{imgs_dir}/128_fixed_samples_epoch_{epoch}.png', normalize=True)
            save_checkpoint(netG, netD, optimizerG, optimizerD, errG.item(), errD.item(), f"128_checkpoint_epoch_{epoch}.pth.tar")

    # Final Save image and checkpoint
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
    save_image(fake, f'{imgs_dir}/fake_samples_final.png', normalize=True)
    save_checkpoint(netG, netD, optimizerG, optimizerD, errG.item(), errD.item(), f"128_checkpoint_final.pth.tar")
    return disc_loss, gene_loss