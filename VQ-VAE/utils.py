from torch.utils.data import Dataset, random_split
import os
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch

class AnimeImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')  # Convert image to RGB
        if self.transform:
            image = self.transform(image)
        return image


def train_test_split(dataset, test_ratio):
    test_size = int(len(dataset) * test_ratio)
    train_size = len(dataset) - test_size
    return random_split(dataset, [train_size, test_size])


def train(model, 
          dataloader, 
          latent_dim, 
          epochs=40, 
          device='cuda', 
          lr=0.0001,
          commitment_beta=0.25):
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    rec_loss = []
    cb_loss = []
    total_loss = []
    for epoch in range(epochs):
        for _, data in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            x_hat, commitment_loss, codebook_loss = model(data)
            recon_loss = mse_loss(x_hat, data)
            loss =  recon_loss + commitment_loss * commitment_beta + codebook_loss
            loss.backward()
            optimizer.step()

        rec_loss.append(recon_loss.item())
        cb_loss.append(codebook_loss.item())
        total_loss.append(loss.item())

        print(f'Epoch {epoch + 1}, Rec Loss: {recon_loss.item():.4f}, Codebook Loss: {codebook_loss.item():.4f}, Total Loss: {loss.item():.4f}')
    torch.save(model.state_dict(), f"./vae_anime_{latent_dim}_factor.pt")
    return total_loss, rec_loss, cb_loss


def load_checkpoint(model, model_path):
    model.load_state_dict(torch.load(model_path))