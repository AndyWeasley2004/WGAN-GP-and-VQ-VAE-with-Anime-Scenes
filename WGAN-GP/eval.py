import pickle
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
from torchvision.utils import save_image
from torchvision import models
from scipy.stats import entropy
import torch
import torch.nn as nn
import os

inception_model = models.inception_v3(pretrained=True, transform_input=False)
inception_model = inception_model.eval()

def save_loss(lossD, lossG, loss_dir, signature=''):
    data = {'lossD': lossD, 'lossG': lossG}

    with open(f'{loss_dir}/{signature}_loss.pkl', 'wb') as file:
        pickle.dump(data, file)


def load_loss(loss_dir, signature):
    with open(f'{loss_dir}/{signature}_loss.pkl', 'rb') as file:
        data = pickle.load(file)

    return data['lossD'], data['lossG']


def update_loss(lossD, lossG, loss_dir, signature='', new_sig=''):
    with open(f'{loss_dir}/{signature}_loss.pkl', 'rb') as file:
        data = pickle.load(file)

    list1, list2 = load_loss(signature)

    lossD = list1 + lossD
    lossG = list2 + lossG

    save_loss(lossD, lossG, signature=new_sig)


def plot_loss_curve(lossD, lossG):
    plt.figure(figsize=(20, 12))
    plt.plot(lossD, label='Discriminator Loss', zorder=3)
    plt.plot(lossG, label='Generator Loss', zorder=2)

    plt.legend()
    plt.xlabel('Number of Epoch', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title('Training Loss', fontsize=24)
    plt.grid()
    plt.show()


def visualize_model(netG, nz, device):
    with torch.no_grad():
        noise = torch.randn(36, nz, 1, 1, device=device)
        fake_images = netG(noise).detach().cpu()

    # Plot the generated images
    plt.figure(figsize=(16, 16))
    plt.axis("off")
    plt.title("Generated Anime Scenes", fontsize=24)
    plt.imshow(np.transpose(vutils.make_grid(fake_images, nrow=6, padding=1, normalize=True), (1, 2, 0)))
    plt.show()


def generate_images(netG, nz, num_images, device, path='./eval/'):
    with torch.no_grad():
        noise = torch.randn(num_images, nz, 1, 1, device=device)
        fake_image = netG(noise).detach().cpu()
    if os.path.isdir(path):
        for filename in os.listdir(path):
            os.remove(os.path.join(path, filename))
    else:
        os.makedirs(path)
    for i in range(num_images):
        save_image(fake_image[i], os.path.join(path, f"image_{i:04d}.png"), normalize=True)

def inception_score(dataloader, cuda=True, batch_size=32, splits=10):
    device = 'cuda' if cuda else 'cpu'
    inception_model = inception_model.to(device)

    N = len(dataloader.dataset)

    def get_pred(x):
        x = inception_model(x)
        return nn.functional.softmax(x, dim=1).data.cpu().numpy()

    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader):
        batch = batch.to(device)
        batch_size_i = batch.size(0)

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batch)

    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)