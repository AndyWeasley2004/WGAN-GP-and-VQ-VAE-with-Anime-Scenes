import numpy as np
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import entropy
import os
from torchvision import models
import torch.nn as nn

inception_model = models.inception_v3(pretrained=True, transform_input=False)
inception_model = inception_model.eval()

def draw_sample_image(x, postfix):

    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Visualization of {}".format(postfix))
    plt.imshow(np.transpose(make_grid(x.detach().cpu(), padding=2, normalize=True), (1, 2, 0)))


def plot_train_loss(total_loss, rec_loss, cb_loss):
    plt.figure(figsize=(10, 6))
    plt.plot(total_loss, color='blue', label='Total Loss')
    plt.plot(rec_loss, color='green', label='Reconstruct Loss')
    plt.plot(cb_loss, color='red', label='Codebook Loss')
    plt.title("Training Loss", fontsize=20)
    plt.legend()
    plt.show()


def generate_images(model, test_dataloader, device='cuda', 
                    gen_path='./reconstruct/', test_path='./original'):
    if os.path.isdir(gen_path):
        for filename in os.listdir(gen_path):
            os.remove(os.path.join(gen_path, filename))
    else:
        os.makedirs(gen_path)

    if os.path.isdir(test_path):
        for filename in os.listdir(test_path):
            os.remove(os.path.join(test_path, filename))
    else:
        os.makedirs(test_path)

    for _, x in enumerate(tqdm(test_dataloader)):
        x = x.to(device)
        x_hat,_,_ = model(x)
        for i in range(len(x_hat)):
            save_image(x_hat[i], os.path.join(gen_path, f"image_{i:04d}.png"), normalize=True)
            save_image(x[i], os.path.join(test_path, f"image_{i:04d}.png"), normalize=True)


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