import seaborn as sns
import matplotlib.pyplot as plt
import torch
from PIL import Image
import config


def plot_image(image):
    if isinstance(image, torch.Tensor):
        image = image.numpy()[0, :, :]

    sns.heatmap(image, cmap='gray')
    plt.show()


def save_img(img, path):
    img = img.cpu().detach().numpy() * 255
    im = Image.fromarray(img)
    im = im.convert('L')
    im.save(path)
