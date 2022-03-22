import seaborn as sns
import matplotlib.pyplot as plt
import torch


def plot_image(image):
    if isinstance(image, torch.Tensor):
        image = image.numpy()[0, :, :]

    sns.heatmap(image, cmap='gray')
    plt.show()
