import io
from typing import Union, List

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_attention(
    image_path: Union[str, Image.Image],
    words: List[str], 
    alphas, 
    betas, 
    image_size=(50, 50)
) -> Image.Image:
    """Visualize attention maps
    
    Args:
        image_path (Union[str, Image.Image]): Image path or PIL image
        words (List[str]): List of words
        alphas: Attention weights
        betas: Sentinel gate values
        image_size (tuple, optional): Image size. Defaults to (49, 49).

    Returns:
        Image.Image: PIL image
    """

    if isinstance(image_path, str):    
        image = Image.open(image_path)
    else:
        image = image_path

    image = image.resize(image_size, Image.LANCZOS)
    if isinstance(alphas, torch.Tensor):
        alphas = alphas.detach().numpy()
    elif isinstance(alphas, list):
        alphas = np.array(alphas)

    if isinstance(betas, torch.Tensor):
        betas = betas.detach().numpy()
    elif isinstance(betas, list):
        betas = np.array(betas)

    for t in range(len(words)):
        plt.subplot(int(np.ceil(len(words) / 5.)), 5, t + 1)
        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.text(10, 65, '%.2f' % (1-(betas[t])), color='green', backgroundcolor='white', fontsize=15)
        plt.imshow(image)
        current_alpha = alphas[t, :]

        pil_alpha = Image.fromarray(current_alpha)
        pil_alpha = pil_alpha.resize(image_size, Image.LANCZOS)
        alpha = np.array(pil_alpha)

        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.5)
        plt.set_cmap('jet')
        plt.axis('off')
        
    # convert to PIL image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return Image.open(buf)
