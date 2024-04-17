import PIL
from PIL import Image 
import numpy as np
import matplotlib.pyplot as plt


def display_images(images_with_titles: dict, show_axes=True, figsize=(15, 5)):
    num_images = len(images_with_titles)
    fig, axes = plt.subplots(1, num_images, figsize=figsize)
    if num_images == 1:
        axes = [axes]
    for ax, (title, image) in zip(axes, images_with_titles.items()):
        ax.imshow(image)
        ax.set_title(title)
        if not show_axes:
            ax.axis('off')
    plt.show()
    

def forward_diffusion_process(image, steps, beta):
    images_with_titles = {'Original': image}
    for step in range(1, steps + 1):
        image = forward_diffusion_step(image, step, beta)
        images_with_titles[f'Step {step}'] = image
    display_images(images_with_titles, show_axes=False)


def forward_diffusion_step(image, step, beta):
    """
    Применяет гауссовский шум к изображению согласно формуле зашумления.
    q(x_t | x_{t-1}) = N(x_t; μ_t = (1 - β_t) * x_{t-1}, Σ_t = β_t * I)
    """
    image_array = np.array(image)
    image_array = image_array / 255.0
    mean = (1 - beta) * image_array
    sigma = np.sqrt(beta)
    noisy_image_array = mean + sigma * np.random.randn(*image_array.shape) # Добавление гауссовского шума
    noisy_image_array = np.clip(noisy_image_array, 0, 1)
    noisy_image_array = (255 * noisy_image_array).astype(np.uint8)
    noisy_image = Image.fromarray(noisy_image_array)
    return noisy_image



