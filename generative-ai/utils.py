import cv2
import PIL
from PIL import Image 
import numpy as np
import matplotlib.pyplot as plt


def display_images(images: dict | list, show_axes: bool = True, grid: tuple = (1, None), figsize: tuple = (15, 5), tight_layout: bool = True):
    if isinstance(images, list):
        titles = []
        for i, img in enumerate(images):
            if isinstance(img, np.ndarray):
                if len(img.shape) == 2:
                    mode = "L"
                if img.shape[2] == 3:
                    mode = 'RGB'
                if img.shape[2] == 4:
                    mode = 'RGBA'
                else:
                    mode = 'N/A'
                title = f"img{i}, {img.shape}, {mode}, N/A"
            else:
                title = f"img{i}, {img.size}, {img.mode}, {img.format}"
            titles.append(title)

        images_with_titles = dict(zip(titles, images))
    else:
        images_with_titles = images

    num_images = len(images_with_titles)
    rows, cols = grid if grid[1] is not None else (grid[0], max(1, num_images // grid[0]))
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if num_images > 1 else [axes]

    for ax, (title, image) in zip(axes, images_with_titles.items()):
        ax.imshow(image if isinstance(image, np.ndarray) else np.array(image))
        ax.set_title(title)
        if not show_axes:
            ax.axis('off')
    
    if tight_layout:
        plt.tight_layout()
    plt.show()
    
# 
def simple_ae_plot():
    # Определение координат точек
    x_values = np.array([-1, 0, 1, 2, 3, 4])
    y_values = x_values  # h = x1 + x2, простая линия y=x
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].plot(x_values, y_values, color='black', marker='o', mfc='green', markeredgecolor='green')
    axs[0].set_title('$h_1 = x_1 + x_2$')
    axs[0].axhline(0, color='black', linewidth=0.5)
    axs[0].axvline(0, color='black', linewidth=0.5)
    axs[0].hlines(y=2, xmin=0, xmax=2, linestyle='--', color='gray')
    axs[0].vlines(x=2, ymin=0, ymax=2, linestyle='--', color='gray')
    axs[0].set_xticks(x_values)
    axs[0].set_yticks(y_values)
    axs[0].set_xlabel('$x_1$')
    axs[0].set_ylabel('$x_2$')

    axs[1].plot(x_values, y_values, color='black', marker='o', mfc='green', markeredgecolor='green')
    axs[1].set_title('$h_1 = x_1 + x_2$')
    axs[1].axhline(0, color='black', linewidth=0.5)
    axs[1].axvline(0, color='black', linewidth=0.5)
    axs[1].set_xticks(x_values)
    axs[1].set_yticks(y_values)
    axs[1].set_xlabel('$x_1$')
    axs[1].set_ylabel('$x_2$')

    x1, y1 = 1, 3
    x2, y2 = 2, 2
    
    axs[1].hlines(y=y2, xmin=0, xmax=x2, linestyle=':', color='green')
    axs[1].vlines(x=x2, ymin=0, ymax=y2, linestyle=':', color='green')
    axs[1].annotate(f'(2, 2)', (2, 2), textcoords="offset points", xytext=(10, -10), ha='left', va='top')
    
    axs[1].plot(x1, y1, marker='o', mfc='red', markeredgecolor='red')
    axs[1].hlines(y=y1, xmin=0, xmax=x1, linestyle=':', color='red')
    axs[1].vlines(x=x1, ymin=0, ymax=y1, linestyle=':', color='red')
    axs[1].annotate(f'(1, 3)', (1, 3), textcoords="offset points", xytext=(0, 10), ha='right')

    plt.arrow(x1, y1, x2 - x1, y2 - y1, lw=1, length_includes_head=True, head_width=0.15, color='gray')

    # Показать график
    plt.tight_layout()
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


def refine_img_morphology(img_pil, kernel_size=15, erosion_iter=1, dilation_iter=3):
    """ 
    - убрать черные засечки вне маски
    - белые засечки внутри маски
    - увеличить маскируемую область (по форме)
    
    """

    img_np = np.array(img_pil)
    
    # Создание ядра для морфологических операций
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Удаление белых засечек вне объекта
    erosion = cv2.erode(img_np, kernel, iterations=erosion_iter)

    # Закрашивание черных областей внутри объекта
    dilation = cv2.dilate(erosion, kernel, iterations=dilation_iter)
    closing = cv2.erode(dilation, kernel, iterations=erosion_iter)

    # Конвертация обработанного изображения обратно в формат PIL
    if closing.ndim == 3:
        closing = cv2.cvtColor(closing, cv2.COLOR_BGR2RGB)
    processed_pil_image = Image.fromarray(closing)

    return processed_pil_image


def overlay_colored_masks(img_pil: Image.Image, masks: list[Image.Image], color=(0, 255, 0, 153)):
    """ Накладываем все маски (L) на исходное изображение """
    
    # Конвертируем исходное изображение в формат RGBA, если оно не в этом формате
    if img_pil.mode != 'RGBA':
        img_pil = img_pil.convert('RGBA')

    # Создаем полупрозрачное изображение для наложения масок
    overlay = Image.new('RGBA', img_pil.size, (255, 255, 255, 0))

    # Проходим по всем маскам
    for mask in masks:
        # Проверяем, является ли маска изображением в режиме 'L'
        if mask.mode != 'L':
            mask = mask.convert('L')
        
        colored_mask = Image.new('RGBA', img_pil.size, color)

        # Накладываем маску на изображение с цветом
        overlay.paste(colored_mask, (0, 0), mask)

    # Накладываем полупрозрачное изображение на оригинальное
    combined = Image.alpha_composite(img_pil, overlay)
    
    return combined



