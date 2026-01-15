import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import skimage.morphology
import skimage.measure

def clean_and_show_binary_image(binary_image, min_size):

    if binary_image is None:
        print("Error: No se pudo cargar la imagen binaria.")
        return None

    # Remove small objects
    labels = skimage.morphology.label(binary_image)
    region_props = skimage.measure.regionprops(labels)
    for region in region_props:
        if region.area < min_size:
            for coord in region.coords:
                labels[coord[0], coord[1]] = 0

    # Convert to binary image
    cleaned_binary_image = np.where(labels >= 1, 1, 0)

    # Show the images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(binary_image, cmap='gray')
    plt.title('Imagen Original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cleaned_binary_image, cmap='gray')
    plt.title('Imagen Limpia')
    plt.axis('off')

    plt.show()

    return cleaned_binary_image

def segment_image(image1, thresholds):
    # Crear una lista para almacenar las imágenes segmentadas
    segmented_images = []

    # Asignar etiquetas a cada región según los umbrales
    for i in range(len(thresholds) + 1):
        segmented_image = np.zeros_like(image1)  # Crear una nueva matriz en cada iteración

        if i == 0:
            segmented_image[image1 <= thresholds[i]] = 1
        elif i == len(thresholds):
            segmented_image[image1 > thresholds[i - 1]] = i + 1
        else:
            segmented_image[(image1 > thresholds[i - 1]) & (image1 <= thresholds[i])] = i + 1

        # Almacenar la imagen segmentada actual en la lista
        segmented_images.append(segmented_image)

    return segmented_images

def generate_single_channel_image(segmented_images):
    # Sumar todas las imágenes segmentadas para obtener una imagen en un solo canal
    result_image = np.sum(segmented_images, axis=0)

    # Escalar la imagen resultante para que esté en el rango [0, 255]
    result_image = (result_image / np.max(result_image) * 255).astype(np.uint8)
    result_image = np.asarray(result_image)
    return result_image

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))
