import numpy as np
import cv2
from skimage import io, transform
from scipy import ndimage
import matplotlib.pyplot as plt

# Wczytanie obrazu
image = io.imread(r"./Bayer/circle.jpg")

# Zmiana rozmiaru obrazu na 1024x1024
image = image[:, :, :3]
image = transform.resize(image, output_shape=(1024, 1024, 3))

# Konwersja na typ uint8 przed zapisaniem
image_to_save = (image * 255).astype(np.uint8)

# Zapis oryginalnego obrazu
io.imsave("oryginalny_obraz.jpg", image_to_save)

def color_filter(mask, shape):
    return np.dstack([np.tile(color_mask, np.asarray(shape) // len(color_mask)) for color_mask in mask])

# Filtr Bayera
bayer_mask = np.array([[[0, 1], [0, 0]],
                       [[1, 0], [0, 1]],
                       [[0, 0], [1, 0]]], dtype=np.uint8)

# Zastosowanie filtra Bayera do obrazu
bayer_filter = color_filter(bayer_mask, shape=np.array([1024, 1024]))
sensor_im = image * bayer_filter

# Zapis obrazu po zastosowaniu filtra Bayera
sensor_im_to_save = (sensor_im * 255).astype(np.uint8)
io.imsave("obraz_po_filtrowaniu.jpg", sensor_im_to_save)

# Maska demozajkowania
demosaicking_mask = np.dstack([
    np.ones([2, 2]),
    0.5 * np.ones([2, 2]),
    np.ones([2, 2]),
])

# Demozajkowanie
original_image = np.dstack([ndimage.convolve(sensor_im[:, :, channel], demosaicking_mask[:, :, channel], mode="constant", cval=0.0) for channel in range(3)])

# Zapis oryginalnego obrazu po demozajkowaniu
original_image_to_save = (original_image * 255).astype(np.uint8)
io.imsave("oryginalny_obraz_po_demozajkowaniu.jpg", original_image_to_save)
