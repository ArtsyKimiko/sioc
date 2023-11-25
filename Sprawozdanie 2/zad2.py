import numpy as np
import cv2
import math

def calculate_mse(img1, img2):
    return np.square(np.subtract(img1, img2)).mean()

def resize_image(img, new_width=None, new_height=None, scale_factor=None, method='interpolation'):
    if scale_factor is not None:
        new_width = int(img.shape[1] * scale_factor)
        new_height = int(img.shape[0] * scale_factor)
    
    if method == 'interpolation':
        if len(img.shape) == 3:
            old_height, old_width, channels = img.shape
        elif len(img.shape) == 2:
            old_height, old_width = img.shape
            channels = 1
            img = img.reshape((old_height, old_width, 1))

        new_img = np.zeros((new_height, new_width, channels))
        width_factor = old_width / new_width if new_width != 0 else 0
        height_factor = old_height / new_height if new_height != 0 else 0

        for i in range(new_height):
            for j in range(new_width):
                x = i * height_factor
                y = j * width_factor
                x_floor = math.floor(x)
                y_floor = math.floor(y)
                x_ceil = min(old_height - 1, math.ceil(x))
                y_ceil = min(old_width - 1, math.ceil(y))

                for c in range(channels):
                    if (x_floor == x_ceil) and (y_floor == y_ceil):
                        new_img[i, j, c] = img[x_floor, y_floor, c]
                    elif (x_ceil == x_floor):
                        q1 = img[int(x), int(y_floor), c]
                        q2 = img[int(x), int(y_ceil), c]
                        new_img[i, j, c] = q1 * (y_ceil - y) + q2 * (y - y_floor)
                    elif (y_ceil == y_floor):
                        q1 = img[x_floor, y_floor, c]
                        q2 = img[x_ceil, y_floor, c]
                        new_img[i, j, c] = q1 * (x_ceil - x) + q2 * (x - x_floor)
                    else:
                        v1 = img[x_floor, y_floor, c]
                        v2 = img[x_ceil, y_floor, c]
                        v3 = img[x_floor, y_ceil, c]
                        v4 = img[x_ceil, y_ceil, c]
                        q1 = v1 * (x_ceil - x) + v2 * (x - x_floor)
                        q2 = v3 * (x_ceil - x) + v4 * (x - x_floor)
                        new_img[i, j, c] = q1 * (y_ceil - y) + q2 * (y - y_floor)

        return new_img.squeeze() if channels == 1 else new_img

    elif method == 'convolution':
        kernel = np.ones((3, 3), np.float32) / 9
        img_convolved = cv2.filter2D(img, -1, kernel)
        new_img = img_convolved[::2, ::2]
        return new_img
    
    else:
        raise ValueError("Nieprawidłowa metoda. Dostępne metody to 'interpolation' lub 'convolution'.")

# Przykłady użycia dla zmniejszania obrazu
img1 = cv2.imread("Input1_256x256.png")
img2 = cv2.imread("Input2_640x635.jpg")

img1_resized_convolution = resize_image(img1, 128, 128, method='convolution')
img2_resized_convolution = resize_image(img2, 320, 320, method='convolution')

cv2.imwrite("Image1_128x128_convolution.png", img1_resized_convolution)
cv2.imwrite("Image2_320x320_convolution.jpg", img2_resized_convolution)

# Ocena jakości za pomocą MSE
print("MSE(img1, img1_resized_convolution):", calculate_mse(img1, resize_image(img1_resized_convolution, 256, 256)))
print("MSE(img2, img2_resized_convolution):", calculate_mse(img2, resize_image(img2_resized_convolution, 640, 635)))