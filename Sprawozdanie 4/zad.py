import numpy as np
from skimage import io
import cv2
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy.fft import fft2, fftshift, dct, idct
from scipy import fftpack
from PIL import Image
import pywt

def MAE(image1,image2):
    assert image1.shape == image2.shape, "Obrazy mają różne wielkości"
    return np.mean(np.abs(image1-image2))

def C_ratio(original, image):
    return original.size * original.dtype.itemsize * 8 / (image.size * image.dtype.itemsize * 8)

def compress(image, compression_ratio):
    f_image = np.fft.fft2(image)
    threshold = np.percentile(np.abs(f_image), 100 - compression_ratio * 100)
    f_image[np.abs(f_image) <  threshold] = 0
    return np.fft.ifft2(f_image).real

def compress_RGB(image, cr):
    compressed_channels = [compress(image[:,:,c], cr) for c in range(image.shape[2])]
    return np.dstack(compressed_channels)

def wavelet_compress(image, compression_ratio):
    img = np.array(image)
    coeffs = pywt.dwt2(img,"bior1.3")
    cA, (cH, cV, cD) = coeffs
    threshold = np.max(np.abs(cD)) * compression_ratio / 10.0 
    cD_threshold = pywt.threshold(cD, threshold, mode="soft")
    coeefs_threshold = (cA,(cH, cV, cD_threshold))
    output = pywt.idwt2(coeefs_threshold, "bior1.3")
    
    return output

def wavelet_compress_color(image, compression_ratio):
    img_compress = np.zeros_like(image, dtype=np.uint8)

    for channel in range(3):
        coeffs = pywt.dwt2(image[:,:,channel], "bior1.3")
        cA, (cH, cV, cD) = coeffs
        threshold = np.max(np.abs(cD)) * compression_ratio / 10.0
        cD_threshold = pywt.threshold(cD, threshold, mode="soft")
        img_compress[:,:,channel] = pywt.idwt2((cA, (cH, cV, cD_threshold)), "bior1.3")
    
    return img_compress

def dct2(image):
    return dct(dct(image.T, norm='ortho').T, norm='ortho')

def idct2a(image):
    return idct(idct(image.T,norm='ortho').T, norm='ortho')



def poissoning(image, lambda_value=64):
    noisy_image = np.zeros_like(image, dtype=np.float64)

    for i in range(image.shape[2]):
        channel_noise = np.random.poisson(image[:, :, i] / 255.0 * lambda_value) / lambda_value * 255.0
        noisy_image[:, :, i] = np.clip(channel_noise, 0, 255)

    noisy_image = noisy_image.astype(np.uint8)
    
    return noisy_image

def poissoning_gray(image, lambda_value=64):
    scaled_image = image / 255.0
    noisy_scaled_image = np.random.poisson(scaled_image * lambda_value) / lambda_value * 255.0
    clipped_noisy_image = np.clip(noisy_scaled_image, 0, 255).astype(np.uint8)
    
    return clipped_noisy_image

def display_TF(image):
    abs_val = np.abs(image)
    plt.figure(figsize=(15, 5))
    
    for i in range(image.shape[2]):
        plt.subplot(1, 3, i + 1)
        plt.imshow(np.fft.fftshift(abs_val[:, :, i]), norm=LogNorm(vmin=5))
        plt.title(f"Channel {i + 1}")
        plt.colorbar()

    plt.tight_layout()

def filter(channel):
    channel[-1000:-200, ] = 0 
    return channel

def denoising(image):
    return np.apply_along_axis(filter, 2, image)

def denoising_gray(image):
    denoised_img = np.zeros_like(image,dtype = np.complex128)
    denoised_img = filter(image)
    return denoised_img

def wavelet_denoise(image, threshold=0.1):
    coeffs2 = pywt.dwt2(image, 'bior1.3')
    thresholded_coeffs = [pywt.threshold(x, threshold * np.max(np.abs(x)), mode='soft') for x in coeffs2]
    denoised_image = pywt.idwt2(tuple(thresholded_coeffs), 'bior1.3')
    denoised_image = np.round(denoised_image).astype(np.uint8)

    return denoised_image

def gaussian_denoising(image, kernel_size=(5, 5), sigma=1.0):
    return np.round(cv2.GaussianBlur(image, kernel_size, sigma)).astype(np.uint8)


def display_comparison(original, compressed, title_original, title_compressed):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap="gray" if len(original.shape) == 2 else None)
    plt.title(title_original)

    plt.subplot(1, 2, 2)
    plt.imshow(compressed, cmap="gray" if len(compressed.shape) == 2 else None)
    plt.title(title_compressed)
    plt.show()

def evaluate_compression(original, compressed):
    mae = MAE(original, compressed)
    compression_ratio = C_ratio(original, compressed)
    print(f"Mean Absolute Error: {mae}")
    print(f"Compression Ratio (C): {compression_ratio}")



image_gray = io.imread("Colors.jpg", as_gray=True)
color_image = io.imread("Colors.jpg")

compressed_fft_gray = compress(image_gray, 0.01)
compressed_fft_color = compress_RGB(color_image, 0.1)

compressed_wavelet_gray = wavelet_compress(image_gray, 0.3)
compressed_wavelet_color = wavelet_compress_color(color_image, 0.01)

compressed_dct_gray = idct2a(dct2(image_gray))

display_comparison(image_gray, compressed_fft_gray, "Obraz oryginalny (FFT)", "Obraz skompresowany (FFT)")
evaluate_compression(image_gray, compressed_fft_gray)

display_comparison(color_image, np.uint8(compressed_fft_color), "Obraz oryginalny (FFT)", "Obraz skompresowany (FFT)")
evaluate_compression(color_image, np.uint8(compressed_fft_color))

display_comparison(image_gray, compressed_wavelet_gray, "Obraz oryginalny (Wavelet)", "Obraz skompresowany (Wavelet)")
evaluate_compression(image_gray, compressed_wavelet_gray)

display_comparison(color_image, np.uint8(compressed_wavelet_color), "Obraz oryginalny (Wavelet)", "Obraz skompresowany (Wavelet)")
evaluate_compression(color_image, np.uint8(compressed_wavelet_color))

display_comparison(image_gray, compressed_dct_gray, "Obraz oryginalny (DCT)", "Obraz skompresowany (DCT)")
evaluate_compression(image_gray, compressed_dct_gray)