import numpy as np
from skimage import io
from matplotlib import pyplot as plt
import pywt

# 1. Zaimplementować algorytm kompresji stratnej korzystający z transformacji Fouriera dla obrazow szarych.   
def Fourier_compress(img, ratio):
    transformed = np.fft.fft2(img)
    sorted_transformed = np.sort(np.abs(transformed.flatten()))
    threshold = sorted_transformed[int((1 - ratio) * len(sorted_transformed))]
    transformed[np.abs(transformed) < threshold] = 0
    return np.fft.ifft2(transformed).real

# 2. Zaimplementować algorytm kompresji stratnej korzystający z transformacji Fouriera dla obrazow kolorowych.
def Fourier_color_compress(image, ratio):
    RED, GREEN, BLUE = image[:,:,0], image[:,:,1], image[:,:,2]
    red, green, blue = Fourier_compress(RED, ratio), Fourier_compress(GREEN, ratio), Fourier_compress(BLUE, ratio)
    
    return np.dstack((red,green,blue))

# 3. Zaimplementować algorytm kompresji stratnej korzystający z transformacji falkowej, lub innej transformacji ortogonalnej.
def Wavelet_compress(image, ratio):
    img_array = np.array(image)
    coeffs = pywt.dwt2(img_array, "bior1.3")
    cA, (cH, cV, cD) = coeffs
    threshold = np.max(np.abs(cD)) * ratio / 10.0
    cD_threshold = pywt.threshold(cD, threshold, mode="soft")
    coeefs_threshold = (cA, (cH, cV, cD_threshold))
    result = pywt.idwt2(coeefs_threshold, "bior1.3")
    return result

def Wavelet_compress_color(img, ratio):
    img_c = np.zeros_like(img, dtype=np.uint8)
    for ch in range(3):
        img_c[:,:,ch] = Wavelet_compress(img[:,:,ch], ratio)
    return img_c

# Zaszumianie Poissona
def Poissoning(image, lambda_value=64):
    noisy_image = np.zeros_like(image, dtype=np.float64)
    for i in range(image.shape[2]):
        noisy_channel = np.random.poisson(image[:, :, i] / 255.0 * lambda_value) / lambda_value * 255.0
        noisy_image[:, :, i] = np.clip(noisy_channel, 0, 255)

    noisy_image = noisy_image.astype(np.uint8)
    
    return noisy_image

def Poissoning_gray(image,lambda_value=64):
    noisy_image = np.zeros_like(image, dtype=np.float64)
    noisy_image = np.random.poisson(image / 255.0 * lambda_value) / lambda_value * 255.0
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image   
 

def filter(channel):
    channel[-540:-70, ] = 0 
    return channel

# 1. Wykonać odszumianie obrazu szarego za pomocą transformacji Fouriera
def Fourier_denoise(image):
    denoised = np.zeros_like(image,dtype = np.complex128)
    denoised = filter(image)
    return denoised

# 2. Wykonać odszumianie obrazu kolorowego za pomocą trzech transformacji Fouriera, po jednej na każdy kanał obrazu.
def Fourier_color_denoise(image):
    denoised = np.zeros_like(image,dtype = np.complex128)
    for i in range(image.shape[2]):
        denoised[:,:,i] = filter(image[:,:,i])

    return denoised

# 3. Wykonać odszumianie obrazu szarego za pomocą transformacji falkowej, lub innej transformacji ortogonalnej.
def Wavelet_denoise(image,threshold = 0.1):
    coeffs_2d = pywt.dwt2(image, 'bior1.3')
    coeffs_2d = tuple(map(lambda x: pywt.threshold(x, threshold*np.max(np.abs(x)), mode='soft'), coeffs_2d))

    image = pywt.idwt2(coeffs_2d, 'bior1.3')
    image = np.round(image).astype(np.uint8)

    return image

#--------------------------------------------------
def display_img(original, compressed, title_original, title_compressed):
    plt.figure(figsize=(12,6))
    
    # Original image
    plt.subplot(1,2,1)
    plt.imshow(original, cmap="gray" if len(original.shape) == 2 else None)
    plt.title(title_original)

    # Compressed image
    plt.subplot(1,2,2)
    plt.imshow(compressed, cmap="gray" if len(compressed.shape) == 2 else None)
    plt.title(title_compressed)
    
    plt.show()

image_gray = io.imread("gray_cat.jpg", as_gray=True)
compressed_gray = Fourier_compress(image_gray, 0.01)
display_img(image_gray, compressed_gray, "Original Image", "Compressed Image")

color_image = io.imread("cat.jpg")
compressed_color = Fourier_color_compress(color_image, 0.1)
display_img(color_image, np.uint8(compressed_color), "Original Image", "Compressed Image")

wavelet_compressed_image = Wavelet_compress(image_gray, 0.3)
display_img(image_gray, wavelet_compressed_image, "Original Image", "Wavelet Compressed Image")

wavelet_compressed_image_color = Wavelet_compress_color(color_image, 0.01)
display_img(color_image, wavelet_compressed_image_color, "Original Image", "Wavelet Compressed Image")

image = io.imread(r"cat.jpg")
gray_image = io.imread(r"gray_cat.jpg")

noised_image = Poissoning(image, 64)
display_img(image, noised_image, "Original Image", "Noised Image")

gray_noised_image = Poissoning(gray_image, 64)
display_img(gray_image, gray_noised_image, "Original Image", "Noised Image")

img_Fourier = np.fft.fft2(noised_image, axes=(0, 1))
denoised_image_TF = Fourier_color_denoise(img_Fourier)
denoised_image = np.fft.ifft2(denoised_image_TF, axes=(0, 1)).real
display_img(noised_image, np.uint8(denoised_image), "Noised Image", "Fourier Denoised Image")

gray_img_Fourier = np.fft.fft2(gray_noised_image, axes=(0, 1))
gray_denoised_image_TF = Fourier_denoise(gray_img_Fourier)
gray_denoised_image = np.fft.ifft2(gray_denoised_image_TF, axes=(0, 1)).real
display_img(gray_noised_image, np.uint8(gray_denoised_image), "Noised Image", "Fourier Denoised Image")

gray_img_wavelet = np.fft.fft2(gray_noised_image, axes=(0, 1))
gray_denoised_image_wav = Fourier_denoise(gray_img_wavelet)
gray_denoised_image = np.fft.ifft2(gray_denoised_image_wav, axes=(0, 1)).real
display_img(gray_noised_image, np.uint8(gray_denoised_image), "Noised Image", "Wavelet Denoised Image")