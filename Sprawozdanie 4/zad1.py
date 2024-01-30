import cv2
import numpy as np
from matplotlib import pyplot as plt
import pywt

# Wykonać odszumianie obrazu szarego za pomocą transformacji Fouriera

def fourier_denoise(image, threshold=0.1):
    # Przekształcenie Fouriera
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    
    # Obliczenie amplitudy
    amplitude_spectrum = np.abs(f_transform_shifted)

    # Zastosowanie progu, aby usunąć składowe o niskiej amplitudzie (szum)
    mask = amplitude_spectrum > threshold * np.max(amplitude_spectrum)
    f_transform_shifted_denoised = f_transform_shifted * mask
    
    # Przekształcenie odwrotne Fouriera
    f_transform_denoised = np.fft.ifftshift(f_transform_shifted_denoised)
    image_denoised = np.fft.ifft2(f_transform_denoised).real
    
    return image_denoised

# Wykonać odszumianie obrazu kolorowego za pomocą trzech transformacji Fouriera, po jednej na każdy kanał obrazu.

def fourier_denoise_color(image, threshold=0.1):
    # Przekształcenie Fouriera dla każdego kanału
    channels = cv2.split(image)
    denoised_channels = []

    for channel in channels:
        # Przekształcenie Fouriera
        f_transform = np.fft.fft2(channel)
        f_transform_shifted = np.fft.fftshift(f_transform)

        # Obliczenie amplitudy
        amplitude_spectrum = np.abs(f_transform_shifted)

        # Zastosowanie progu, aby usunąć składowe o niskiej amplitudzie (szum)
        mask = amplitude_spectrum > threshold * np.max(amplitude_spectrum)
        f_transform_shifted_denoised = f_transform_shifted * mask

        # Przekształcenie odwrotne Fouriera
        f_transform_denoised = np.fft.ifftshift(f_transform_shifted_denoised)
        channel_denoised = np.fft.ifft2(f_transform_denoised).real

        denoised_channels.append(channel_denoised)

    # Złożenie odszumionych kanałów w obraz kolorowy
    image_denoised = cv2.merge(denoised_channels)
    
    return image_denoised

# Wykonać odszumianie obrazu szarego za pomocą transformacji falkowej, lub innej transformacji ortogonalnej.

def wavelet_denoise(image, wavelet='haar', level=1):
    # Przekształcenie falkowe
    coeffs = pywt.wavedec2(image, wavelet, level=level)

    # Zastosowanie progu do współczynników
    threshold = 0.1  # Dostosuj ten próg według potrzeb
    coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]

    # Odwrotne przekształcenie falkowe
    image_denoised = pywt.waverec2(coeffs_thresholded, wavelet)

    return image_denoised

# Bonus: Porównać odszumianie z innymi metodami (np. z odszumianiem poprzez rozmycie Gaussowskie)

# Funkcja do odszumiania poprzez rozmycie Gaussowskie
def gaussian_blur_denoise(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

# Funkcja do porównania odszumiania dla różnych metod
def compare_denoising_methods(image):
    # Odszumianie przez rozmycie Gaussowskie
    denoised_gaussian = gaussian_blur_denoise(image)

    # Odszumianie przez transformację falkową
    denoised_wavelet = wavelet_denoise(image)

    # Wyświetlenie oryginalnego, odszumionego przez rozmycie Gaussowskie i odszumionego przez transformację falkową obrazu
    plt.figure(figsize=(10, 5))

    plt.subplot(131), plt.imshow(image, cmap='gray'), plt.title('Oryginał')
    plt.subplot(132), plt.imshow(denoised_gaussian, cmap='gray'), plt.title('Odszumianie Gaussowskie')
    plt.subplot(133), plt.imshow(denoised_wavelet, cmap='gray'), plt.title('Odszumianie falkowe')

    plt.show()

# Zaimplementować algorytm kompresji stratnej korzystający z transformacji Fouriera dla obrazow szarych.

def quantize_coefficients(coefficients, quantization_step):
    return np.round(coefficients / quantization_step)

def dequantize_coefficients(quantized_coefficients, quantization_step):
    return quantized_coefficients * quantization_step

def compress_lossy(image, quantization_step):
    # Przekształcenie Fouriera
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Kwantyzacja współczynników
    quantized_coefficients = quantize_coefficients(f_transform_shifted, quantization_step)

    # Odwrotna kwantyzacja
    dequantized_coefficients = dequantize_coefficients(quantized_coefficients, quantization_step)

    # Odwrotne przekształcenie Fouriera
    f_transform_inverse = np.fft.ifftshift(dequantized_coefficients)
    image_compressed = np.fft.ifft2(f_transform_inverse).real

    return image_compressed

# Zaimplementować algorytm kompresji stratnej korzystający z transformacji Fouriera dla obrazow szarych kolorowych.

def compress_lossy_color(image, quantization_step):
    # Przekształcenie Fouriera dla każdego kanału
    channels = cv2.split(image)
    compressed_channels = []

    for channel in channels:
        # Przekształcenie Fouriera
        f_transform = np.fft.fft2(channel)
        f_transform_shifted = np.fft.fftshift(f_transform)

        # Kwantyzacja współczynników
        quantized_coefficients = quantize_coefficients(f_transform_shifted, quantization_step)

        # Odwrotna kwantyzacja
        dequantized_coefficients = dequantize_coefficients(quantized_coefficients, quantization_step)

        # Odwrotne przekształcenie Fouriera
        f_transform_inverse = np.fft.ifftshift(dequantized_coefficients)
        channel_compressed = np.fft.ifft2(f_transform_inverse).real

        compressed_channels.append(channel_compressed)

    # Złożenie skompresowanych kanałów w obraz kolorowy
    image_compressed = cv2.merge(compressed_channels)

    return image_compressed

# Zaimplementować algorytm kompresji stratnej korzystający z transformacji falkowej, lub innej transformacji ortogonalnej.

def wavelet_compress(image, wavelet='haar', level=1, quantization_step=10):
    # Przekształcenie falkowe
    coeffs = pywt.wavedec2(image, wavelet, level=level)

    # Kwantyzacja współczynników
    quantized_coeffs = [quantize_coefficients(c, quantization_step) for c in coeffs]

    # Odwrotna kwantyzacja
    dequantized_coeffs = [dequantize_coefficients(c, quantization_step) for c in quantized_coeffs]

    # Odwrotna transformacja falkowa
    image_compressed = pywt.waverec2(dequantized_coeffs, wavelet)

    return image_compressed

def quantize_coefficients(coefficients, quantization_step):
    return np.round(coefficients / quantization_step)

def dequantize_coefficients(quantized_coefficients, quantization_step):
    return quantized_coefficients * quantization_step

# Bonus: Zaimplementować adaptacyjny algorytm doboru współczynnika kompresji.

def calculate_energy(image):
    # Obliczenie energii sygnału w transformacji Fouriera
    f_transform = np.fft.fft2(image)
    energy = np.sum(np.abs(f_transform) ** 2)
    return energy

def adaptive_compression(image, initial_quantization_step, threshold_energy_ratio=0.99):
    # Inicjalizacja współczynnika kompresji
    quantization_step = initial_quantization_step

    # Obliczenie początkowej energii
    initial_energy = calculate_energy(image)

    while quantization_step > 1:
        # Kompresja z aktualnym współczynnikiem
        compressed_image = compress_lossy(image, quantization_step)

        # Obliczenie energii po kompresji
        compressed_energy = calculate_energy(compressed_image)

        # Sprawdzenie stosunku energii
        energy_ratio = compressed_energy / initial_energy

        if energy_ratio > threshold_energy_ratio:
            # Jeśli stosunek energii jest zbyt niski, zmniejsz współczynnik kompresji
            quantization_step /= 2
        else:
            # W przeciwnym razie przerwij pętlę
            break

    return compressed_image, quantization_step