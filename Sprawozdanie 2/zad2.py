import numpy as np
import cv2
import math
import os
import matplotlib.pyplot as plt

def calculate_mse(img1, img2):
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0])) # Resize img2 to match the dimensions of img1
    return np.square(np.subtract(img1, img2)).mean() # Calculate Mean Squared Error (MSE)

def resize_image(img, new_width=None, new_height=None, scale_factor=None, method='interpolation'):
    # Calculate new dimensions based on scale factor or provided width/height
    if scale_factor is not None:
        new_width = int(img.shape[1] * scale_factor)
        new_height = int(img.shape[0] * scale_factor)
    elif new_width is not None and new_height is not None:
        scale_factor = new_width / img.shape[1]
    else:
        raise ValueError("Either 'scale_factor' or both 'new_width' and 'new_height' must be provided.")

    if method == 'interpolation' or method == 'max_pooling':
        # Check the number of image channels
        if len(img.shape) == 3:
            old_height, old_width, channels = img.shape
        elif len(img.shape) == 2:
            old_height, old_width = img.shape
            channels = 1
            img = img.reshape((old_height, old_width, 1))

        # Create an empty new_img with the specified dimensions
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

                if method == 'interpolation':
                    for c in range(channels):
                        # Determine how to fill in the new pixel based on the relationship between source and target pixels:
                        if (x_floor == x_ceil) and (y_floor == y_ceil):
                            # If the source and target pixels are the same, no interpolation is needed
                            new_img[i, j, c] = img[x_floor, y_floor, c]
                        elif (x_ceil == x_floor):
                            # If the source pixels are on the same row, interpolate along the y-axis.
                            q1 = img[int(x), int(y_floor), c]
                            q2 = img[int(x), int(y_ceil), c]
                            new_img[i, j, c] = q1 * (y_ceil - y) + q2 * (y - y_floor)
                        elif (y_ceil == y_floor):
                            # If the source pixels are in the same column, interpolate along the x-axis.
                            q1 = img[x_floor, y_floor, c]
                            q2 = img[x_ceil, y_floor, c]
                            new_img[i, j, c] = q1 * (x_ceil - x) + q2 * (x - x_floor)
                        else:
                            # For general cases, use bilinear interpolation with four surrounding pixels (v1, v2, v3, v4)
                            v1 = img[x_floor, y_floor, c]
                            v2 = img[x_ceil, y_floor, c]
                            v3 = img[x_floor, y_ceil, c]
                            v4 = img[x_ceil, y_ceil, c]
                            q1 = v1 * (x_ceil - x) + v2 * (x - x_floor)
                            q2 = v3 * (x_ceil - x) + v4 * (x - x_floor)
                            new_img[i, j, c] = q1 * (y_ceil - y) + q2 * (y - y_floor)

                elif method == 'max_pooling':
                    for c in range(channels):
                        # Take the maximum value from the specified region in the original image
                        new_img[i, j, c] = np.max(img[x_floor:x_ceil + 1, y_floor:y_ceil + 1, c])
                        return new_img.squeeze() if channels == 1 else new_img

    elif method == 'convolution':
        # Convolution using a 3x3 averaging kernel
        kernel = np.ones((3, 3), np.float32) / 9
        img_convolved = cv2.filter2D(img, -1, kernel)
        new_img = img_convolved[::2, ::2]
        return new_img

    else:
        raise ValueError("Invalid method. Available methods are 'interpolation', 'convolution' or 'max_pooling'.")

def run_tests(img, img_number):
    methods = ['interpolation', 'convolution', 'max_pooling']
    scale_factors = [0.5, 2]

    # Iterate over methods and scale factors to create combined images
    for method in methods:
        for scale_factor in scale_factors:
            create_combined_image(img, scale_factor, method, img_number)

def create_combined_image(img, scale_factor, method, img_number):
    original_img = img.copy()

    enlarged_img = resize_image(img, scale_factor=scale_factor, method=method)
    shrunk_img = resize_image(img, scale_factor=1/scale_factor, method=method)

    # Image dimensions
    original_size = f"{original_img.shape[1]}x{original_img.shape[0]}"
    enlarged_size = f"{enlarged_img.shape[1]}x{enlarged_img.shape[0]}"
    shrunk_size = f"{shrunk_img.shape[1]}x{shrunk_img.shape[0]}"

    # Calculate MSE values
    mse_enlarged = calculate_mse(original_img, enlarged_img)
    mse_shrunk = calculate_mse(original_img, shrunk_img)

    # Create a combined image with text annotations
    max_height = max(original_img.shape[0], enlarged_img.shape[0], shrunk_img.shape[0])
    combined_img = np.ones((max_height + 250,
                            original_img.shape[1] + enlarged_img.shape[1] + shrunk_img.shape[1] + 300,
                            original_img.shape[2]), dtype=np.uint8) * 255

    # Insert images into the combined image
    combined_img[100:100+original_img.shape[0], 100:100+original_img.shape[1]] = original_img
    combined_img[100:100+enlarged_img.shape[0], 150+original_img.shape[1]:150+original_img.shape[1]+enlarged_img.shape[1]] = enlarged_img
    combined_img[100:100+shrunk_img.shape[0], 200+original_img.shape[1]+enlarged_img.shape[1]:200+original_img.shape[1]+enlarged_img.shape[1]+shrunk_img.shape[1]] = shrunk_img

    # Add text with image sizes
    cv2.putText(combined_img, original_size, (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(combined_img, enlarged_size, (150+original_img.shape[1], 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(combined_img, shrunk_size, (200+original_img.shape[1]+enlarged_img.shape[1], 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    # Add text with MSE values
    mse_text = f"MSE (Enlarged): {mse_enlarged:.4f}"
    mse_text_shrunk = f"MSE (Shrunk): {mse_shrunk:.4f}"
    cv2.putText(combined_img, mse_text, (100, max_height + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(combined_img, mse_text_shrunk, (100, max_height + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    # Save the combined image to a file
    folder = os.path.dirname(os.path.abspath(__file__))
    filename = f"{method}_combined_img{img_number}.png"
    filepath = os.path.join(folder, filename)
    cv2.imwrite(filepath, combined_img)

# Read input images
img1 = cv2.imread("Input1_256x256.png")
img2 = cv2.imread("Input2_640x635.jpg")

# Run tests for both images
run_tests(img1, 1)
run_tests(img2, 2)