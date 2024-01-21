import numpy as np
from skimage import io, transform
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import os

def save_images(images, titles, suptitle, rows, cols, name, folder="output_images"):
    num_images = len(images)
    rows = min(rows, num_images)
    cols = min(cols, num_images)
    
    fig = plt.figure(figsize=(cols*4, rows*4))
    fig.suptitle(suptitle)
    
    for i in range(num_images):
        plt.subplot(rows, cols, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        
    plt.tight_layout()
    file_path = os.path.join(folder, f"{name}.png")
    plt.savefig(file_path)
    plt.close(fig)

def calculate_mse(img1, img2):
    return np.square(np.subtract(img1, img2)).mean()

# Demozaikowanie obrazów korzystając z interpolacji dla filtru Bayera
def demosaic_Bayer_interpolation(image,type="average"):
    height, width, c = image.shape

<<<<<<< HEAD
    mosaiking = Mosaicing_Bayer(image)

    red_channel, green_channel, blue_channel = mosaiking[:,:,0], mosaiking[:,:,1], mosaiking[:,:,2]

=======
    red_channel, green_channel, blue_channel = np.zeros((height, width)), np.zeros((height, width)), np.zeros((height, width))
    
>>>>>>> e263a40e69e6e9db70a3d9dbcfcee3ed70774f24
    # 1.  Interpolacja koloru czerwonego w wierszach nieparzystych
    for i in range(1,height,2):
        for j in range(1,width,2):
            neighbors = [(i - 1, j), (i + 1, j)]        # Górny i dolny sąsiad
            
            valid_neighbors = [(x, y) for x, y in neighbors if 0 <= x < height and 0 <= y < width]

            if valid_neighbors:
                if type == "average":
                    red_channel[i, j] = np.mean([red_channel[x, y] for x, y in valid_neighbors])
                elif type == "max":
                    red_channel[i, j] = np.max([red_channel[x, y] for x, y in valid_neighbors])

    # 2. Interpolacja koloru czerwonego w kolumnach
    for i in range(height):
        for j in range(0,width,2):
            neighbors = [(i, j - 1), (i, j + 1)]        # Prawy i lewy sąsiad
            
            valid_neighbors = [(x, y) for x, y in neighbors if 0 <= x < height and 0 <= y < width]

            if valid_neighbors:
                if type == "average":
                    red_channel[i, j] = np.mean([red_channel[x, y] for x, y in valid_neighbors])
                elif type == "max":
                    red_channel[i, j] = np.max([red_channel[x, y] for x, y in valid_neighbors])                

    # 3.  Interpolacja koloru zielonego w kolumnach
    for i in range(0, height, 2):
        for j in range(1, width, 2):
            neighbors = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]        # Wszyscy sąsiedzi

            valid_neighbors = [(x, y) for x, y in neighbors if 0 <= x < height and 0 <= y < width]

            if valid_neighbors:
                if type == "average":
                    green_channel[i, j] = np.mean([green_channel[x, y] for x, y in valid_neighbors])
                elif type == "max":
                    green_channel[i, j] = np.max([green_channel[x, y] for x, y in valid_neighbors])
    
    for i in range(1, height, 2):
        for j in range(0, width, 2):
            neighbors = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]        # Wszyscy sąsiedzi

            valid_neighbors = [(x, y) for x, y in neighbors if 0 <= x < height and 0 <= y < width]

            if valid_neighbors:
                if type == "average":
                    green_channel[i, j] = np.mean([green_channel[x, y] for x, y in valid_neighbors])
                elif type == "max":
                    green_channel[i, j] = np.max([green_channel[x, y] for x, y in valid_neighbors])
   
    # 4. Interpolacja koloru niebieskiego w wierszach parzystych
    for i in range(0,height,2):
        for j in range(0,width,2):
            neighbors = [(i - 1, j), (i + 1, j)]

            valid_neighbors = [(x, y) for x, y in neighbors if 0 <= x < height and 0 <= y < width]

            if valid_neighbors:
                if type == "average":
                    blue_channel[i, j] = np.mean([blue_channel[x, y] for x, y in valid_neighbors])
                elif type == "max":
                    blue_channel[i, j] = np.max([blue_channel[x, y] for x, y in valid_neighbors])

    # 5. Interpolacja koloru niebieskiego w kolumnach
    for i in range(height):
        for j in range(1,width,2):
            neighbors = [(i, j - 1), (i, j + 1)]

            valid_neighbors = [(x, y) for x, y in neighbors if 0 <= x < height and 0 <= y < width]

            if valid_neighbors:
                if type == "average":
                    blue_channel[i, j] = np.mean([blue_channel[x, y] for x, y in valid_neighbors])
                elif type == "max":
                    blue_channel[i, j] = np.max([blue_channel[x, y] for x, y in valid_neighbors])


    result = np.dstack((red_channel,green_channel,blue_channel))
    return result    

<<<<<<< HEAD
# Demozaikowanie obrazów korzystając z konwolucji 2D dla filtru Bayera
def demosaic_Bayer_convolution(image):
    bayer_mask = np.array([[[1, 1], [1, 1]], [[1/2, 1/2], [1/2, 1/2]], [[1, 1], [1, 1]]])
    
    result = np.dstack([cv2.filter2D(image[:, :, i], -1, bayer_mask[i]) for i in range(3)])
    return result
=======
def mosaic_Bayer()
    for i in range(height):
        for j in range(width):
            if i % 2 == 0:                              # wiersz nieparzysty
                if j % 2 == 0:                          # kolumna nieparzysta
                    green_channel[i,j] = image[i,j,1]
                else:                                   # kolumna parzysta
                    red_channel[i,j] = image[i,j,0] 
            else:                                       # wiersz parzysty
                if j % 2 == 0:                          # kolumna nieparzysta
                    blue_channel[i,j] = image[i,j,2]
                else:                                   # kolumna parzysta
                    green_channel[i,j] = image[i,j,1]


# Porównanie jakości demozaikowania metodą konwolucji 2D i interpolacji
def Bayer_test():
    paths = [r"Bayer/circle.npy",r"Bayer/milky-way.npy",r"Bayer/mond.npy",r"Bayer/namib.npy",r"Bayer/pandas.npy"]
    for path in paths:
        image = np.load(path)
        
        image1 = demosaic_Bayer_interpolation(image, type="average")        
        image2 = demosaic_Bayer_interpolation(image, type="max")
        image3 = demosaic_Bayer_convolution(image)
        
        mse1 = round(calculate_mse(image,image1),5)
        mse2 = round(calculate_mse(image,image2),5)
        mse3 = round(calculate_mse(image,image3),5)
>>>>>>> e263a40e69e6e9db70a3d9dbcfcee3ed70774f24

# Demozaikowanie obrazów dowolną metodą dla filtru Fuji (konwolucja)
def demosaic_Fuji_convolution(image):
    fuji_mask = np.array([[[1/8]*6]*6, [[1/20]*6]*6, [[1/8]*6]*6])

    result = np.dstack([cv2.filter2D(image[:, :, i], -1, fuji_mask[i]) for i in range(3)])
    return result

# Mozaikowanie metodą Bayera
def Mosaicing_Bayer(image):
    height, width, c = image.shape
    image = transform.resize(image, output_shape=(height,width,c))
    
    red = np.zeros((height,width), dtype=np.uint8)
    red[::2, 1::2] = 1

    green = np.zeros((height,width), dtype=np.uint8)
    green[::2,::2] = 1
    green[1::2, 1::2] = 1

    blue = np.zeros((height,width), dtype=np.uint8)
    blue[1::2,::2] = 1
    
    bayer = np.dstack((red,green,blue))

    return bayer * image

# Mozaikowanie metodą Fuji
def Mosaicing_Fuji(image):
    pattern = np.array([[[0,0,1,0,0,1],[1,0,0,0,0,0],[0,0,0,1,0,0],[0,0,1,0,0,1],[1,0,0,0,0,0],[0,0,0,1,0,0]],
                        [[1,0,0,1,0,0],[0,1,1,0,1,1],[0,1,1,0,1,1],[1,0,0,1,0,0],[0,1,1,0,1,1],[0,1,1,0,1,1]],
                        [[0,1,0,0,1,0],[0,0,0,1,0,0],[1,0,0,0,0,0],[0,1,0,0,1,0],[0,0,0,1,0,0],[1,0,0,0,0,0]]])

    height, width, c = image.shape
    rows_repeat, cols_repeat = height // 6 + 1, width // 6 + 1

    Mask = np.dstack([np.tile(pattern[i],(rows_repeat,cols_repeat))[:height, :width] for i in range(c)])

    return Mask * image

# Porównanie jakości demozaikowania metodą interpolacji i konwolucji 2D
def Demosaic_test():
    paths = [r"Fuji/circle.npy",r"Fuji/milky-way.npy",r"Fuji/mond.npy",r"Fuji/namib.npy",r"Fuji/panda.npy"]
    for path in paths:
        image = np.load(path)
        
        Bayer_image = Mosaicing_Bayer(image)
        Fuji_image = Mosaicing_Fuji(image)

        image1 = demosaic_Bayer_interpolation(Bayer_image, type="average")        
        image2 = demosaic_Bayer_interpolation(Bayer_image, type="max")
        image3 = demosaic_Bayer_convolution(Bayer_image)
        image4 = demosaic_Fuji_convolution(Fuji_image)

        mse1 = round(calculate_mse(image,image1),5)
        mse2 = round(calculate_mse(image,image2),5)
        mse3 = round(calculate_mse(image,image3),5)
        mse4 = round(calculate_mse(image,image4),5)

        images = [Bayer_image, image1, image2, Fuji_image, image3,  image4]
        titles = ["Obraz z maską Bayera",  f"Interpolacja (average pooling) \n MSE: {mse1}", f"Interpolacja (max pooling) \n MSE: {mse2}", 
                  "Obraz z maską Fuji", f"Konwolucja Bayera \n MSE: {mse3}",f"Konwolucja Fuji \n MSE: {mse4}"]
        name = f"Demozaikowanie_{path.split('/')[-1].split('.')[0]}"
        save_images(images, titles, "", 2, 3, name)


# Zaimplementować 3 podane zastosowania konwolucji, każde za pomocą jednego filtru.
# 1. Wykrywanie krawędzi:
def Laplace(image):
    Laplace = np.array([[0,1,0], [1,-4,1], [0,1,0]])

    result = np.dstack([cv2.filter2D(image[:,:,i],-1,Laplace) for i in range(3)])
    return result

# BONUS: Zaimplementować detektor Sobela, złożony z sumy detekcji w osiach X oraz Y
def Sobel(image):
    Sobel_X= np.array([[1,0,-1], [2,0,-2], [1,0,-1]])
    Sobel_Y= np.array([[1,2,1], [0,0,0], [-1,-2,-1]])

    result_X = np.dstack([cv2.filter2D(image[:,:,i],-1,Sobel_X) for i in range(3)])
    result_Y = np.dstack([cv2.filter2D(image[:,:,i],-1,Sobel_Y) for i in range(3)])

    result_XY = result_X + result_Y
    return result_XY

# BONUS: Zaimplementować dodatkowe filtry krawędzi (operator Sobela-Feldmana, operator Scharra lub Operator Prewitta)
def Scharr(image):
    Scharr_X = np.array([[-3,0,3], [-10,0,10], [-3,0,3]])
    Scharr_Y = np.array([[-3,-10,-3], [0,0,0], [3,10,3]])

    result_X = np.dstack([cv2.filter2D(image[:,:,i],-1,Scharr_X) for i in range(3)])
    result_Y = np.dstack([cv2.filter2D(image[:,:,i],-1,Scharr_Y) for i in range(3)])

    result_XY = result_X + result_Y
    return result_XY        

def Prewitt(image):
    Prewitt_X = np.array([[-1,0,1], [-1,0,1], [-1,0,1]])
    Prewitt_Y = np.array([[1,1,1], [0,0,0], [-1,-1,-1]])
    
    result_X = np.dstack([cv2.filter2D(image[:,:,i],-1,Prewitt_X) for i in range(3)])
    result_Y = np.dstack([cv2.filter2D(image[:,:,i],-1,Prewitt_Y) for i in range(3)])

    result_XY = result_X + result_Y
    return result_XY

# 2. Rozmycie:
def Gaussian(image):
    kernel = (1/16) * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])

    result = np.dstack([cv2.filter2D(image[:,:,i],-1,kernel) for i in range(3)])
    return result

# BONUS: Zaimplementować rozmycie Gaussowskie z większym rozmiarem jądra, 5 na 5 lub 7 na 7.
def Gaussian_blur(image, kernel_size):
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*(kernel_size/2)**2)) * np.exp(-((x-(kernel_size-1)/2)**2 + (y-(kernel_size-1)/2)**2) / (2.0*(kernel_size/2)**2)),
        (kernel_size, kernel_size)
    )

    kernel = kernel / np.sum(kernel)

    result = np.dstack([cv2.filter2D(image[:,:,i],-1,kernel) for i in range(3)])
    return result

def Average_Blurr(image,size):
    filter = np.ones((size,size)) / size ** 2

    result = np.dstack([cv2.filter2D(image[:,:,i],-1,filter) for i in range(3)])
    return result

# 3. Wyostrzanie:
def Sharpen(image):
    Sharp = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])

    result = np.dstack([cv2.filter2D(image[:,:,i],-1,Sharp) for i in range(3)])
    return result

# BONUS: Zaimplementować dodatkowe filtry wyostrzające.
def Unsharp_masking(image, kernel_size, strength=1.5):
    blurred = Gaussian_blur(image, kernel_size)

    result = image + strength * (image - blurred)
    return result

# Porównanie metod wykrywania krawędzi
def Detection_test():
    paths = [r"Colors.jpg", r"Fuji/panda.npy", r"Fuji/circle.npy"]
    for path in paths:
        if path == r"Colors.jpg":
            image = cv2.imread(path)
        else:
            image = np.load(path)
        
        Lap = Laplace(image)
        Sob = Sobel(image)
        Pre = Prewitt(image)
        Sch = Scharr(image)

        images = [image, Lap, Sob, Pre, Sch]
        titles = ["Oryginalny obraz","Operator Laplace'a", "Operator Sobela", "Operator Prewitta", "Operator Scharra"]
        name = f"Wykrywanie_krawędzi_{path.split('/')[-1].split('.')[0]}"
        save_images(images, titles, "", 2, 3, name)

# Porównanie metod rozmywających
def Blurr_test():
    paths = [r"Colors.jpg", r"Fuji/panda.npy", r"Fuji/circle.npy"]
    for path in paths:
        if path == r"Colors.jpg":
            image = cv2.imread(path)
        else:
            image = np.load(path)

        Gauss = Gaussian(image)
        Gauss_5x5 = Gaussian_blur(image, 5)
        Gauss_7x7 = Gaussian_blur(image, 7)
        Blurr = Average_Blurr(image,10)

        images = [image, Gauss, Gauss_5x5, Gauss_7x7, Blurr]
        titles = ["Oryginalny obraz", "Rozmycie Gaussowskie 3x3", "Rozmycie Gaussowskie 5x5", 
                  "Rozmycie Gaussowskie 7x7", "Rozmycie uśredniające 10x10"]
        name = f"Rozmywanie_{path.split('/')[-1].split('.')[0]}"
        save_images(images, titles, "", 2, 3, name)

# Porównanie metod wyostrzania
def Sharpen_test():
    paths = [r"Fuji/panda.npy", r"Fuji/milky-way.npy", r"Fuji/circle.npy"]
    for path in paths:
        image = np.load(path)

        Sh = Sharpen(image)    
        Un = Unsharp_masking(image, 5)

        images = [image, Sh, Un]
        titles = ["Orginalny obraz","Wykorzystując jądro","Wyostrzanie odwrotne"]
        name = f"Wyotrzanie_{path.split('/')[-1].split('.')[0]}"
        save_images(images, titles, "", 1, 3, name)

Demosaic_test()
Detection_test()
Blurr_test()
Sharpen_test()