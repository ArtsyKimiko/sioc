import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as scp
from scipy.interpolate import interp1d
from sklearn import metrics
import os

N = 100

def sin(x):
    return np.sin(x)

def sinx(x):
    return np.sin(1/x)

def sin8x(x):
    return np.sign(np.sin(8*x))

method = 'linear'

def interpolate(x, y, method, filename):
    interpolated_function = scp.interp1d(x, y, kind=method)
    x_int = np.linspace(x.min(), x.max(), N)
    y_int = interpolated_function(x_int)
    y_true = np.sin(x_int)

    print(f"MSE: {metrics.mean_squared_error(y_pred=y_int, y_true=y_true):.8f}")

    plt.plot(x, y, 'go', label='Original Points')
    plt.plot(x_int, y_int, 'r-', label='Interpolation')
    plt.grid(True)
    plt.legend()

    script_dir = os.path.dirname(__file__)
    save_path = os.path.join(script_dir, filename)
    plt.savefig(save_path)
    
    plt.clf()
    plt.show()

x = np.linspace(-np.pi, np.pi, N)
y = [sin(alfa) for alfa in x]
y_sinx = [sinx(alfa) for alfa in x]
y_sin8x = [sin8x(alfa) for alfa in x]

functions = [sin, sinx, sin8x]
filenames = ['sinx.png', 'sin1_x.png', 'sin8x.png']

for func, filename in zip(functions, filenames):
    y_values = [func(alfa) for alfa in x]
    interpolate(x, y_values, 'linear', filename)

def h1(x):
    if(x>=0 and x<1):
        return 1
    else:
        return 0
    
def h2(x):
    if(x>= - 1 / 2 and x < 1 /2 ):
        return 1
    else:
        return 0
    
def h3(x):
    if( -1 <= x <= 1):
        return 1 - abs(x)
    else:
        return 0
    
def h4(x):
    return (np.sin(x) / x)

def h_functions(h, x, method, filename):
    y = [h(x0) for x0 in x]

    _ = plt.figure(figsize=[10, 5])
    _ = plt.plot(x, y)

    interpolated_function = scp.interp1d(x, y, kind=method)
    x_int = np.linspace(x.min(), x.max(), 10 * N)
    y_int = interpolated_function(x_int)

    plt.plot(x, y, 'go', label='Orginal Points')
    plt.plot(x_int, y_int, 'r-', label='Interpolation')
    plt.grid(True)
    plt.legend()

    script_dir = os.path.dirname(__file__)
    save_path = os.path.join(script_dir, filename)
    plt.savefig(save_path)
    
    plt.clf()
    plt.show()

x = np.linspace(-5, 5, 300)

h_functions(h1, x, 'linear', 'h1.png')
h_functions(h2, x, 'linear', 'h2.png')
h_functions(h3, x, 'linear', 'h3.png')
h_functions(h4, x, 'linear', 'h4.png')

#A case where the interpolated function will not be continuous / points will not be equidistant
N = 100
x = np.linspace(-np.pi, np.pi, N)
y = np.sin(x)

remove_mask = (x >= -2) & (x <= 2)
x = x[~remove_mask]
y = y[~remove_mask]

method = 'linear'
interpolated_function = scp.interp1d(x, y, kind=method)

x_int = np.linspace(x.min(), x.max(), 10*N)
y_int = interpolated_function(x_int)
y_true = np.sin(x_int)

mse = metrics.mean_squared_error(y_pred=y_int, y_true=y_true)

plt.plot(x, y, 'go', label='Original Points')
plt.plot(x_int, y_int, 'r-', label='Interpolation')
plt.plot(x_int, y_true, 'y--', label='True Function')
plt.grid(True)
plt.legend()
plt.text(0.5, 1.05, f'MSE = {mse:.8f}', fontsize=25, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    
folder = os.path.dirname(os.path.abspath(__file__))
filename = 'without_some_points.png'
filepath = os.path.join(folder, filename)
plt.savefig(filepath)
plt.clf()
plt.show()

#Examining the impact of the number of points on the MSE indicator
N_values = [5, 10, 20, 30, 70, 100]

filenames = ['n5.png', 'n10.png', 'n20.png', 'n30.png', 'n70.png', 'n100.png']

def interpolate_and_mse(N, filenames):
    x = np.linspace(-np.pi, np.pi, N)
    y = np.sin(x)

    interpolated_function = interp1d(x, y, kind='linear')

    x_true = np.linspace(-np.pi, np.pi, 10*N)
    y_true = np.sin(x_true)

    y_interp = interpolated_function(x_true)

    mse = metrics.mean_squared_error(y_true, y_interp)
    
    print(f"N={N}: MSE = {mse:.8f}")

    plt.plot(x, y, 'go', label='Original Points')
    plt.plot(x_true, y_interp, 'r-', label='Interpolation')
    plt.grid(True)
    plt.legend()
    plt.text(0.5, 1.05, f'MSE = {mse:.8f}', fontsize=25, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    
    script_dir = os.path.dirname(__file__)
    save_path = os.path.join(script_dir, filenames[N_values.index(N)])
    plt.savefig(save_path)
    plt.clf()

for N in N_values:
    interpolate_and_mse(N,filenames)

#Comparison of interpolation with a large number of points with a sequence of interpolations generating fewer points
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 1, 4, 2])

# Interpolation with a large number of points
x_interp_large = np.linspace(min(x), max(x), 1000)
interp_function_large = interp1d(x, y, kind='linear')
y_interp_large = interp_function_large(x_interp_large)

plt.plot(x, y, 'o', label='Input Data')
plt.plot(x_interp_large, y_interp_large, label='Interpolation with Many Points')
plt.legend()

folder = os.path.dirname(os.path.abspath(__file__))
filename_large = 'with_many_points.png'
filepath_large = os.path.join(folder, filename_large)
plt.savefig(filepath_large)
plt.clf()


num_steps = 4
total_points = 2 * len(x)

# Sequential interpolation
x_interp_seq = np.linspace(min(x), max(x), len(x))
y_interp_seq = y.copy()

for _ in range(num_steps):
    interp_function_seq = interp1d(x_interp_seq, y_interp_seq, kind='linear', fill_value="extrapolate")
    x_interp_seq = np.linspace(min(x_interp_seq), max(x_interp_seq), total_points)
    y_interp_seq = interp_function_seq(x_interp_seq)


plt.plot(x, y, 'o', label='Input Data')
plt.plot(x_interp_seq, y_interp_seq, label='Sequential Interpolation')
plt.legend()

folder = os.path.dirname(os.path.abspath(__file__))
filename_seq = 'sequential.png'
filepath_seq = os.path.join(folder, filename_seq)
plt.savefig(filepath_seq)
plt.clf()