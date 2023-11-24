import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as scp
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
    y_true = sin(x_int)

    print(f"MSE: {metrics.mean_squared_error(y_pred=y_int, y_true=y_true):.8f}")

    plt.plot(x, y, 'go', label='Original Points')
    plt.plot(x_int, y_int, 'r-', label='Interpolation')
    plt.grid(True)
    plt.legend()

    script_dir = os.path.dirname(__file__)
    save_path = os.path.join(script_dir, filename)

    plt.savefig(save_path)
    plt.clf()
    #plt.show()

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
    #plt.show()

x = np.linspace(-5, 5, 300)

h_functions(h1, x, 'linear', 'h1.png')
h_functions(h2, x, 'linear', 'h2.png')
h_functions(h3, x, 'linear', 'h3.png')
h_functions(h4, x, 'linear', 'h4.png')