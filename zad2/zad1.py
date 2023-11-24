import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as scp
from sklearn import metrics

N = 100

def sin(x):
    return np.sin(x)

def sinx(x):
    return np.sin(1/x)

def sin8x(x):
    return np.sign(np.sin(8*x))

method = 'linear'

def interpolate_and_plot(x, y, method, filename):
    interpolated_function = scp.interp1d(x, y, kind=method)
    x_int = np.linspace(x.min(), x.max(), N)
    y_int = interpolated_function(x_int)
    y_true = sin(x_int)

    print(f"MSE: {metrics.mean_squared_error(y_pred=y_int, y_true=y_true):.8f}")

    plt.plot(x, y, 'go', label='Original Points')
    plt.plot(x_int, y_int, 'r-', label='Interpolation')
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
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
    interpolate_and_plot(x, y_values, 'linear', filename)


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

x = np.linspace(-5,5,300)
_ = plt.figure(figsize=[10,5])
y1 = []
y2 = []
y3 = []
y4 = []

for x0 in x:
    y1.append(h1(x0))
    y2.append(h2(x0))
    y3.append(h3(x0))
    y4.append(h4(x0))
_ = plt.plot(x,y1)

for i, y in enumerate([y1, y2, y3], start=1):
    interpolated_function = scp.interp1d(x, y, kind=method)
    x_int = np.linspace(x.min(), x.max(), 10 * N)
    y_int = interpolated_function(x_int)

    plt.plot(x, y, 'go', label=f'Orginal Points {i}')
    plt.plot(x_int, y_int, 'r-', label=f'Interpolation {i}')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'h{i}.png')
    plt.clf()
    #plt.show()
