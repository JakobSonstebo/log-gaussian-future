import torch
import torch.nn as nn
from resnet import ResNet, BatchResNet
from balanced_resnet import BalancedResNet
import matplotlib.pyplot as plt
import numpy as np
import scipy.special

def sample_data(num_samples, input_dim):
    X = torch.ones((num_samples, input_dim)) / np.sqrt(input_dim)
    return X

def run_model(model, X, n, d):
    if model == 'resnet_batch':
        model = BatchResNet(10, 10, n, d)
    elif model == 'resnet':
        model = ResNet(10, 10, n, d)
    elif model == 'balanced_resnet':
        model = BalancedResNet(10, 10, n, d)
    y = model(X)
    return y

def main(N, n, d, model):
    X = sample_data(N, 10)
    y = torch.zeros((N, 10))
    with torch.no_grad():
        for i in range(N):
            y[i] = run_model(model, X[i], n, d)
    y = torch.log(torch.norm(y, dim=1)**2)
    return y

def plot_histogram(ys, ax):
    ax.hist(ys, density=True, bins=100)

def plot_infinite_width(x, ax, n_in=10, n_out=10):
    def f(x):
        return n_in**(n_out/2) / (2**(n_out / 2) * scipy.special.gamma(n_out / 2)) * np.exp(n_out*x/2)*np.exp(-n_in*np.exp(x)/2)
    
    ax.plot(x, f(x), label='Infinite width')

def J2overPi(theta):
    return (3*np.sin(theta)*np.cos(theta)+(np.pi - theta)*(1+2*(np.cos(theta)**2)))/np.pi

def plot_infinite_width_infinite_depth(x, ax, n_in=10, n_out=10):
    def f(x):
        return 1 / (2**(n_out / 2) * scipy.special.gamma(n_out / 2)) * np.exp(n_out*x/2)*np.exp(-np.exp(x)/2)
    
    def gaussian(x, mean, std):
        return 1 / (std * np.sqrt(2*np.pi)) * np.exp(-(x - mean)**2 / (2*std**2))
    
    a = 1 / np.sqrt(2)
    d = 10
    n = 100

    mean = - 1 / 2 * d / n * ( 5 * a**2 + 4 * (1 - a) * a ) + \
                    0 * 2 * a * (-0.876) * d / n - np.log(n_in)
    
    var = d*(5*(a**2) + 4*a*(1-a)) / n

    conv = np.convolve(f(x), gaussian(x, mean, np.sqrt(var)), mode='same') * (x[1] - x[0])
    ax.plot(x, conv, label='Infinite width, infinite depth')

if __name__ == '__main__':
    N = 10000
    n = 100
    d = 10
    model = 'resnet'
    ys = main(N, n, d, model)
    fig, ax = plt.subplots()
    plot_histogram(ys, ax)
    x = np.linspace(-10, 10, 1000)
    plot_infinite_width(x, ax)
    plot_infinite_width_infinite_depth(x, ax)
    plt.legend()
    plt.show()

