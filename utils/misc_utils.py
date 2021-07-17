"""
Plotting the input signal, output signal and the denoised signal
using matplotlib library for better visualisation
"""
import matplotlib.pyplot as plt
import numpy as np


def graph_plot(x, y):
    """
    for plotting graph with two inputs( same size required as of x-axis)
    """
    x_axis = [i for i in range(len(x))]
    # plt.scatter(parameter_on_x, parameter_on_y, size of dot, color of dot)
    plt.scatter(x_axis, x, s=1, c='blue')
    plt.scatter(x_axis, y, s=1, c='red')
    plt.show()


graph_plot(np.random.randint(1, 1000, 100), np.random.randint(10, 450, 100))
