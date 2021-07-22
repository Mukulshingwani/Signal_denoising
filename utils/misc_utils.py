"""
Plotting the input signal, output signal and the denoised signal
using matplotlib library for better visualisation
"""
import matplotlib.pyplot as plt
import numpy as np


def graph_plot(x: np.ndarray, y: np.ndarray, ylim=50, title: str = "",
               legend_1: str = "", legend_2: str = ""):
    """
    for plotting graph with two inputs( same size required as of x-axis)
    """
    x, y = np.nan_to_num(x), np.nan_to_num(y)
    x_axis = [i for i in range(len(x))]
    # plt.scatter(parameter_on_x, parameter_on_y, size of dot, color of dot)
    plt.scatter(x_axis, x, s=1, c='blue', label=legend_1)
    plt.scatter(x_axis, y, s=1, c='red', label=legend_2)
    plt.xlim(0, 200)  # for limitting the value of x axis
    plt.ylim(0, ylim)  # for limitting the value of y axis
    plt.title(title)
    plt.legend()
    plt.show()
