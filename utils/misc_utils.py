"""
Plotting the input signal, output signal and the denoised signal
using matplotlib library for better visualisation
"""
import matplotlib.pyplot as plt


def graph_plot(x, y):
    """
    for plotting graph with two inputs( same size required as of x-axis)
    """
    x_axis = [i for i in range(len(x))]
    # plt.scatter(parameter_on_x, parameter_on_y, size of dot, color of dot)
    plt.scatter(x_axis, x, s=1, c='blue')
    plt.scatter(x_axis, y, s=1, c='red')
    plt.xlim(0, 200)  # for limitting the value of x axis
    plt.ylim(0, 50)  # for limitting the value of y axis
    plt.show()
