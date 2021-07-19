from ops import deblur, denoise
from numpy import genfromtxt
from utils import misc_utils
import numpy as np
given_data = genfromtxt('data.csv', delimiter=',')

x = given_data[1:, 0]
y = given_data[1:, 1]

x = np.array(x)
y = np.array(y)
h = np.array([1/16, 4/16, 6/16, 4/16, 1/16])

misc_utils.graph_plot(x, y)
p = denoise(y, 5)
misc_utils.graph_plot(x, p)
q = deblur(p, h)
misc_utils.graph_plot(x, q)
