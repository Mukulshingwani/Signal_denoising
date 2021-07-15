from numpy import genfromtxt
my_data = genfromtxt('data.csv', delimiter=',')

x = my_data[:, 0]

y = my_data[:, 1]
