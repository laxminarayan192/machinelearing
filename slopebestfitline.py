from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

x = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
y = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)


def create_datasets(hm, varients, step=2, correlation=False):
    return hm


def best_fit_line(x, y):
    m = (((mean(x) * mean(y)) - mean(x * y)) / ((mean(x) * mean(x)) - mean(x * x)))
    b = mean(y) - m * mean(x)
    return m, b


def squared_error(y_origin, y_line):
    return sum((y_origin - y_line) ** 2)


def coeficent_of_determination(y_origin, y_line):
    y_men_line = [mean(y_origin) for y in y_origin]
    squared_error_reg = squared_error(y_origin, y_line)
    squared_error_y_mean = squared_error(y_origin, y_men_line)
    return 1 - (squared_error_reg / squared_error_y_mean)


m, b = best_fit_line(x, y)

line = [(m * t) + b for t in x]

pred_x = 8
pred_y = (m * pred_x) + b

print(coeficent_of_determination(y, line))
plt.scatter(x, y)
plt.plot(line)
plt.show()
