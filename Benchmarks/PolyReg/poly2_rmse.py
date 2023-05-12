import math
import numpy as np

x_test = [-10, -8.94737, -7.89474, -6.8421, -5.78947, -4.73684, -3.68421, -2.63158, -1.57895, -0.526316, 0.526316,
 1.57895, 2.63158, 3.68421, 4.73684, 5.78947, 6.84211, 7.89474, 8.94737, 10]

y_test = [2399.87, 1911.16, 1479.28, 1102.17, 780.253, 513.596, 302.175, 147.68, 46.889, 1.70888, 12.3189, 
78.9571, 199.534, 375.746, 608.969, 896.316, 1238.49, 1637.55, 2090.63, 2599.93]

def fixed(x):
    return 24.99 * x * x + 9.47 * x


def float_(x):
    return 25.01 * x * x + 10.03 * x


def double_(x):
    return 24.99* x * x + 9.94 * x

def original(x):
    return 25 * x * x + 10 * x


def rmse(x,y, func):
    #squared error
    sqe = [(func(x[i]) - y[i])**2 for i in range(len(x))]
    return math.sqrt(sum(sqe) / len(x))


if __name__ == "__main__":
    # x_test = [2,2.5, 3,3.5]
    # y_test = [23.5, 37, 54,74]

    # x_test = [2,2.2,2.4,3]
    # y_test = [original(x) + np.random.normal(-0.25,0.25,1) for x in x_test]

    print("MSE for fixed: ", rmse(x_test, y_test, fixed))
    print("MSE for float: ", rmse(x_test, y_test, float_))
    print("MSE for double: ", rmse(x_test, y_test, double_))
    print("MSE for original: ", rmse(x_test, y_test, original))