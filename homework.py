import pandas as pd
import numpy as np
import math as mt
import matplotlib as matp
import matplotlib.pyplot as plt


def gaussianDist(X, mean, sig):
    y = []
    for i in X:
        a = ((i - mean) ** 2) / (2 * (sig ** 2))
        e = np.exp(-a)
        y.append(1 / (sig * mt.sqrt(2 * mt.pi)) * e)
    return y


def gix_plot(arr, u0, sig0, u1, sig1):
    y = []
    for i in arr:
        a0 = ((i - u0) ** 2) / (2 * (sig0 ** 2))
        e0 = np.exp(-a0)

        a1 = ((i - u1) ** 2) / (2 * (sig1 ** 2))
        e1 = np.exp(-a1)

        p0 = (1 / (sig0 * mt.sqrt(2 * mt.pi)) * e0)
        p1 = (1 / (sig1 * mt.sqrt(2 * mt.pi)) * e1)

        y.append(p0 / (p0 + p1))
    return y


def gixold_plot(arr, u, sig, pCi):
    y = []
    for i in arr:
        a = ((i - u) ** 2) / (2 * (sig ** 2)) - mt.log(sig)
        c = 0
        y.append(c - a)
    return y


def calculations(X, y):
    m0 = np.mean(X[y == 0])
    m1 = np.mean(X[y == 1])

    var0 = np.var(X[y == 0])
    var1 = np.var(X[y == 1])
    s0 = mt.sqrt(var0)
    s1 = mt.sqrt(var1)

    pC0 = np.sum(y == 0) / len(y)
    pC1 = np.sum(y == 1) / len(y)

    return m0, m1, var0, var1, s0, s1, pC0, pC1


def import_data():
    df = pd.read_csv('hmw2_data.csv', sep=';')

    X = df.iloc[:, 0:1].values
    y = df.iloc[:, 1].values
    input_l = len(X)
    X = X.reshape(input_l, 1)
    y = y.reshape(input_l, 1)
    return X, y, input_l


def plot(X, y, m0, m1, var0, var1, s0, s1):
    # plot data
    plt.title("Dataset")
    plt.scatter(X, y, c=y)
    plt.xlabel("$x$")
    plt.ylabel("y")
    plt.show()

    plt.title("Likelihood of Dataset")
    y0 = gaussianDist(X, m0, s0)
    y1 = gaussianDist(X, m1, s1)
    plt.plot(X, y0)
    plt.plot(X, y1)
    plt.show()

    plt.title("Posteriors of Dataset")
    g0 = gix_plot(X, m0, s0, m1, s1)
    g1 = gix_plot(X, m1, s1, m0, s0)
    plt.plot(X, g0)
    plt.plot(X, g1)
    plt.show()

    x = np.linspace(-3, 3, 1000)

    y0 = gaussianDist(x, m0, s0)
    y1 = gaussianDist(x, m1, s1)

    plt.title("Likelihood of Smoothed Data")
    plt.plot(x, y0)
    plt.plot(x, y1)
    plt.show()

    d = 2 * var0 * var1 * (mt.log(s1) - mt.log(s0))
    a = var1 - var0
    b = 2 * (var0 * m1 - var1 * m0)
    c = var1 * m0 ** 2 - var0 * m1 ** 2 - d

    delta = (b ** 2) - (4 * a * c)
    sol1 = (-b - mt.sqrt(delta)) / (2 * a)
    sol2 = (-b + mt.sqrt(delta)) / (2 * a)

    plt.title("Posteriors of Smoothed Data")
    g0 = gix_plot(x, m0, s0, m1, s1)
    g1 = gix_plot(x, m1, s1, m0, s0)
    plt.plot(x, g0, label="Class0")
    plt.plot(x, g1, label="Class1")
    plt.axvline(sol1, color="g", label="Min. for 0")
    plt.axvline(sol2, color="r", label="Max. for 0")
    plt.axis([0.75, 1.75, 0, 1])
    plt.legend()
    plt.show()


X, y, length = import_data()

mean0, mean1, var0, var1, s0, s1, pc0, pc1 = calculations(X, y)

plot(X, y, mean0, mean1, var0, var1, s0, s1)
