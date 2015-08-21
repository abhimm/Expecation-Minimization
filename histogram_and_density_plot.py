__author__ = 'Abhinav'

import matplotlib.pyplot as pyplot
import matplotlib.mlab as mlab
import numpy as np
from em_gmm import expectation_maximization

def plot_histogram():
    data = np.loadtxt('./hw5.data')

    pyplot.axis([-5, 7, 0, 100])
    pyplot.xticks([1*k for k in range(-5, 8)])

    pyplot.title("Histogram for the data")
    pyplot.xlabel("Data")
    pyplot.ylabel("Frequency")

    pyplot.hist(data, 36)
    pyplot.savefig('./histogram.png')
    pyplot.close()

def plot_density(mean, class_prior):
    x = np.linspace(-5, 7, 100)
    pyplot.xticks([1*k for k in range(-5, 8)])
    pyplot.plot(x, (mlab.normpdf(x, mean[0], 1)*class_prior[0] + mlab.normpdf(x, mean[1], 1)*class_prior[1]))
    pyplot.title("Density plot for the data")
    pyplot.xlabel("Data")
    pyplot.ylabel("Density")
    pyplot.savefig('./density_plot.png')
    pyplot.close()

def main():
    plot_histogram()

    #load dataset
    data = np.loadtxt('hw5.data', dtype=float)
    mean, class_prior= expectation_maximization(data, np.array([1, 2], float), np.array([0.33, 0.67], float))
    plot_density(mean, class_prior)


if __name__ == '__main__':
    main()