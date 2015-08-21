__author__ = 'Abhinav'

import matplotlib.pyplot as pyplot
import matplotlib.mlab as mlab
import numpy as np

def plot_countour(data):
    delta = 0.25
    mean_1 = np.arange(-1, 4.25, delta)
    mean_2 = np.arange(-1, 4.25, delta)

    X, Y = np.meshgrid(mean_1, mean_2)
    log_likelihood = np.zeros((mean_1.shape[0], mean_2.shape[0]))
    for i in range(len(mean_1)):
        for j in range(len(mean_2)):
            temp = np.log(mlab.normpdf(data, mean_1[i], 1)*0.5 + mlab.normpdf(data, mean_2[j], 1)*0.5)
            log_likelihood[i, j] = sum(temp)

    countour = pyplot.contourf(X, Y, log_likelihood)
    pyplot.colorbar(countour, shrink=0.8, extend='both')
    pyplot.savefig('./countour.png')


def main():
    #load dataset
    data = np.loadtxt('hw5.data', dtype=float)
    plot_countour(data)

if __name__ == '__main__':
    main()