__author__ = 'Abhinav'

import numpy as np


def expectation_maximization(data, initial_mean, initial_class_prior):
    instance_class_probability = np.zeros((data.shape[0], initial_mean.shape[0]))
    # Initialize mean and class prior
    mean = initial_mean
    class_prior = initial_class_prior

    mean_convergence_threshold = np.ones_like(mean)
    class_prior_convergence_threshold = np.ones_like(class_prior)
    no_of_iter = 0

    print "Executing Expectation-Maximization"
    print "Mean convergence threshold:", 1e-4
    print "Class prior convergence threshold:", 1e-4
    print "*********************************************"

    # Till convergence condition satisfies
    while np.greater_equal(mean_convergence_threshold, 1e-4).any() or\
            np.greater_equal(class_prior_convergence_threshold,1e-4).any() :
        no_of_iter += 1
        # ****************************************Expectation Step***********************************
        # get Piy or class probability for each instance
        for i in range(len(initial_mean)):
            instance_class_probability[:, i] =\
                np.exp((-0.5)*(np.power(np.tile(mean[i], data.shape[0]) - data, 2)))*np.tile(class_prior[i], data.shape[0])

        total_ins_class_prob = np.sum(instance_class_probability, axis=1)

        instance_class_probability = instance_class_probability / np.column_stack((total_ins_class_prob, total_ins_class_prob))

        # update previous values
        prev_class_prior = np.copy(class_prior)
        prev_mean = np.copy(mean)
        total_ins_class_prob = np.sum(instance_class_probability, axis=1)

        # ****************************************Maximization Step***********************************
        for i in range(len(initial_mean)):
            mean[i] = np.sum(data*instance_class_probability[:, i])/np.sum(instance_class_probability[:, i])
            class_prior[i] = np.sum(instance_class_probability[:, i])/np.sum(total_ins_class_prob)

        # print results for this iteration
        print "Iteration:", no_of_iter
        print "Previous mean:", prev_mean
        print "New mean:", mean
        print '------------------------------------------'
        print "Previous class prior:", prev_class_prior
        print "New class prior:", class_prior
        print "*********************************************"

        # update thresholds
        mean_convergence_threshold = np.abs(prev_mean - mean)
        class_prior_convergence_threshold = np.abs(prev_class_prior - class_prior)

    print "Final mean:", mean
    print "Final class prior", class_prior
    print "Total no. of iteration", no_of_iter

    return mean, class_prior

def main():
    #load dataset
    data = np.loadtxt('hw5.data', dtype=float)
    expectation_maximization(data, np.array([1, 2], float), np.array([0.33, 0.67], float))



if __name__ == '__main__':
    main()