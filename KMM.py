import numpy as np
import math,sklearn.metrics.pairwise as sk
from sklearn import svm
from cvxopt import matrix, solvers

def computeKernelWidth(data):
    dist = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            # s = self.__computeDistanceSq(data[i], data[j])
            # dist.append(math.sqrt(s))
            dist.append(np.sqrt(np.sum((np.array(data[i]) - np.array(data[j])) ** 2)))
    return np.median(np.array(dist))

def get_beta(train_x,test_x):
    beta = []
    gamma = computeKernelWidth(train_x)
    print('gamma:',gamma)
    beta = Kernel_mean_matching(train_x,test_x,gamma)
    return beta

def Kernel_mean_matching(train_x,test_x,sigma):
    n_tr = len(train_x)
    n_te = len(test_x)

    # calculate Kernel
    print('Computing kernel for training data ...')
    K_ns = sk.rbf_kernel(train_x, train_x, sigma)
    # make it symmetric
    K = 0.9 * (K_ns + K_ns.transpose())

    # calculate kappa
    print('Computing kernel for kappa ...')
    kappa_r = sk.rbf_kernel(train_x, test_x, sigma)
    ones = np.ones(shape=(n_te, 1))
    kappa = np.dot(kappa_r, ones)
    kappa = -(float(n_tr) / float(n_te)) * kappa

    # calculate eps
    eps = (math.sqrt(n_tr) - 1) / math.sqrt(n_tr)

    # constraints
    A0 = np.ones(shape=(1, n_tr))
    A1 = -np.ones(shape=(1, n_tr))
    A = np.vstack([A0, A1, -np.eye(n_tr), np.eye(n_tr)])
    b = np.array([[n_tr * (eps + 1), n_tr * (eps - 1)]])
    b = np.vstack([b.T, -np.zeros(shape=(n_tr, 1)), np.ones(shape=(n_tr, 1)) * 1000])

    print('Solving quadratic program for beta ...')
    P = matrix(K, tc='d')
    q = matrix(kappa, tc='d')
    G = matrix(A, tc='d')
    h = matrix(b, tc='d')
    beta = solvers.qp(P, q, G, h)
    return [i for i in beta['x']]
