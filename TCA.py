"""
Transfer Component Analysis
"""
import numpy as np
import scipy.io
import scipy.linalg
from sklearn.metrics.pairwise import rbf_kernel,linear_kernel,laplacian_kernel

class TCA:
    def __init__(self, kernel='linear', dim=30, lamb=1, gamma=1):
        '''
        Init func
        :param kernel_type: kernel
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        '''
        self.kernel = kernel
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

    def fit(self, Xs, Xt):
        '''
        Transform Xs and Xt
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: Xs_new and Xt_new after TCA
        '''
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))

        if self.kernel == 'linear':
            K = linear_kernel(X.T)


        elif self.kernel == 'rbf':
            K = rbf_kernel(X.T,None,self.gamma)

        else:
            K = laplacian_kernel(X.T,None,self.gamma)
        n_eye = n
        a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        a = np.linalg.inv(a)

        w, V = scipy.linalg.eig(np.dot(a,b))
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        Z = np.dot(A.T, K)
        Z /= np.linalg.norm(Z, axis=0)
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        return Xs_new, Xt_new