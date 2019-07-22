import numpy as np
from sklearn.metrics.pairwise import rbf_kernel,linear_kernel,laplacian_kernel
from cvxopt import matrix, solvers

class TKL:
    def __init__(self,source_x,target_x,gamma=None,eta=2.0,kernel='linear'):
        self.source_x = source_x
        self.target_x = target_x
        self.gamma = gamma
        self.eta = eta
        self.kernel = kernel

    def tkl(self):
        if self.kernel == 'linear':
            K_Z = linear_kernel(self.source_x, self.source_x)
            K_X = linear_kernel(self.target_x, self.target_x)
            K_ZX = linear_kernel(self.source_x, self.target_x)

        elif self.kernel == 'rbf':
            K_Z = rbf_kernel(self.source_x, self.source_x,self.gamma)
            K_X = rbf_kernel(self.target_x, self.target_x,self.gamma)
            K_ZX = rbf_kernel(self.source_x, self.target_x,self.gamma)

        else:
            K_Z = laplacian_kernel(self.source_x, self.source_x,self.gamma)
            K_X = laplacian_kernel(self.target_x, self.target_x,self.gamma)
            K_ZX = laplacian_kernel(self.source_x, self.target_x,self.gamma)



        #Calculate the Kernel matrix for the source and the target space
        K_Z = (K_Z + K_Z.transpose()) / 2
        print(K_Z)
        K_X = (K_X + K_X.transpose()) / 2

        labda_X , phi_X = np.linalg.eig(K_X)
        labda_X = np.diag(labda_X)
        for i in range(len(labda_X)):
            labda_X[i][i] = 1./labda_X[i][i]

        #print(labda_X.shape,phi_X)
        phi_Z = np.dot(np.dot(K_ZX,phi_X),labda_X)

        phi_A = np.concatenate((phi_Z,phi_X),axis=0)

        Q = np.dot(phi_Z.transpose(),phi_Z) * np.dot(phi_Z.transpose(),phi_Z)
        Q = (Q + Q.transpose()) / 2
        r = - np.diag(np.dot(np.dot(phi_Z.transpose(),K_Z),phi_Z)).reshape(len(self.target_x),1)
        I = np.diag(np.ones(len(self.target_x)))#n * n
        I_ = np.diag(np.ones(len(self.target_x)-1),1) # n * n

        C = - I + self.eta * I_

        C = np.concatenate((C,-np.ones((len(self.target_x),len(self.target_x)))),axis=0)
        h = np.zeros((len(self.target_x)*2,1))

        P = matrix(Q)
        q = matrix(r)
        G = matrix(C)
        h = matrix(h)
        labda = solvers.qp(P,q,G,h)
        labda = labda['x']
        labda = np.array(labda)
        labda_temp = np.zeros(shape=(len(labda),len(labda)))
        for i in range(len((labda))):
            labda_temp[i][i] = labda[i]

        labda = labda_temp
        #K = np.dot(np.dot(phi_A,labda),phi_A.transpose())

        #K = (K + K.transpose()) / 2

        train_space = np.dot(np.dot(phi_Z,labda),phi_Z.transpose())
        test_space = np.dot(np.dot(phi_X,labda),phi_Z.transpose())

        return train_space , test_space







