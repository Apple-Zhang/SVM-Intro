import numpy as np
from numpy.core.numeric import Inf
import scipy.spatial.distance as spd
import cvxopt as cvx

cvx.solvers.options['show_progress'] = False

class BinarySVM:
    def __init__(self, c=1.0, kernel="lin", gamma=None, d=None):
        self._c = c
        self._kernel = kernel
        self._gamma = gamma
        self._d = d
        self.model = None
    
    def _kernel_deal(self, X1: np.ndarray, X2: np.ndarray):
        """
        Compute kernel dot product: K = \phi(X1)' @ \phi(X2) = K(X1, X2)

        input:
        @X1, X2: two matrix with same #cols

        return: K = K(X1, X2) according to the kernel type in self._kernel
        """
        if self._kernel == "lin":
            return X1 @ X2.T
        elif self._kernel == "rbf":
            Kmat = spd.cdist(X1, X2)
            Kmat = np.power(Kmat, 2)
            Kmat = np.exp(-self._gamma * Kmat)
            return Kmat
        elif self._kernel == "poly":
            return np.power(X1 @ X2.T + self._gamma, self._d)
        else:
            print("Error: Unknwon kernel. Only 'lin', 'rbf', 'poly' are supported.")
            return None

    def _qpsolver(self, Qmat: np.ndarray, y: np.ndarray):
        """
        Solve the quadratic programming problem (QPP) of SVM with cvxopt module.
        
        input:
        @Qmat: the Q matrix of QPP in SVM
        @y: the label array

        return: the solution of the QPP
        """
        n_sample = y.shape[0]
        P = cvx.matrix(Qmat)
        q = cvx.matrix(-np.ones(n_sample))
        A = cvx.matrix(y, (1, n_sample))
        b = cvx.matrix(0.0)

        if self._c >= Inf:
            # G is a sparse matrix of idendity
            # h = zeros
            G = cvx.spmatrix(-1.0, range(n_sample), range(n_sample))
            h = cvx.matrix(np.zeros(n_sample))
        else:
            # G is a sparse matrix of [-I; I]
            # h = [zeros; c*ones]
            all_ones = np.ones(n_sample)
            ones_elements = np.hstack((-all_ones, all_ones))
            ones_r_index  = list(range(n_sample << 1))
            ones_c_index  = list(range(n_sample))
            ones_c_index.extend(ones_c_index)
            G = cvx.spmatrix(ones_elements, ones_r_index, ones_c_index)
            h = cvx.matrix(np.hstack((np.zeros(n_sample), self._c * all_ones)), (2*n_sample, 1))
        return cvx.solvers.qp(P, q, G, h, A, b)

    def bsvm_pred(self, Xt: np.ndarray, Y: np.ndarray):
        """
        Binary class SVM prediction

        input:
        @X: testing instances. Each row is an sample.
        @Y: the label of testing instances, 1-d array with binary value.
        
        return: (pred, pred_label, accu), where pred is the prediction value, pred_label is 
                the prediction label, and accu denotes the accuracy.
        """
        if len(Y.shape) > 1:
            print("Error: only scale label is supported. Y is supposed be a 1-dim array.")
            return None, None, None
        if Xt.shape[0] != Y.shape[0]:
            print("Error: the 1st dim of Xt should match that of Y.")
            return None, None, None
        predK = self._kernel_deal(Xt, self.model["sv"])
        predY = predK @ self.model["sv_alpha"] + self.model["b"]

        pred_label = np.zeros(Y.shape)
        pred_label[predY >= 0] = self.model["labelA"]
        pred_label[predY <  0] = self.model["labelB"]
        accu = 100 * np.count_nonzero(pred_label == Y) / Y.shape[0]

        return predY, np.sign(predY), accu

    def bsvm_train(self, X: np.ndarray, Y: np.ndarray, epsilon=1e-7):
        """
        Binary class SVM training. Using 1v1 training.

        input:
        @X: training instances. Each row is an sample.
        @Y: the label of training instances, 1-d array with binary value.
        
        return: trained SVM model
        """
        if len(Y.shape) > 1:
            print("Error: only scale label is supported. Y is supposed be a 1-dim array.")
            return None
        if X.shape[0] != Y.shape[0]:
            print("Error: the 1st dim of X should match that of Y.")
            return None
        
        labelt = np.unique(Y)
        n_sample = X.shape[0]

        if len(labelt) != 2:
            print("Error: the input should be binary class.")
            return None
        
        # set label
        model = {"labelA": np.max(Y), "labelB": np.min(Y)}
        Y = np.sign(Y - (labelt[0] + labelt[1]) / 2)
        squareY = np.outer(Y, Y)

        # solve the quadratic progaming problem
        Qmat = squareY * self._kernel_deal(X, X)
        solution = self._qpsolver(Qmat, Y)
        
        # get alpha and support vector index
        alpha = np.ravel(solution["x"])
        sv_ind = alpha > epsilon

        # fill model
        sv   = X[sv_ind]
        sv_y = Y[sv_ind]
        model["sv_ind"] = sv_ind
        model["sv_alpha"] = alpha[sv_ind] * sv_y
        temp = self._kernel_deal(sv, sv)

        # compute bias with support vector
        model["sv"] = sv
        model["b"]  = np.mean(sv_y - temp @ model["sv_alpha"])

        # assign the model
        self.model = model
        return model

class MulticlassSVM:
    def __init__(self, c=1.0, kernel="lin", gamma=None, d=None):
        self._c = c
        self._kernel = kernel
        self._gamma = gamma
        self._d = d
        self._nclass = 0
        self.models = []

    def msvm_train(self, X: np.ndarray, Y: np.ndarray):
        """
        Multiclass SVM training. Using 1v1 training.

        input:
        @X: training instances. Each row is an sample.
        @Y: the label of training instances, 1-d array with discrete value.
        
        return: the set of models
        """
        if len(Y.shape) > 1:
            print("Error: only scale label is supported. Y is supposed be a 1-dim array.")
            return None
        if X.shape[0] != Y.shape[0]:
            print("Error: the 1st dim of X should match that of Y.")
            return None
        
        # find the number of class
        cate = np.unique(Y)
        self._nclass = len(cate)

        # 1 vs 1 training, each sub-training is a binary SVM
        for i in range(self._nclass):
            for j in range(i+1, self._nclass):
                Yij = Y[Y == cate(i) or Y == cate(j)]
                Xij = X[Y == cate(i) or Y == cate(j)]
                bsvm = BinarySVM(self._c, self._kernel, self._gamma, self._d)
                model = bsvm.bsvm_train(Xij, Yij)
                self.models.append(model)
        return self.models

    def msvm_pred(self, Xt: np.ndarray, Y: np.ndarray):
        """
        Multiclass SVM prediction

        input:
        @X: testing instances. Each row is an sample.
        @Y: the label of testing instances, 1-d array with discrete value.
        
        return: (pred, accu), where pred is the prediction label and accu denotes the accuracy.
        """
        if len(Y.shape) > 1:
            print("Error: only scale label is supported. Y is supposed be a 1-dim array.")
            return None, None
        if Xt.shape[0] != Y.shape[0]:
            print("Error: the 1st dim of Xt should match that of Y.")
            return None, None

        n_sample = Xt.shape[0]
        cate = np.zeros(n_sample, self._nclass)

        # initialize a binary SVM model for inference only
        bsvm = BinarySVM(self._c, self._kernel, self._gamma, self._d)
        for i in range(len(self.models)):
            bsvm.model = self.models[i]
            # vote for each sample
            _, pred, _ = bsvm.bsvm_pred(Xt, Y)
            for j in range(n_sample):
                cate[j, pred[j]] += 1

        # select winner and compute accuracy
        pred = np.max(cate, axis=1)
        accu = 100 * np.count_nonzero(pred == Y) / len(Y)
        return pred, accu

def generate_lin_toy(point_num: int):
    # a linear separable case
    x  = 6 * np.random.rand(point_num, 2) - 3
    y1 = x[:, 0] + x[:, 1] <  0.6
    y2 = x[:, 0] + x[:, 1] >= 1.2
    x1 = x[y1]
    x2 = x[y2]
    y = np.hstack((y1[y1 == 1]+1, y2[y2 == 1]))
    x = np.vstack((x1, x2))
    return x, y

def generate_ker_toy(point_num: int):
    # a circle data
    x  = 6 * np.random.rand(point_num, 2) - 3
    y1 = x[:, 0]*x[:, 0] + x[:, 1]*x[:, 1] <  4
    y2 = x[:, 0]*x[:, 0] + x[:, 1]*x[:, 1] >= 5.5
    x1 = x[y1]
    x2 = x[y2]
    y = np.hstack((y1[y1 == 1]+1, y2[y2 == 1]))
    x = np.vstack((x1, x2))
    return x, y

if __name__ == "__main__":
    import time
    
    # linear case
    x, y = generate_lin_toy(250)
    t_start = time.time()
    svm  = BinarySVM()
    svm.bsvm_train(x, y)
    _, _, acc = svm.bsvm_pred(x, y)
    t_end = time.time()
    t = t_end - t_start
    print(f"Elapsed time is {t} secs.")
    print(f"Accuracy: {acc}%")

    # nonlinear case
    x, y = generate_ker_toy(250)
    t_start = time.time()
    svm  = BinarySVM(kernel="poly", d=2, gamma=0)
    svm.bsvm_train(x, y)
    _, _, acc = svm.bsvm_pred(x, y)
    t_end = time.time()
    t = t_end - t_start
    print(f"Elapsed time is {t} secs.")
    print(f"Accuracy: {acc}%")