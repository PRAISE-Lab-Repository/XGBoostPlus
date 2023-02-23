from __future__ import division, print_function
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
import numpy as np

# experiment 1: noiseless labels as privileged info
def synthetic_01(a,n):
    x  = np.random.randn(n,a.size)
    e  = (np.random.randn(n))[:,np.newaxis]
    xs = np.dot(x,a)[:,np.newaxis]
    y  = ((xs+e) > 0).ravel()
    return (xs,x,y)

class Loss(object):
    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()

    def acc(self, y, y_pred):
        return 0
        
class CrossEntropy(Loss):
    def __init__(self): pass

    def loss(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def acc(self, y, p):
        return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))

    def gradient(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)

class GradientBoosting(object):
    def __init__(self, n_estimators, learning_rate, min_samples_split, max_depth=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
        self.loss = CrossEntropy()

        # Initialize regression trees
        self.trees = []
        for _ in range(n_estimators):
            tree = DecisionTreeRegressor(
                    min_samples_split=self.min_samples_split,
                    max_depth=self.max_depth)
            self.trees.append(tree)
        
        
    def fit(self, X, Xs, y, c1 = 0.01, c2 = 0.01):
        y_pred = np.full(np.shape(y), np.mean(y, axis=0))
        self.weights_priv = np.random.rand(1, Xs.shape[1])

        for i in range(self.n_estimators):
            gradient = self.loss.gradient(y, y_pred)
            gradient_star = gradient + ((c1/(c1+1)) * self.weights_priv @ Xs.transpose())[0]

            self.trees[i].fit(X, gradient_star)
            update = self.trees[i].predict(X)

            A = update - gradient

            self.weights_priv = (c1/(c1+c2)) * ((A.transpose()@Xs) @ (np.linalg.inv(Xs.transpose()@Xs)))

            # Update y prediction
            y_pred -= np.multiply(self.learning_rate, update)

    def predict(self, X):
        y_pred = np.array([])
        # Make predictions
        for tree in self.trees:
            update = tree.predict(X)
            update = np.multiply(self.learning_rate, update)
            y_pred = -update if not y_pred.any() else y_pred - update

        # Turn into probability distribution
        y_pred = np.exp(y_pred) / (1 + np.exp(y_pred))
        y_pred = (y_pred > 0.5).astype(int) 
        return y_pred



if __name__ == "__main__":

    d      = 10
    n_tr   = 200
    n_te   = 1000
    n_reps = 100
    a   = np.random.randn(d)
    (xs_tr,x_tr,y_tr) = synthetic_01(a,n=n_tr)
    (xs_te,x_te,y_te) = synthetic_01(a,n=n_te)
    
    clf = GradientBoosting(100, 0.1, 2)
    clf.fit(x_tr, xs_tr, y_tr, c1=0.01, c2=0.01)
    print("Accuracy: ", (clf.predict(x_te) == y_te).mean())