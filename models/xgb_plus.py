from math import e
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_validate, RepeatedStratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_curve, auc

import warnings
warnings.filterwarnings("ignore")


def splitDataUCI(X, Y, X_indices, X_priv_indicies):
    # Y[Y == 0] = -1
    X_train, X_test, y_train, y_test = train_test_split(X.values, Y.values.reshape(-1), test_size=0.2) # Splited data into train and test

    # print((X_train))

    X_train_priv = X_train[:, X_priv_indicies]
    X_train = X_train[:,X_indices]

    X_test_priv = X_test[:,X_priv_indicies]
    X_test = X_test[:,X_indices]

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    return X_train_priv, X_train, X_test_priv, X_test, y_train, y_test 

def splitSynthData(X_priv, X, Y):
    Y = pd.DataFrame(Y.astype(int))
    # Y[Y == 0] = -1
    X = np.hstack((X,X_priv))

    X_train, X_test, y_train, y_test = train_test_split(X, Y.values.reshape(-1), test_size=0.2) # Splited data into train and test

    X_train_priv = X_train[:,24:]
    X_train = X_train[:,:24]

    X_test_priv = X_test[:,24:]
    X_test = X_test[:,:24]

    return X_train_priv, X_train, X_test_priv, X_test, y_train, y_test 

def run_model(data, datasetName):
    spec_list  = [] # specificity 
    recall_list = [] # recall (sensitivity)
    precision_list = [] # precision 
    f1_list = [] # f1
    g_mean_list = [] # g_mean
    acc_list = [] # accuracy 
    auc_list = [] # AUC score
    weights_priv = []

    for trial in range(50):
        print(f'Running Trial #{trial+1}')

        X_train_priv, X_train, X_test_priv, X_test, y_train, y_test = None, None, None, None, None, None
        priv_columns = None
        if datasetName in ['diabetes', 'cancer', 'wine', 'abalone']:
            X, Y, X_indices, X_priv_indicies, priv_columns = data
            X_train_priv, X_train, X_test_priv, X_test, y_train, y_test = splitDataUCI(X, Y, X_indices, X_priv_indicies)
        else:
            X_priv, X, Y = data
            X_train_priv, X_train, X_test_priv, X_test, y_train, y_test = splitSynthData(X_priv, X, Y)

        estimator = XGBoostClassifier()
        _ = estimator.fit(X_train, X_train_priv, y_train, c1=0.1, c2=0.1, depth=3, learning_rate=0.01, boosting_rounds=50)
        
        preds, preds_probas, weights_priv_temp = estimator.predict(X_test)

        recall, precision, specificity, f1, g_mean, acc = metrics(y_test, preds)
        fpr, tpr, thresholds = roc_curve(y_test, preds_probas, drop_intermediate = False, pos_label=1)
        auc1 = auc(fpr, tpr)

        print(f'Recall: {recall}\nPrecision: {precision}\nSpecificity: {specificity}\nF1: {f1}\nG_mean: {g_mean}\nACC: {acc}\nAUC: {auc1}\n')

        spec_list.append(specificity)
        recall_list.append(recall)
        precision_list.append(precision)
        f1_list.append(f1)
        g_mean_list.append(g_mean)
        acc_list.append(acc)
        auc_list.append(auc1)
        weights_priv.append(weights_priv_temp)
    
    print('AVERAGE SPECIFICITY')
    print(f'Mean of Specificity: {np.mean(spec_list)}')
    print(f'Best of Specificity: {np.max(spec_list)}')
    print(f'Std of Specificity: {np.std(spec_list)}')

    print('AVERAGE AUC')
    print(f'Mean of AUC: {np.mean(auc_list)}')
    print(f'Best of AUC: {np.max(auc_list)}')
    print(f'Std of AUC: {np.std(auc_list)}')

    print('AVERAGE G-MEAN')
    print(f'Mean of G-Mean: {np.mean(g_mean_list)}')
    print(f'Best of G-Mean: {np.max(g_mean_list)}')
    print(f'Std of G-Mean: {np.std(g_mean_list)}')

    print('AVERAGE ACC')
    print(f'Mean of Accuracy: {np.mean(acc_list)}')
    print(f'Best of Accuracy: {np.max(acc_list)}')
    print(f'Std of Accuracy: {np.std(acc_list)}')

    print('AVERAGE SENSITIVITY')
    print(f'Mean of Sensitivity: {np.mean(recall_list)}')
    print(f'Best of Sensitivity: {np.max(recall_list)}')
    print(f'Std of Sensitivity: {np.std(recall_list)}')

    avg_weights_priv = np.mean(weights_priv, axis = 0)
    norm = np.linalg.norm(avg_weights_priv)
    normal_array = avg_weights_priv/norm

    print('NORMALIZED WEIGHTS')
    print(f'{priv_columns}')
    print(f'{normal_array}')


def metrics(y_valid, y_predict):

    # compute number of true positives

    tp = np.sum((y_valid == +1) * (y_predict == +1))
    fp = np.sum((y_valid == 0) * (y_predict == +1))
    fn = np.sum((y_valid == +1) * (y_predict == 0))
    tn = np.sum((y_valid == 0) * (y_predict == 0))
    # compute recall
    recall = tp / (tp + fn)

    if tp == 0 and fp == 0 and fn == 0:
        precision = 1
        recall = 1
        f1 = 1
    elif tp == 0:
        precision = 0
        recall = 0
        f1 = 0
    else :
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        # compute f1 score
        f1 = 2 / (recall ** -1 + precision ** -1)

    # compute specificity
    specificity = tn / (tn + fp)

    # compute g mean
    g_mean = np.sqrt(specificity * recall)

    acc = (tp+ tn) / len(y_valid)

    return recall, precision, specificity, f1, g_mean, acc

class Node:
    
    '''
    A node object that is recursivly called within itslef to construct a regression tree. Based on Tianqi Chen's XGBoost 
    the internal gain used to find the optimal split value uses both the gradient and hessian. Also a weighted quantlie sketch 
    and optimal leaf values all follow Chen's description in "XGBoost: A Scalable Tree Boosting System" the only thing not 
    implemented in this version is sparsity aware fitting or the ability to handle NA values with a default direction.
    Inputs
    ------------------------------------------------------------------------------------------------------------------
    x: pandas datframe of the training data
    gradient: negative gradient of the loss function
    hessian: second order derivative of the loss function
    idxs: used to keep track of samples within the tree structure
    subsample_cols: is an implementation of layerwise column subsample randomizing the structure of the trees
    (complexity parameter)
    min_leaf: minimum number of samples for a node to be considered a node (complexity parameter)
    min_child_weight: sum of the heassian inside a node is a meaure of purity (complexity parameter)
    depth: limits the number of layers in the tree
    lambda: L2 regularization term on weights. Increasing this value will make model more conservative.
    gamma: This parameter also prevents over fitting and is present in the the calculation of the gain (structure score). 
    As this is subtracted from the gain it essentially sets a minimum gain amount to make a split in a node.
    eps: This parameter is used in the quantile weighted skecth or 'approx' tree method roughly translates to 
    (1 / sketch_eps) number of bins
    Outputs
    --------------------------------------------------------------------------------------------------------------------
    A single tree object that will be used for gradient boosintg.
    '''

    def __init__(self, x, gradient, hessian, idxs, subsample_cols = 0.8 , min_leaf = 5, min_child_weight = 1 ,depth = 10, lambda_ = 1, gamma = 1, eps = 0.1):
      
        self.x, self.gradient, self.hessian = x, gradient, hessian
        self.idxs = idxs 
        self.depth = depth
        self.min_leaf = min_leaf
        self.lambda_ = lambda_
        self.gamma  = gamma
        self.min_child_weight = min_child_weight
        self.row_count = len(idxs)
        self.col_count = x.shape[1]
        self.subsample_cols = subsample_cols
        self.eps = eps
        self.column_subsample = np.random.permutation(self.col_count)[:round(self.subsample_cols*self.col_count)]

        self.val = self.compute_gamma(self.gradient[self.idxs], self.hessian[self.idxs])
          
        self.score = float('-inf')
        self.find_varsplit()
        
        
    def compute_gamma(self, gradient, hessian):
        '''
        Calculates the optimal leaf value equation (5) in "XGBoost: A Scalable Tree Boosting System"
        '''
        return(-np.sum(gradient)/(np.sum(hessian) + self.lambda_))
        
    def find_varsplit(self):
        '''
        Scans through every column and calcuates the best split point.
        The node is then split at this point and two new nodes are created.
        Depth is only parameter to change as we have added a new layer to tre structure.
        If no split is better than the score initalised at the begining then no splits further splits are made
        '''
        for c in self.column_subsample: self.find_greedy_split(c)
        if self.is_leaf: return
        x = self.split_col
        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]
        self.lhs = Node(x = self.x, gradient = self.gradient, hessian = self.hessian, idxs = self.idxs[lhs], min_leaf = self.min_leaf, depth = self.depth-1, lambda_ = self.lambda_ , gamma = self.gamma, min_child_weight = self.min_child_weight, eps = self.eps, subsample_cols = self.subsample_cols)
        self.rhs = Node(x = self.x, gradient = self.gradient, hessian = self.hessian, idxs = self.idxs[rhs], min_leaf = self.min_leaf, depth = self.depth-1, lambda_ = self.lambda_ , gamma = self.gamma, min_child_weight = self.min_child_weight, eps = self.eps, subsample_cols = self.subsample_cols)
        
    def find_greedy_split(self, var_idx):
        '''
         For a given feature greedily calculates the gain at each split.
         Globally updates the best score and split point if a better split point is found
        '''
        x = self.x[self.idxs, var_idx]
        
        for r in range(self.row_count):
            lhs = x <= x[r]
            rhs = x > x[r]
            
            lhs_indices = np.nonzero(x <= x[r])[0]
            rhs_indices = np.nonzero(x > x[r])[0]
            if(rhs.sum() < self.min_leaf or lhs.sum() < self.min_leaf 
               or self.hessian[lhs_indices].sum() < self.min_child_weight
               or self.hessian[rhs_indices].sum() < self.min_child_weight): continue

            curr_score = self.gain(lhs, rhs)
            if curr_score > self.score: 
                self.var_idx = var_idx
                self.score = curr_score
                self.split = x[r]
                
    def weighted_qauntile_sketch(self, var_idx):
        '''
        XGBOOST Mini-Version
        Yiyang "Joe" Zeng
        Is an approximation to the eact greedy approach faster for bigger datasets wher it is not feasible
        to calculate the gain at every split point. Uses equation (8) and (9) from "XGBoost: A Scalable Tree Boosting System"
        '''
        x = self.x[self.idxs, var_idx]
        hessian_ = self.hessian[self.idxs]
        df = pd.DataFrame({'feature':x,'hess':hessian_})
        
        df.sort_values(by=['feature'], ascending = True, inplace = True)
        hess_sum = df['hess'].sum() 
        df['rank'] = df.apply(lambda x : (1/hess_sum)*sum(df[df['feature'] < x['feature']]['hess']), axis=1)
        
        for row in range(df.shape[0]-1):
            # look at the current rank and the next ran
            rk_sk_j, rk_sk_j_1 = df['rank'].iloc[row:row+2]
            diff = abs(rk_sk_j - rk_sk_j_1)
            if(diff >= self.eps):
                continue
                
            split_value = (df['rank'].iloc[row+1] + df['rank'].iloc[row])/2
            lhs = x <= split_value
            rhs = x > split_value
            
            lhs_indices = np.nonzero(x <= split_value)[0]
            rhs_indices = np.nonzero(x > split_value)[0]
            if(rhs.sum() < self.min_leaf or lhs.sum() < self.min_leaf 
               or self.hessian[lhs_indices].sum() < self.min_child_weight
               or self.hessian[rhs_indices].sum() < self.min_child_weight): continue
                
            curr_score = self.gain(lhs, rhs)
            if curr_score > self.score: 
                self.var_idx = var_idx
                self.score = curr_score
                self.split = split_value
                
    def gain(self, lhs, rhs):
        '''
        Calculates the gain at a particular split point bases on equation (7) from
        "XGBoost: A Scalable Tree Boosting System"
        '''
        gradient = self.gradient[self.idxs]
        hessian  = self.hessian[self.idxs]
        
        lhs_gradient = gradient[lhs].sum()
        lhs_hessian  = hessian[lhs].sum()
        
        rhs_gradient = gradient[rhs].sum()
        rhs_hessian  = hessian[rhs].sum()
        
        gain = 0.5 *( (lhs_gradient**2/(lhs_hessian + self.lambda_)) + (rhs_gradient**2/(rhs_hessian + self.lambda_)) - ((lhs_gradient + rhs_gradient)**2/(lhs_hessian + rhs_hessian + self.lambda_))) - self.gamma
        return(gain)
                
    @property
    def split_col(self):
        '''
        splits a column 
        '''
        return self.x[self.idxs , self.var_idx]
                
    @property
    def is_leaf(self):
        '''
        checks if node is a leaf
        '''
        return self.score == float('-inf') or self.depth <= 0                 

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])
    
    def predict_row(self, xi):
        if self.is_leaf:
            return(self.val)

        node = self.lhs if xi[self.var_idx] <= self.split else self.rhs
        return node.predict_row(xi)

    
class XGBoostTree:
    '''
    Wrapper class that provides a scikit learn interface to the recursive regression tree above
    
    Inputs
    ------------------------------------------------------------------------------------------------------------------
    x: pandas datframe of the training data
    gradient: negative gradient of the loss function
    hessian: second order derivative of the loss function
    idxs: used to keep track of samples within the tree structure
    subsample_cols: is an implementation of layerwise column subsample randomizing the structure of the trees
    (complexity parameter)
    min_leaf: minimum number of samples for a node to be considered a node (complexity parameter)
    min_child_weight: sum of the heassian inside a node is a meaure of purity (complexity parameter)
    depth: limits the number of layers in the tree
    lambda: L2 regularization term on weights. Increasing this value will make model more conservative.
    gamma: This parameter also prevents over fitting and is present in the the calculation of the gain (structure score). 
    As this is subtracted from the gain it essentially sets a minimum gain amount to make a split in a node.
    eps: This parameter is used in the quantile weighted skecth or 'approx' tree method roughly translates to 
    (1 / sketch_eps) number of bins
    
    Outputs
    --------------------------------------------------------------------------------------------------------------------
    A single tree object that will be used for gradient boosintg.
    
    '''
    def fit(self, x, gradient, hessian, subsample_cols = 0.8 , min_leaf = 5, min_child_weight = 1 ,depth = 10, lambda_ = 1, gamma = 1, eps = 0.1):
        self.dtree = Node(x, gradient, hessian, np.array(np.arange(len(x))), subsample_cols, min_leaf, min_child_weight, depth, lambda_, gamma, eps)
        return self
    
    def predict(self, X):
        return self.dtree.predict(X)
   
   
class XGBoostClassifier:
    '''
    Full application of the XGBoost algorithm as described in "XGBoost: A Scalable Tree Boosting System" for 
    Binary Classification.
    Inputs
    ------------------------------------------------------------------------------------------------------------------
    x: pandas datframe of the training data
    gradient: negative gradient of the loss function
    hessian: second order derivative of the loss function
    idxs: used to keep track of samples within the tree structure
    subsample_cols: is an implementation of layerwise column subsample randomizing the structure of the trees
    (complexity parameter)
    min_leaf: minimum number of samples for a node to be considered a node (complexity parameter)
    min_child_weight: sum of the heassian inside a node is a meaure of purity (complexity parameter)
    depth: limits the number of layers in the tree
    lambda: L2 regularization term on weights. Increasing this value will make model more conservative.
    gamma: This parameter also prevents over fitting and is present in the the calculation of the gain (structure score). 
    As this is subtracted from the gain it essentially sets a minimum gain amount to make a split in a node.
    eps: This parameter is used in the quantile weighted skecth or 'approx' tree method roughly translates to 
    (1 / sketch_eps) number of bins
    Outputs
    --------------------------------------------------------------------------------------------------------------------
    A single tree object that will be used for gradient boosintg.
    '''
    def __init__(self):
        self.estimators = []
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    # first order gradient logLoss
    def grad(self, preds, labels):
        preds = self.sigmoid(preds)
        return(preds - labels)
    
    # second order gradient logLoss
    def hess(self, preds, labels):
        preds = self.sigmoid(preds)
        return(preds * (1 - preds))
    
    @staticmethod
    def log_odds(column):
        binary_yes = np.count_nonzero(column == 1)
        binary_no  = np.count_nonzero(column == 0)
        return(np.log(binary_yes/binary_no))
    
    
    def fit(self, X, X_priv, y, subsample_cols = 0.8 , min_child_weight = 1, depth = 5, min_leaf = 5, learning_rate = 0.4, boosting_rounds = 5, lambda_ = 1.5, gamma = 1, eps = 0.1, c1 = 0.01, c2 = 0.01):
        self.X, self.X_priv, self.y = X, X_priv, y
        self.depth = depth
        self.subsample_cols = subsample_cols
        self.eps = eps
        self.min_child_weight = min_child_weight 
        self.min_leaf = min_leaf
        self.learning_rate = learning_rate
        self.boosting_rounds = boosting_rounds 
        self.lambda_ = lambda_
        self.gamma  = gamma

        
        self.base_pred = np.full((X.shape[0], 1), np.bincount(y).argmax()).flatten().astype('float64')

        self.weights_priv = np.random.rand(1, self.X_priv.shape[1])

        for booster in range(self.boosting_rounds):

            # print(booster)
            # TODO: incorporate IPL here
            
            
            r_priv = self.base_pred + ((c1/(c1+1)) * (self.weights_priv@self.X_priv.transpose())).reshape(-1)

            Grad = self.grad(r_priv, self.y)
            Hess = self.hess(r_priv, self.y)

            boosting_tree = XGBoostTree().fit(self.X, Grad, Hess, depth = self.depth, min_leaf = self.min_leaf, lambda_ = self.lambda_, gamma = self.gamma, eps = self.eps, min_child_weight = self.min_child_weight, subsample_cols = self.subsample_cols)
            
            A = boosting_tree.predict(self.X) - (self.y - self.base_pred)

            self.weights_priv = (c1/(c1+c2)) * ((A.transpose()@self.X_priv) @ (np.linalg.inv(self.X_priv.transpose()@self.X_priv)))

            self.base_pred += self.learning_rate * boosting_tree.predict(self.X)
            
            self.estimators.append(boosting_tree)

        return self.weights_priv

    def predict_proba(self, X):
        pred = np.zeros(X.shape[0])
        
        for estimator in self.estimators:
            pred += self.learning_rate * estimator.predict(X) 
          
        return(self.sigmoid(np.full((X.shape[0], 1), 1).flatten().astype('float64') + pred))
    
    def predict(self, X):
        pred = np.zeros(X.shape[0])
        for estimator in self.estimators:
            pred += self.learning_rate * estimator.predict(X) 
        
        predicted_probas = self.sigmoid(np.full((X.shape[0], 1), 1).flatten().astype('float64') + pred)
        preds = np.where(predicted_probas > np.mean(predicted_probas), 1, 0)
        return (preds, predicted_probas, self.weights_priv)

