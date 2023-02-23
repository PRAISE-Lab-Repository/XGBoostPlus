from re import L
import cvxpy as cvx
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_validate, RepeatedStratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_curve, auc

import warnings
warnings.filterwarnings("ignore")


def splitDataUCI(X, Y, X_indices, X_priv_indicies):
    Y[Y == 0] = -1
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
    Y[Y == 0] = -1
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

    for trial in range(50):
        print(f'Running Trial #{trial+1}')

        X_train_priv, X_train, X_test_priv, X_test, y_train, y_test = None, None, None, None, None, None
        if datasetName in ['diabetes', 'cancer', 'wine', 'abalone']:
            X, Y, X_indices, X_priv_indicies, _ = data
            X_train_priv, X_train, X_test_priv, X_test, y_train, y_test = splitDataUCI(X, Y, X_indices, X_priv_indicies)
        else:
            X_priv, X, Y = data
            X_train_priv, X_train, X_test_priv, X_test, y_train, y_test = splitSynthData(X_priv, X, Y)

        mdl = SVMPlus(datasetName)
        params = mdl.hyper_parameters()
            # for p in params:
        success = mdl.train(X_train, y_train, x_star=X_train_priv, params = {'gamma_w': 1.0, 'gamma_w_star': 0.1})

        if success:
            preds, preds_probas = mdl.predict(X_test)

            recall, precision, specificity, f1, g_mean, acc = metrics(y_test, preds)
            fpr, tpr, thresholds = roc_curve(y_test, preds_probas, drop_intermediate = False, pos_label=1)
            auc1 = auc(fpr, tpr)

            # result = [y_test, preds_probas]
            # f = open(output_dir + 'trail_{}.pkl'.format(trail_id), 'wb')
            # pickle.dump(result, f)
            # f.close()

            print(f'Recall: {recall}\nPrecision: {precision}\nSpecificity: {specificity}\nF1: {f1}\nG_mean: {g_mean}\nACC: {acc}\nAUC: {auc1}\n')

            spec_list.append(specificity)
            recall_list.append(recall)
            precision_list.append(precision)
            f1_list.append(f1)
            g_mean_list.append(g_mean)
            acc_list.append(acc)
            auc_list.append(auc1)

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


class SVMPlus:

    def __init__(self, datasetName):

        # model name
        self.name = 'svm+'
        self.linestyle = '-'
        self.color = 'r'

        self.datasetName = datasetName

        # prediction decision threshold
        self.pred_thresh = 0

        # model parameters
        self.w = None
        self.b = None

    def hyper_parameters(self):

        # generate hyper-parameter test space
        gamma = np.logspace(-8, 2, 11)

        # return list of parameter dictionaries
        p = []
        for i in range(len(gamma)):
            for j in range(len(gamma)):
                p.append({'gamma_w': gamma[i], 'gamma_w_star': gamma[j]})

        return p

    def train(self, x, y, params=None, x_star=None):

        # default parameters
        if params is None:
            params = {'gamma_w': 1e-3,
                      'gamma_w_star': 1e-3}

        # ensure labels are [-1, 1]
        y[y == 0] = -1
        assert np.unique(y).tolist() == [-1, 1]

        # ensure y is a m x 1 vector
        if len(y.shape) < 2:
            y = np.expand_dims(y, axis=-1)
        assert y.shape[1] == 1

        # if x* not supplied just make it ones
        if x_star is None:
            x_star = np.ones([x.shape[0], 1])

        # regularization parameter
        gamma_w = params['gamma_w'] / x.shape[1]
        gamma_w_star = params['gamma_w_star'] / x_star.shape[1]

        # define model variables
        w = cvx.Variable((x.shape[1], 1))
        b = cvx.Variable()
        w_star = cvx.Variable((x_star.shape[1], 1))
        d = cvx.Variable()

        # compute slack
        u = x_star * w_star - d

        # balanced slack term
        slack = cvx.sum(cvx.multiply(u, y + 1)) / cvx.sum(y + 1) + \
                cvx.sum(cvx.multiply(u, y - 1)) / cvx.sum(y - 1)

        # define objective
        obj = cvx.Minimize(gamma_w * cvx.sum_squares(w) + gamma_w_star * cvx.sum_squares(w_star) + slack)

        # define constraints
        constraints = [cvx.multiply(y, (x * w - b)) >= 1 - u,
                       u >= 0]

        # form problem and solve
        prob = cvx.Problem(obj, constraints)

        # ‘ECOS’, ‘SCS’, or ‘OSQP’.
        if self.datasetName == 'cancer':
            prob.solve(verbose=False, solver = 'SCS' )
        else:
            prob.solve(verbose=False, solver = 'ECOS' )

        try:

            # throw error if not optimal
            assert prob.status == 'optimal'

            # save model parameters
            self.w = np.array(np.squeeze(w.value, axis=-1))
            self.b = np.array(b.value)

            # return success
            return True

        except:

            # return failure
            return False

    def predict(self, x):

        # make prediction
        y_prob  = x @ self.w - self.b - self.pred_thresh
        y_hat = np.squeeze(np.sign(y_prob))
        

        return y_hat, y_prob

def metrics(y_valid, y_predict):

    # compute number of true positives

    tp = np.sum((y_valid == +1) * (y_predict == +1))
    fp = np.sum((y_valid == -1) * (y_predict == +1))
    fn = np.sum((y_valid == +1) * (y_predict == -1))
    tn = np.sum((y_valid == -1) * (y_predict == -1))
    # compute recall
    recall = tp / (tp + fn)
    #DEFINE https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
    # compute precision, recall, f1
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

