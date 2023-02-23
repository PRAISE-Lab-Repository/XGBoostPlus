import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_validate, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report,roc_auc_score # for model evaluation metrics

import h2o
# from h2o.automl import H2OAutoML

from h2o.estimators import H2OXGBoostEstimator
from sklearn.metrics import roc_curve
        # from sklearn.metrics import auc

def splitDataUCI(X, Y):
    # Y[Y == 0] = -1
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) # Splited data into train and test

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    return X_train, X_test, y_train, y_test

def splitSynthData(X, Y):
    Y = pd.DataFrame(Y.astype(int))

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) # Splited data into train and test

    return X_train, X_test, y_train, y_test

def calc_acc_sen_spec(df):

  values = df.to_list()

  tp = values[1][1]
  fp = values[0][1]
  fn = values[1][0]
  tn = values[0][0]

  precision = tp / (tp + fp)
  recall = tp / (tp + fn)
  
   # compute f1 score
  f1 = 2 / (recall ** -1 + precision ** -1)

  # compute specificity
  specificity = tn / (tn + fp)

  g_mean = np.sqrt(specificity * recall)

  accuracy  = (values[0][0] + values[1][1]) / (values[0][1] + values[0][0] + values[1][1] + values[1][0])

  fpr = fp / (values[0][0] + values[1][0])
  tpr = tp / (values[0][1] + values[1][1])

  return  specificity, accuracy, g_mean, recall, f1, fpr, tpr

def run_model(data, datasetName):
    h2o.init()

    auc_avg = []
    acc_avg = []
    acc_avg_con = []
    spec_avg = []
    sen_avg = []
    var_importance = []

    best_model = None
    best_acc = 0

    best_auc_roc_curve = {}
    g_mean_avg = []
    recall_avg = []
    f1_avg = []

    for trial in range(50):
        print(f'Running Trial #{trial+1}')

        X_train, X_test, y_train, y_test = None, None, None, None
        y_name = '25'
        
        if datasetName in ['diabetes', 'cancer', 'wine', 'abalone']:
            X, Y, y_name = data
            X_train, X_test, y_train, y_test = splitDataUCI(X, Y)
        else:
            X, Y = data
            X_train, X_test, y_train, y_test = splitSynthData( X, Y)


        train_df = pd.concat([X_train, y_train] ,axis = 1)
        test_df = pd.concat([X_test, y_test],axis = 1)

        train = h2o.H2OFrame(train_df)
        test = h2o.H2OFrame(test_df)

        y = y_name
        x = train.columns
        # print(y)
        x.remove(y)

        train[y] = train[y].asfactor()
        test[y] = test[y].asfactor()

        titanic_xgb2 = H2OXGBoostEstimator( nfolds=5,  keep_cross_validation_predictions=True)

        titanic_xgb2.train(x=x, y=y, training_frame=train)

        perf_stack_test = titanic_xgb2.model_performance(test)
        y_hat = titanic_xgb2.predict(test)
        temp = titanic_xgb2.predict(test)
        y_hat = y_hat.as_data_frame()['p1'].values

        specificity, accuracy_con, g_mean, recall, f1, _ , _ = (calc_acc_sen_spec(perf_stack_test.confusion_matrix()))

        fpr, tpr, thresholds = roc_curve(y_test, y_hat, drop_intermediate = False, pos_label=1)

        print(f'Recall: {recall}\nSpecificity: {specificity}\nG_mean: {g_mean}\nACC: {perf_stack_test.accuracy()[0][1]}\nAUC: {perf_stack_test.auc()}\n')
        

        best_auc_roc_curve[perf_stack_test.auc()] = (fpr, tpr)
        g_mean_avg.append(g_mean)
        recall_avg.append(recall)
        f1_avg.append(f1)

        acc_avg.append(perf_stack_test.accuracy()[0][1])
        auc_avg.append(perf_stack_test.auc())
        spec_avg.append(specificity)
        acc_avg_con.append(accuracy_con)

    print('AVERAGE SPECIFICITY')
    print(f'Mean of Specificity: {np.mean(spec_avg)}')
    print(f'Best of Specificity: {np.max(spec_avg)}')
    print(f'Std of Specificity: {np.std(spec_avg)}')

    print('AVERAGE AUC')
    print(f'Mean of AUC: {np.mean(auc_avg)}')
    print(f'Best of AUC: {np.max(auc_avg)}')
    print(f'Std of AUC: {np.std(auc_avg)}')

    print('AVERAGE G-MEAN')
    print(f'Mean of G-Mean: {np.mean(g_mean_avg)}')
    print(f'Best of G-Mean: {np.max(g_mean_avg)}')
    print(f'Std of G-Mean: {np.std(g_mean_avg)}')

    print('AVERAGE ACC')
    print(f'Mean of Accuracy: {np.mean(acc_avg)}')
    print(f'Best of Accuracy: {np.max(acc_avg)}')
    print(f'Std of Accuracy: {np.std(acc_avg)}')

    print('AVERAGE SENSITIVITY')
    print(f'Mean of Sensitivity: {np.mean(recall_avg)}')
    print(f'Best of Sensitivity: {np.max(recall_avg)}')
    print(f'Std of Sensitivity: {np.std(recall_avg)}')
