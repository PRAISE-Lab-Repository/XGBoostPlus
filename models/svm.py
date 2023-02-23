from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_validate, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report,roc_auc_score # for model evaluation metrics

import pandas as pd
import numpy as np

# from sklearn.model_selection import train_test_split, cross_validate, RepeatedStratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import auc


def splitDataUCI(X, Y):
    # Y[Y == 0] = -1
    X = X.fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X.values, Y.values.reshape(-1), test_size=0.2) # Splited data into train and test

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    return X_train, X_test, y_train, y_test

def splitSynthData(X, Y):
    Y = pd.DataFrame(Y.astype(int))
    

    X_train, X_test, y_train, y_test = train_test_split(X, Y.values.reshape(-1), test_size=0.2) # Splited data into train and test

    return X_train, X_test, y_train, y_test

def run_model(data, datasetName):
    sen_list = []
    spec_list = []
    auc_list = []
    acc_list = []
    cv_score = []
    precision_list = []
    g_mean_list = []
    f1_avg = []

    for trial in range(50):
        print(f'Running Trial #{trial+1}')

        X_train, X_test, y_train, y_test = None, None, None, None
        
        if datasetName in ['diabetes', 'cancer', 'wine', 'abalone']:
            X, Y, _ = data
            X_train, X_test, y_train, y_test = splitDataUCI(X, Y)
        else:
            X, Y = data
            X_train, X_test, y_train, y_test = splitSynthData( X, Y)

        def train_model(model, X_train, y_train, scoring, folds, repeats):
            cv_results = cross_validate(model, X_train, y_train, scoring=scoring,
                                        cv=RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats), return_estimator = True)
            score = np.max(cv_results['test_score'])
            best_model = cv_results['estimator'][np.argmax(cv_results['test_score'])]
            return best_model, score

        param_grid = {
                'C' : list(np.arange(0.05,1)),
                'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'] 
            }
        # print(len(X_train))
        # if datasetName not in ['diabetes', 'cancer', 'wine', 'abalone']:
        #     param_grid = {
        #         'C' : list(np.arange(0.05,1)),
        #         'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'] 
        #     }
        # else:
        #     param_grid = {
        #         'C' : list(np.arange(4,5)),
        #         'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'] 
        #     }


        model = SVC(kernel='rbf' , class_weight = 'balanced' , probability = True)
        clf = GridSearchCV(model, param_grid, scoring='accuracy',
                            cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=1), refit=True)
        clf.fit(X_train, y_train)
        grid_model = clf.best_estimator_

        best_model, score = train_model(grid_model, X_train, y_train, 'accuracy', 5,1)
        best_model = best_model.fit(X_train, y_train)
        pred_labels_te = best_model.predict(X_test)

        y_prob = best_model.predict_proba(X_test)
        # print(y_prob[:, 1])
        # print(pred_labels_te)

        # result.append([y_test, y_prob])

        # f = open(output_dir + 'trail_{}.pkl'.format(trail_id), 'wb')
        # pickle.dump(result, f)
        # f.close()

        report = (classification_report(y_test, pred_labels_te, output_dict=True))

        sensitivity = report['1']['recall']
        specificity = report['0']['recall']
        precision = report['1']['precision']

        auc = roc_auc_score(y_test, pred_labels_te)
        acc = best_model.score(X_test, y_test)
        g_mean = np.sqrt(specificity * sensitivity)
        # f1 = f1_score(y_test, pred_labels_te)

        print(f'Recall: {sensitivity}\nSpecificity: {specificity}\nG_mean: {g_mean}\nACC: {acc}\nAUC: {auc}\n')
        
        sen_list.append(sensitivity)
        spec_list.append(specificity)
        auc_list.append(auc)
        acc_list.append(acc)
        cv_score.append(score)
        precision_list.append(precision)
        g_mean_list.append(g_mean)
        # f1_avg.append(f1)

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
    print(f'Mean of Sensitivity: {np.mean(sen_list)}')
    print(f'Best of Sensitivity: {np.max(sen_list)}')
    print(f'Std of Sensitivity: {np.std(sen_list)}')


