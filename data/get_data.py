import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import data.synth_data as synth_data


def get_wine(priv):
    data = pd.read_csv('data/wine.csv')

    mean_outcome = data['Proline'].mean()

    conds = [
        data['Proline'] < mean_outcome, 
        data['Proline'] >= mean_outcome,
    ]
    choices = [ 0,1]
    data['Proline'] = np.select(conds, choices)

    x = data[['Wine', 'Alcohol', 'Malic.acid', 'Ash', 'Acl', 'Mg', 'Phenols',
        'Flavanoids', 'Nonflavanoid.phenols', 'Proanth', 'Color.int', 'Hue',
        'OD']]
    y = data[['Proline']]
    y_name = 'Proline'
    X_indicies = [0,1,2,3]
    X_priv_indicies = [4,5,6,7,8]

    if priv:
        return x, y, X_indicies, X_priv_indicies, [x.columns[i] for i in X_priv_indicies]
    else:
        return x, y, y_name

def get_abalone(priv):
    data = pd.read_csv('data/abalone.csv', header = None)
    data.columns = ['sex', 'length', 'diameter', 'height', 'whole height', 'shucked weight', 'visera weight', 'shell weight', 'outcome']

    labelencoder = LabelEncoder()
    data['sex'] = labelencoder.fit_transform(data['sex'])

    mean_outcome = data['outcome'].mean()

    conds = [
        data['outcome'] < mean_outcome, 
        data['outcome'] >= mean_outcome,
    ]
    choices = [ 0,1]
    data['outcome'] = np.select(conds, choices)

    data = data.sample(frac=0.2, replace=False)

    x = data[['sex', 'length', 'diameter', 'height', 'whole height', 'shucked weight', 'visera weight', 'shell weight']]
    y = data[['outcome']]
    y_name = 'outcome'
    X_indicies = [0,1,2,3,7,6,5]
    X_priv_indicies = [4]

    if priv:
        return x, y, X_indicies, X_priv_indicies, [x.columns[i] for i in X_priv_indicies]
    else:
        return x, y, y_name

def get_cancer(priv):
    data = pd.read_csv('data/breast-cancer-wisconsin.csv', header = None)
    data.columns = ['id', 'clump', 'cell size', 'cell shape', 'adhesion', 'epithelial', 'nuclei', 'chromatin', 'normal nucleoli', 'mitoses', 'outcome']
    data.drop(labels = ['id'], axis = 1, inplace = True)
    data.head()

    conds = [
        data['outcome'] == 4, 
        data['outcome'] == 2,
    ]
    choices = [ 1,0]
    data['outcome'] = np.select(conds, choices)

    data = data.apply(pd.to_numeric, errors='coerce')

    x = data[['clump', 'cell size', 'cell shape', 'adhesion', 'epithelial', 'nuclei', 'chromatin', 'normal nucleoli', 'mitoses']]
    y = data[['outcome']]
    y_name = 'outcome'

    X_indicies = [0,1,2,3,5,6,8]
    X_priv_indicies = [7,4]

    if priv:
        return x, y, X_indicies, X_priv_indicies , [x.columns[i] for i in X_priv_indicies]
    else:
        return x, y, y_name


def get_nonucidata(priv):
    data = pd.read_csv('data/diabetes.csv')  

    conds = [
        data['Outcome'] == 1, 
        data['Outcome'] == 0,
    ]
    choices = [ 1,0]
    data['Outcome'] = np.select(conds, choices)

    x = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
        'BMI', 'DiabetesPedigreeFunction', 'Age']]
    y = data[['Outcome']]
    y_name = 'Outcome'

    X_indicies = [1,2,0,3,4,6]
    X_priv_indicies = [5,7]

    if priv:
        return x, y, X_indicies, X_priv_indicies , [x.columns[i] for i in X_priv_indicies]
    else:
        return x, y, y_name

def gatherData(datasetName, modelName = None,priv = False):
    if datasetName in ['synth1', 'synth2', 'synth3', 'synth4']:
        a  = np.random.randn(25)
        n = 300
        X_priv, X, Y = None, None, None
        if datasetName == 'synth1':
            X_priv, X, Y = synth_data.synthetic_01(a,n)
        elif datasetName == 'synth2':
            X_priv, X, Y = synth_data.synthetic_02(a,n)
        elif datasetName == 'synth3':
            X_priv, X, Y = synth_data.synthetic_03(a,n)
        elif datasetName == 'synth4':
            X_priv, X, Y = synth_data.synthetic_04(a,n)

        Y = pd.DataFrame(Y.astype(int))
        X = pd.DataFrame(X)
    
        Y.columns = ['25']
        X.columns = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']
        
        if priv:
            return X_priv, X, Y
        else:
            return X, Y 
    else:
        if datasetName == 'cancer':
            if modelName == 'SVM+':
                return get_nonucidata(priv)
            else:
                return get_cancer(priv)
        elif datasetName == 'diabetes':
            return get_nonucidata(priv)
        elif datasetName == 'wine':
            return get_wine(priv)
        elif datasetName == 'abalone':
            return get_abalone(priv)
        
