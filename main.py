import sys

from data.get_data import gatherData
import models.svm_plus
import models.xgb_plus
import models.svm
import models.xgb

# python main.py model dataset

def runModel(modelName, datasetName):
    priv = False
    if modelName in ['SVM+', 'XGB+']:
        priv = True
    data = gatherData(datasetName=datasetName, modelName = modelName,priv=priv)
    
    if modelName == 'SVM+':
        models.svm_plus.run_model(data, datasetName)
    elif modelName == 'XGB+':
        models.xgb_plus.run_model(data,datasetName)
    elif modelName == 'SVM':
        models.svm.run_model(data,datasetName)  
    elif modelName == 'XGB':
        models.xgb.run_model(data, datasetName)

if __name__ == "__main__":
    n = len(sys.argv)
    if n != 3:
        print(f'ERROR: Please run the command using the following format: python3 main.py modelName datasetName')
        sys.exit(1)

    inputtedModelName = sys.argv[1]
    inputtedDatasetName = sys.argv[2]

    if inputtedModelName not in ['SVM', 'SVM+', 'XGB', 'XGB+']:
        print(f'ERROR: Incorrect option for modelName. Avaiable options are: SVM, SVM+, XGB, XGB+')
        sys.exit(3)

    if inputtedDatasetName not in ['synth1', 'synth2', 'synth3', 'synth4', 'diabetes', 'cancer', 'wine', 'abalone']:
        print(f'ERROR: Incorrect option for datasetName. Avaiable options are: synth1, synth2, synth3, syth4, diabetes, cancer, wine, abalone')
        sys.exit(2)

    runModel(inputtedModelName, inputtedDatasetName)

