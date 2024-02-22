import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
def LoadData(path,x_cols,y_cols):
    # Load data from the path
    data = pd.read_csv(path)
    # Assign X data
    X = data[x_cols]
    # Assign Y data
    Y = data[y_cols]
    return X,Y

def TrainModel(X,Y,classifier):
    # train the model
    classifier.fit(X,Y)
    return classifier

def Predict(X,Y,classifier):
    #  predict from the model
    Y_predict = classifier.predict(X)
    # Calculate accuracy
    accuracy = accuracy_score(Y,Y_predict)
    return accuracy

def TrainPredict(X_train,Y_train,X,Y):
    
    # Train all the classifiers
    Dummy_maj = TrainModel(X_train,Y_train,DummyClassifier(strategy='most_frequent',random_state=0))
    Dummy_strat = TrainModel(X_train,Y_train,DummyClassifier(strategy='stratified',random_state=0))
    RF = TrainModel(X_train,Y_train, RandomForestClassifier(random_state=0,min_samples_split=10,n_estimators=100))
    ThreeNN = TrainModel(X_train,Y_train,KNeighborsClassifier(n_neighbors=3))
    NineNN = TrainModel(X_train,Y_train,KNeighborsClassifier(n_neighbors=9))
    GP = TrainModel(X_train,Y_train,GaussianProcessClassifier(random_state=0))


if __name__ == "__main__":
    # Features for X and Y
    X_cols = ['Distance(mi)', 'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)',
                     'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)',
                     'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit',
                     'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming',
                     'Traffic_Signal', 'Turning_Loop']
    Y_cols = ['Severity']
    # Load training data
    X_train,Y_train = LoadData('/Users/wadoodalam/ML Models Adapting to Shifts/train-TX.csv',X_cols,Y_cols)
 