import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
import warnings
from label_shift_adaptation import analyze_val_data, update_probs
warnings.filterwarnings('ignore')


def LoadData(path):
    # Load data from the path
    data = pd.read_csv(path,usecols=['Distance(mi)', 'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)',
                     'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)',
                     'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit',
                     'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming',
                     'Traffic_Signal', 'Turning_Loop', 'Severity'])
    # Assign X data
    X = data[['Distance(mi)', 'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)',
                     'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)',
                     'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit',
                     'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming',
                     'Traffic_Signal', 'Turning_Loop']]
    # Assign Y data
    Y = data['Severity']
    return X,Y

def TrainModel(X,Y,classifier):
    # train the model
    classifier.fit(X,Y)
    return classifier

def Predict(X,Y,classifier):
    #  predict from the model
    Y_predict = classifier.predict(X)
    # Calculate accuracy in % with 2 decimal places
    accuracy = round(accuracy_score(Y, Y_predict) * 100, 2)
    return accuracy, Y_predict

def Train(X_train,Y_train):
    
    # Train all the classifiers
    Dummy_maj = TrainModel(X_train,Y_train,DummyClassifier(strategy='most_frequent',random_state=0))
    Dummy_strat = TrainModel(X_train,Y_train,DummyClassifier(strategy='stratified',random_state=0))
    RF = TrainModel(X_train,Y_train, RandomForestClassifier(random_state=0,min_samples_split=10,n_estimators=100))
    ThreeNN = TrainModel(X_train,Y_train,KNeighborsClassifier(n_neighbors=3))
    NineNN = TrainModel(X_train,Y_train,KNeighborsClassifier(n_neighbors=9))
    GP = TrainModel(X_train,Y_train,GaussianProcessClassifier(random_state=0))
    # assign the trained models to a list to return
    classifiers = [Dummy_maj, Dummy_strat,RF,GP,ThreeNN,NineNN]
    return classifiers

def ConvertAccuracyTable(val_accuracies,t1_accuracies,t2_accuracies,t3_accuracies):
    model_names = ['Dummy Classifier with Most-Frequent', 'Dummy Classifier with Stratified', 'Random Forest', 
                   'GaussianProcess','3-NearestNeighbor', '9-NearestNeighbor']
    df = pd.DataFrame({'Classifier': model_names, 'Val-TX':val_accuracies, 'Test1-TX':t1_accuracies,
                       'Test2-FL':t2_accuracies, 'Test3-FL':t3_accuracies})
    return df

def PredictAccuracies(classifiers,X_val,Y_val,X_t1,Y_t1,X_t2,Y_t2,X_t3,Y_t3):
    # Init list for accuracies
    val_accuracies = []
    t1_accuracies = []
    t2_accuracies = []
    t3_accuracies = []
    i = 0
    weights_adapted = {}
    # Predict and get the accuracy on val set for all models
    for model in classifiers:
        # Predict, using .values to predict using the values, otherwise gives error
        accuracy,Y_val_predict_var = Predict(X_val.values,Y_val.values,model)
        val_accuracies.append(accuracy)
        
    # Predict and get the accuracy on test set 1 for all models

        # Predict, using .values to predict using the values, otherwise gives error
        accuracy,Y_t1_predict_var = Predict(X_t1.values,Y_t1.values,model)
        
        t1_accuracies.append(accuracy)  
    
    # Predict and get the accuracy on test set 2 for all models

        # Predict, using .values to predict using the values, otherwise gives error
        accuracy,Y_t2_predict_var = Predict(X_t2.values,Y_t2.values,model)
        
        t2_accuracies.append(accuracy)   
        
    # Predict and get the accuracy on test set 3 for all models
        # Predict, using .values to predict using the values, otherwise gives error
        accuracy,Y_t3_predict_var = Predict(X_t3.values,Y_t3.values,model)
        
        t3_accuracies.append(accuracy)  
        if i > 1 and i < len(classifiers):
            weights_adapted[model] = [] 
            weights_adapted[model].extend([float(val) for val in analyze_val_data(Y_val,Y_val_predict_var,Y_t1_predict_var)])
            weights_adapted[model].extend([float(val) for val in analyze_val_data(Y_val,Y_val_predict_var,Y_t2_predict_var)])
            weights_adapted[model].extend([float(val) for val in analyze_val_data(Y_val,Y_val_predict_var,Y_t3_predict_var)])
                
        i+=1 
    return val_accuracies,t1_accuracies,t2_accuracies,t3_accuracies,weights_adapted

def ConvertWeightsToTable(weights):
    data = []
    processed_models = set()
    column_name = ['Test1-TX', 'Test2-FL', 'Test3-FL']
    
    for model, values in weights.items():
        # Check if the model is not in set
        if model not in processed_models:
            # Create a dictionary for each model
            row = {'Model': model}
            
            # Add the first 4 values of the model's list
            for i in range(3):
                col_name = column_name[i]
                # slice the 4 corresponding values
                col_values = values[i*4:(i+1)*4]
                # Round each value to 2 decimal places
                col_values_rounded = [round(value, 2) for value in col_values]
                row[col_name] = col_values_rounded
            data.append(row)
        
            # Add the model to the set
            processed_models.add(model)
    df = pd.DataFrame(data)
    return df




if __name__ == "__main__":
    
    # Load training data
    X_train,Y_train = LoadData('/Users/wadoodalam/ML Models Adapting to Shifts/train-TX.csv')
    # Load val data
    X_val,Y_val = LoadData('/Users/wadoodalam/ML Models Adapting to Shifts/val-TX.csv')
    # Load test 1 data
    X_t1,Y_t1 = LoadData('/Users/wadoodalam/ML Models Adapting to Shifts/test1-TX.csv')
    # Load test 2 data
    X_t2,Y_t2 = LoadData('/Users/wadoodalam/ML Models Adapting to Shifts/test2-FL.csv')
    # Load test 3 data
    X_t3,Y_t3 = LoadData('/Users/wadoodalam/ML Models Adapting to Shifts/test3-FL.csv')
    
        
    # Train all the classifiers
    classifiers = Train(X_train,Y_train)
    
    # Get all accuracies without BBSC
    val_accuracies,t1_accuracies,t2_accuracies,t3_accuracies,weights = PredictAccuracies(classifiers,X_val,Y_val,X_t1,Y_t1,X_t2,Y_t2,X_t3,Y_t3)        
    
    
    # Convert accuracies to dataframe and output in a csv table to accuracy.csv
    df_accuracy = ConvertAccuracyTable(val_accuracies,t1_accuracies,t2_accuracies,t3_accuracies)
    df_accuracy.to_csv('accuracy.csv')
    
    # Convert weights to dataframe and output in a csv table to weights.csv
    df_weight = ConvertWeightsToTable(weights)
    df_weight.to_csv('weights.csv')
    
