import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
from label_shift_adaptation import analyze_val_data, update_probs
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


def LoadData(path,Train=False):
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
    if not Train:
        # get sorted true labels by class
        true_labels = data['Severity'].value_counts().sort_index()
 
        return X,Y,true_labels,len(Y)
    else:
        return X,Y

def TrainModel(X,Y,classifier):
    # train the model
    classifier.fit(X,Y)
    return classifier

def Predict(X,Y,classifier,prob=False):
    #  predict from the model
    Y_predict = classifier.predict(X.values)
    # Get predicted probabilities 
    Y_prob = classifier.predict_proba(X.values)
    # Calculate accuracy in % with 2 decimal places
    accuracy = round(accuracy_score(Y, Y_predict) * 100, 2)
    # Return predicted prob if asked
    if prob:
        return Y_predict, Y_prob
    else:
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

def ConvertAccuracyTableBBSC(t1_accuracies,t2_accuracies,t3_accuracies):
    model_names = ['Random Forest','GaussianProcess','3-NearestNeighbor', '9-NearestNeighbor']    
    df = pd.DataFrame({'Classifier': model_names,'Test1-TX':t1_accuracies,'Test2-FL':t2_accuracies, 'Test3-FL':t3_accuracies})
    
    return df

def PredictAccuracies(classifiers,X_val,Y_val,X_t1,Y_t1,X_t2,Y_t2,X_t3,Y_t3):
    # Init list for accuracies
    val_accuracies = []
    t1_accuracies = []
    t2_accuracies = []
    t3_accuracies = []
    i = 0
    weights_adapted = {}
    GB = classifiers[3]
    ThreeNN = classifiers[4]
    # Predict and get the accuracy on val set for all models
    for model in classifiers:
        # Predict, using .values to predict using the values, otherwise gives error
        accuracy,Y_val_predict_var = Predict(X_val,Y_val,model)
        val_accuracies.append(accuracy)
        
    # Predict and get the accuracy on test set 1 for all models

        # Predict, using .values to predict using the values, otherwise gives error
        accuracy,Y_t1_predict_var = Predict(X_t1,Y_t1,model)
        
        t1_accuracies.append(accuracy)  
    
    # Predict and get the accuracy on test set 2 for all models

        # Predict, using .values to predict using the values, otherwise gives error
        accuracy,Y_t2_predict_var = Predict(X_t2,Y_t2,model)
        
        t2_accuracies.append(accuracy)   
        
    # Predict and get the accuracy on test set 3 for all models
        # Predict, using .values to predict using the values, otherwise gives error
        accuracy,Y_t3_predict_var = Predict(X_t3,Y_t3,model)

        t3_accuracies.append(accuracy)
            
        # Analyze and find weights
        if i > 1 and i < len(classifiers):
            weights_adapted[str(model)] = [] 
            weights_adapted[str(model)].extend([float(val) for val in analyze_val_data(Y_val,Y_val_predict_var,Y_t1_predict_var)])
            weights_adapted[str(model)].extend([float(val) for val in analyze_val_data(Y_val,Y_val_predict_var,Y_t2_predict_var)])
            weights_adapted[str(model)].extend([float(val) for val in analyze_val_data(Y_val,Y_val_predict_var,Y_t3_predict_var)])
                
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
                col_values_rounded = ["{:.2f}".format(value) for value in col_values]
                row[col_name] = col_values_rounded
            data.append(row)
        
            # Add the model to the set
            processed_models.add(model)
    df = pd.DataFrame(data)
    return df

def SeparateWights(weights):
    # Init Dict for weight separation
    RF_weights = {'T1': [], 'T2': [], 'T3': []}
    GP_weights = {'T1': [], 'T2': [], 'T3': []}
    ThreeNN_weights = {'T1': [], 'T2': [], 'T3': []}
    NineNN_weights = {'T1': [], 'T2': [], 'T3': []}
    
    # Separate weights for for each classifier per test set
    RF_weights['T1'] = (weights['RandomForestClassifier(min_samples_split=10, random_state=0)'][:4])
    RF_weights['T2'] = (weights['RandomForestClassifier(min_samples_split=10, random_state=0)'][4:8])
    RF_weights['T3'] = (weights['RandomForestClassifier(min_samples_split=10, random_state=0)'][8:12])
    
    GP_weights['T1'] = (weights['GaussianProcessClassifier(random_state=0)'][:4])
    GP_weights['T2'] = (weights['GaussianProcessClassifier(random_state=0)'][4:8])
    GP_weights['T3'] = (weights['GaussianProcessClassifier(random_state=0)'][8:12])
    
    ThreeNN_weights['T1'] = (weights['KNeighborsClassifier(n_neighbors=3)'][:4])
    ThreeNN_weights['T2'] = (weights['KNeighborsClassifier(n_neighbors=3)'][4:8])
    ThreeNN_weights['T3'] = (weights['KNeighborsClassifier(n_neighbors=3)'][8:12])
    
    NineNN_weights['T1'] = (weights['KNeighborsClassifier(n_neighbors=9)'][:4])
    NineNN_weights['T2'] = (weights['KNeighborsClassifier(n_neighbors=9)'][4:8])
    NineNN_weights['T3'] = (weights['KNeighborsClassifier(n_neighbors=9)'][8:12])

    # Return weights dicts
    return RF_weights,GP_weights,ThreeNN_weights,NineNN_weights

def UniqueClasses(Y):
    classes = np.unique(Y).tolist()
    return classes

def PredictAccuraciesBBSC(classifiers,X_val,Y_val,X_t1,Y_t1,X_t2,Y_t2,X_t3,Y_t3,weights):
    # slice to ignore dummy classifiers
    classifiers = classifiers[2:len(classifiers)]
    
    
    # Get Prob and Pred for each model for each test set
    Y_t1_Predict_RF, Y_t1_Prob_RF = Predict(X_t1,Y_t1,classifiers[0],True)
    Y_t2_Predict_RF, Y_t2_Prob_RF = Predict(X_t2,Y_t2,classifiers[0],True)
    Y_t3_Predict_RF, Y_t3_Prob_RF = Predict(X_t3,Y_t3,classifiers[0],True)
    
    Y_t1_Predict_GP, Y_t1_Prob_GP = Predict(X_t1,Y_t1,classifiers[1],True)
    Y_t2_Predict_GP, Y_t2_Prob_GP = Predict(X_t2,Y_t2,classifiers[1],True)
    Y_t3_Predict_GP, Y_t3_Prob_GP = Predict(X_t3,Y_t3,classifiers[1],True)
    
    Y_t1_Predict_3NN, Y_t1_Prob_3NN = Predict(X_t1,Y_t1,classifiers[2],True)
    Y_t2_Predict_3NN, Y_t2_Prob_3NN = Predict(X_t2,Y_t2,classifiers[2],True)
    Y_t3_Predict_3NN, Y_t3_Prob_3NN = Predict(X_t3,Y_t3,classifiers[2],True)
    
    Y_t1_Predict_9NN, Y_t1_Prob_9NN = Predict(X_t1,Y_t1,classifiers[3],True)
    Y_t2_Predict_9NN, Y_t2_Prob_9NN = Predict(X_t2,Y_t2,classifiers[3],True)
    Y_t3_Predict_9NN, Y_t3_Prob_9NN = Predict(X_t3,Y_t3,classifiers[3],True)
     
    # Get unique classes (same for all test sets), hence only getting from 1 test set
    unique_classes = UniqueClasses(Y_t1)
    
    # Get all the weights, dicts, divided by models and test sets
    RF_weights,GP_weights,ThreeNN_weights,NineNN_weights = SeparateWights(weights)
    
    # Update the prob using BBSC
    new_t1_pred_RF, new_t1_prob_RF = update_probs(unique_classes,RF_weights['T1'],Y_t1_Predict_RF,Y_t1_Prob_RF)
    new_t2_pred_RF, new_t2_prob_RF = update_probs(unique_classes,RF_weights['T2'],Y_t2_Predict_RF,Y_t2_Prob_RF)
    new_t3_pred_RF, new_t3_prob_RF = update_probs(unique_classes,RF_weights['T3'],Y_t3_Predict_RF,Y_t3_Prob_RF)
    
    new_t1_pred_GP, new_t1_prob_GP = update_probs(unique_classes,GP_weights['T1'],Y_t1_Predict_GP,Y_t1_Prob_GP)
    new_t2_pred_GP, new_t2_prob_GP = update_probs(unique_classes,GP_weights['T2'],Y_t2_Predict_GP,Y_t2_Prob_GP)
    new_t3_pred_GP, new_t3_prob_GP = update_probs(unique_classes,GP_weights['T3'],Y_t3_Predict_GP,Y_t3_Prob_GP)

    new_t1_pred_3NN, new_t1_prob_3NN = update_probs(unique_classes,ThreeNN_weights['T1'],Y_t1_Predict_3NN,Y_t1_Prob_3NN)
    new_t2_pred_3NN, new_t2_prob_3NN = update_probs(unique_classes,ThreeNN_weights['T2'],Y_t2_Predict_3NN,Y_t2_Prob_3NN)
    new_t3_pred_3NN, new_t3_prob_3NN = update_probs(unique_classes,ThreeNN_weights['T3'],Y_t3_Predict_3NN,Y_t3_Prob_3NN)

    new_t1_pred_9NN, new_t1_prob_9NN = update_probs(unique_classes,NineNN_weights['T1'],Y_t1_Predict_9NN,Y_t1_Prob_9NN)
    new_t2_pred_9NN, new_t2_prob_9NN = update_probs(unique_classes,NineNN_weights['T2'],Y_t2_Predict_9NN,Y_t2_Prob_9NN)
    new_t3_pred_9NN, new_t3_prob_9NN = update_probs(unique_classes,NineNN_weights['T3'],Y_t3_Predict_9NN,Y_t3_Prob_9NN)

    # Confusion matrices for GB model
    conf_matrix_t1 = confusion_matrix(Y_t1,new_t1_pred_GP)
    conf_matrix_t2 = confusion_matrix(Y_t2,new_t2_pred_GP)
    conf_matrix_t3 = confusion_matrix(Y_t3,new_t3_pred_GP)
    print("T1 set conf matrix for GB:\n",conf_matrix_t1)
    print("T2 set conf matrix for GB:\n",conf_matrix_t2)
    print("T3 set conf matrix for GB:\n",conf_matrix_t3)
        
    # Confusion matrices for 3-NN model
    conf_matrix_t1 = confusion_matrix(Y_t1,new_t1_pred_3NN)
    print("T1 set conf matrix for 3-NN:\n",conf_matrix_t1)
 

    # Append all the accuracies in the lists for each test set
    # The indices represent the classifiers by RF,GP,3NN,9NN respectively
    t1_acc_BBSC = []
    t2_acc_BBSC = []
    t3_acc_BBSC = []
    
    t1_acc_BBSC.append(round(accuracy_score(Y_t1, new_t1_pred_RF) * 100, 2))
    t1_acc_BBSC.append(round(accuracy_score(Y_t1, new_t1_pred_GP) * 100, 2))
    t1_acc_BBSC.append(round(accuracy_score(Y_t1, new_t1_pred_3NN) * 100, 2))
    t1_acc_BBSC.append(round(accuracy_score(Y_t1, new_t1_pred_9NN) * 100, 2))
    
    t2_acc_BBSC.append(round(accuracy_score(Y_t2, new_t2_pred_RF) * 100, 2))
    t2_acc_BBSC.append(round(accuracy_score(Y_t2, new_t2_pred_GP) * 100, 2))
    t2_acc_BBSC.append(round(accuracy_score(Y_t2, new_t2_pred_3NN) * 100, 2))
    t2_acc_BBSC.append(round(accuracy_score(Y_t2, new_t2_pred_9NN) * 100, 2))
    
    t3_acc_BBSC.append(round(accuracy_score(Y_t3, new_t3_pred_RF) * 100, 2))
    t3_acc_BBSC.append(round(accuracy_score(Y_t3, new_t3_pred_GP) * 100, 2))
    t3_acc_BBSC.append(round(accuracy_score(Y_t3, new_t3_pred_3NN) * 100, 2))
    t3_acc_BBSC.append(round(accuracy_score(Y_t3, new_t3_pred_9NN) * 100, 2))
    
    return t1_acc_BBSC, t2_acc_BBSC, t3_acc_BBSC

def Normalize(label,count):
    normalized = label/count
    return normalized
 
def TrueLabelsBarPlot(val, t1, t2, t3, val_len, t1_len, t2_len, t3_len):
    # Normalize the data
    val_normal = Normalize(val, val_len)
    t1_normal = Normalize(t1, t1_len)
    t2_normal = Normalize(t2, t2_len)
    t3_normal = Normalize(t3, t3_len)
    

    # Concatenate the normalized data into a single DataFrame
    normalized_df = pd.concat([val_normal, t1_normal, t2_normal, t3_normal], axis=1)
    normalized_df.columns = ['Validation', 'Test 1', 'Test 2', 'Test 3'] 
   
    # Plot the bar plot using pandas
    ax = normalized_df.plot.bar(figsize=(10, 6))

    # Set labels and title
    ax.set_xlabel('Classes')
    ax.set_ylabel('Counts')
    ax.set_title('True Class Label Distribution (Normalized)')

    # Save the plot to BarPlot.png file
    ax.legend(title='Dataset', bbox_to_anchor=(1, 1))
    plt.savefig('BarPlot.png')


if __name__ == "__main__":
    
    # Load training data
    X_train,Y_train = LoadData('/Users/wadoodalam/ML Models Adapting to Shifts/train-TX.csv',True)
    # Load val data
    X_val,Y_val,true_label_val, val_len = LoadData('/Users/wadoodalam/ML Models Adapting to Shifts/val-TX.csv')
    # Load test 1 data
    X_t1,Y_t1,true_label_t1, t1_len = LoadData('/Users/wadoodalam/ML Models Adapting to Shifts/test1-TX.csv')
    # Load test 2 data
    X_t2,Y_t2,true_label_t2, t2_len = LoadData('/Users/wadoodalam/ML Models Adapting to Shifts/test2-FL.csv')
    # Load test 3 data
    X_t3,Y_t3,true_label_t3, t3_len = LoadData('/Users/wadoodalam/ML Models Adapting to Shifts/test3-FL.csv')
    
   
    
    
    # Create Bar plot for True labels
    TrueLabelsBarPlot(true_label_val,true_label_t1,true_label_t2,true_label_t3, val_len,t1_len,t2_len,t3_len)
    # Train all the classifiers
    classifiers = Train(X_train,Y_train)
    
    # Get all accuracies without BBSC
    val_accuracies,t1_accuracies,t2_accuracies,t3_accuracies,weights = PredictAccuracies(classifiers,X_val,Y_val,X_t1,Y_t1,X_t2,Y_t2,X_t3,Y_t3)        
    
    # Get all accuracies with BBSC
    t1_acc_BBSC, t2_acc_BBSC, t3_acc_BBSC = PredictAccuraciesBBSC(classifiers,X_val,Y_val,X_t1,Y_t1,X_t2,Y_t2,X_t3,Y_t3,weights)
    
    # Convert accuracies and weights to dataframe 
    df_accuracy = ConvertAccuracyTable(val_accuracies,t1_accuracies,t2_accuracies,t3_accuracies)
    df_weight = ConvertWeightsToTable(weights)
    df_accuracy_BBSC = ConvertAccuracyTableBBSC(t1_acc_BBSC,t2_acc_BBSC,t3_acc_BBSC)
    
    # Uncomment to get csv outputs for both accuracies and weights table
    df_accuracy.to_csv('accuracy.csv')
    df_weight.to_csv('weights.csv')
    df_accuracy_BBSC.to_csv('accuracy_BBSC.csv')
    
    
 
