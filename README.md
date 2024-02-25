# README

## Author
Wadood Alam

## Date
24th February 2024

## Assignment
AI 539 Assignment 3: Adapt to Change

## Dependencies / Imports Required

  - Python 
  - NumPy
  - Pandas
  - Scikit-learn
  - accuracy_score
  - confusion_matrix
  - RandomForestClassifier
  - GaussianProcessClassifier
  - KNeighborsClassifier
  - DummyClassifier
  - Black Box Shift Correction: analyze_val_data, update_probs


## Instructions

### Program 1: Training and Evaluation 

#### Execution 
1. Install the required dependencies using pip: Pandas, NumPy, sklearn
2. Ensure Dataset is contained in the same directory
3. Donwload the `label_shift_adaptation.py` file in the same directory as your project
4. Run the program using the command `shift_adapt.py`
5. The program will genrate 3 csv files and 1
6. `accuracy.csv`: This will generate a table of accuracies without BBSC
7. `accuracy_BBSC.csv`: This will generate a table of accuracies with BBSC
8. `weights.csv`: This will generate a table of weights calculated by BBSC
9. `BarPlot.png`: Grounded bar plot for true label distribution for all the datasets

## Files in the directory 
1. tarin-TX.csv
2. val_TX.csv
3. test1-TX.csv
4. test2-FL.csv
5. test3-FL.csv
6. shift_adapt.py
7. label_shift_adaptation.py
8. BarPlot.png
9. accuacy.csv
10. accuracy_BBSC.csv
11. weights.csv
