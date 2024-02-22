#!/usr/bin/env python
# Kiri Wagstaff
#
# Compare confusion matrix on a validation set to the class distribution
# of predictions on a test set to identify updated weights (class priors)
# for use in re-training the classifier or modifying the posteriors.
#
# Based on ideas in
# 1) Lipton et al. (2018): http://proceedings.mlr.press/v80/lipton18a.html
# 2) Shrikumar and Kundaje (2020): https://arxiv.org/pdf/1901.06852.pdf

import sys
import os
import numpy as np
import pandas as pd
from scipy import linalg
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency, fisher_exact


# Compare validation set performance (confusion matrix)
# with test set predictions to compute shift adaptation weights.
#
# Arguments:
# - val_labels: list/array of validation set laels
# - val_preds: list/array of validation set predictions
# - test_preds: list/array of test set predictions
# 
# Returns adaptation weights (one value per class in val_labels)
def analyze_val_data(val_labels, val_preds, test_preds):
    """
    >>> # Class 1 and 2 are equally likely in validation data
    >>> val_labels = [1, 1, 1, 1, 2, 2, 2, 2]
    >>> # Classifier is 75% reliable on both classes
    >>> val_preds  = [1, 1, 1, 2, 2, 2, 2, 1]
    >>> # Test set predictions show 50/50 split
    >>> test_preds = [1, 1, 1, 1, 2, 2, 2, 2]
    >>> # Weights should both be 1.0 (no shift)
    >>> analyze_val_data(val_labels, val_preds, test_preds)
    array([1., 1.])

    >>> # Test set predictions show 60/40 split
    >>> test_preds = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
    >>> # Weight of class 1 should increase and class 2 decrease
        equally
    >>> analyze_val_data(val_labels, val_preds, test_preds)
    array([1.4, 0.6])

    >>> # Classifier is 100% reliable when predicting class 1 
        but 25% of the time predicts 2 when it's really 1
    >>> val_preds  = [1, 1, 1, 2, 2, 2, 2, 2]
    >>> # Test set predictions show 60/40 split
    >>> test_preds = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
    >>> # Weight of class 1 should increase and class 2 decrease
        with larger magnitude (we trust class 1 predictions,
        and at least one predicted 2 is probably a 1)
    >>> analyze_val_data(val_labels, val_preds, test_preds)
    array([1.6, 0.4])

    >>> # Classifier is 100% reliable when predicting class 2
        but 25% of the time predicts 1 when it's really 2
    >>> val_preds  = [1, 1, 1, 1, 2, 2, 2, 1]
    >>> # Test set predictions show 60/40 split
    >>> test_preds = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
    >>> # Weight of class 1 should decrease and class 2 increase
        with less magnitude (we trust class 2 predictions, 
        and at least one predicted 1 is probably a 2)
    >>> analyze_val_data(val_labels, val_preds, test_preds)
    array([0.93333333, 1.06666667])

    >>> # Classifier is 100% reliable on both classes
    >>> val_preds  = [1, 1, 1, 1, 2, 2, 2, 2]
    >>> # Test set predictions show 60/40 split
    >>> test_preds = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
    >>> # Weight of class 1 should increase and class 2 decrease
        because we trust both classes equally.  Normalizing gives
        probabilities of 0.6, 0.4 as expected.
    >>> analyze_val_data(val_labels, val_preds, test_preds)
    array([1.2, 0.8])
    """

    conf_matrix = confusion_matrix(val_labels, val_preds)
    classes = np.unique(val_labels)

    # Compute and report per-class weights (Lipton et al., 2018)
    weights = calc_class_weights(conf_matrix, test_preds, classes)

    return weights


# Given original predictions and new class weights,
# adjust test posteriors and predictions.
#
# Arguments:
# - classes: list of c unique class labels
# - weights: list of c adaptation weights (one per class)
# - test_preds: length-n list/array of test set class predictions
# - test_probs: nxc array of test set probabilities
#
# Returns:
# - new_test_preds: length-n array of test set class predictions
# - new_test_probs: nxc array of test set probabilities
def update_probs(classes, weights, test_preds, test_probs):
    """
    >>> classes = [1, 2]
    >>> weights = [1.0, 1.0]
    >>> # Test set predictions show 50/50 split
    >>> test_preds = [1, 1, 1, 2, 2, 2]
    >>> # Classifier is totally confident
    >>> test_probs = np.array([[1, 0]] * 3 + [[0, 1]] * 3)
    >>> # Predictions (and probs) don't change
    >>> update_probs(classes, weights, test_preds, test_probs)
    (array([1, 1, 1, 2, 2, 2]), array([[1, 0],
           [1, 0],
           [1, 0],
           [0, 1],
           [0, 1],
           [0, 1]]))

    >>> # Large shift
    >>> weights = [1.4, 0.6]
    >>> # Predictions shift strongly to class 1
    >>> update_probs(classes, weights, test_preds, test_probs)
    (array([1, 1, 1, 1, 1, 1]), array([[1, 0],
           [1, 0],
           [1, 0],
           [1, 0],
           [1, 0],
           [1, 0]]))

    >>> # Classifier is unsure
    >>> test_probs = np.array([[0.6, 0.4]] * 3 + [[0.4, 0.6]] * 3)
    >>> # Predictions shift more weakly to class 1
    >>> update_probs(classes, weights, test_preds, test_probs)
    (array([1, 1, 1, 1, 1, 1]), array([[0.77777778, 0.22222222],
           [0.77777778, 0.22222222],
           [0.77777778, 0.22222222],
           [0.60869565, 0.39130435],
           [0.60869565, 0.39130435],
           [0.60869565, 0.39130435]]))

    >>> # Small shift
    >>> weights = [0.93333333, 1.06666667]
    >>> # Classifier is totally confident
    >>> test_probs = np.array([[1, 0]] * 3 + [[0, 1]] * 3)
    >>> # Predictions shift strongly to class 2
    >>> update_probs(classes, weights, test_preds, test_probs)
    (array([2, 2, 2, 2, 2, 2]), array([[0, 1],
           [0, 1],
           [0, 1],
           [0, 1],
           [0, 1],
           [0, 1]]))

    >>> # Classifier is unsure
    >>> test_probs = np.array([[0.6, 0.4]] * 3 + [[0.4, 0.6]] * 3)
    >>> # Predictions shift weakly to class 2; preds don't change
    >>> update_probs(classes, weights, test_preds, test_probs)
    (array([1, 1, 1, 2, 2, 2]), array([[0.56756757, 0.43243243],
           [0.56756757, 0.43243243],
           [0.56756757, 0.43243243],
           [0.36842105, 0.63157895],
           [0.36842105, 0.63157895],
           [0.36842105, 0.63157895]]))
    """

    # Adjust test_probs in case it's a binary problem
    # and only the max prob posterior was included.
    if len(classes) == 2 and test_probs.shape[1] == 1:
        test_probs = get_full_probs(test_preds, test_probs, classes)

    # Update the posteriors
    new_test_probs = adjust_probs(weights, test_probs)

    # Generate new predictions (argmax posterior)
    new_test_preds = np.array([classes[i] for i in
                               np.argmax(new_test_probs, axis=1)])

    return (new_test_preds, new_test_probs)


# ------------------------------------------------- #

# Normalize weights so they sum to 1 and form a distribution
def normalize_distribution(weights):
    """
    >>> normalize_distribution([0.1, 0.2, 0.1])
    [0.25, 0.5, 0.25]
    >>> normalize_distribution(np.array([0.1, 0.2, 0.1]))
    [0.25, 0.5, 0.25]
    >>> normalize_distribution([0.25, 0.5, 0.25])
    [0.25, 0.5, 0.25]
    >>> normalize_distribution([0.0, 0.0, 0.0])
    Warning: cannot normalize zero-sum distribution.
    [0.0, 0.0, 0.0]
    """

    weight_sum = np.sum(weights)
    if weight_sum == 0:
        print('Warning: cannot normalize zero-sum distribution.')
        new_weights = [0.0] * len(weights)
    else:
        new_weights = [w/weight_sum for w in weights]

    return new_weights


# Compute and return per-class weights (Lipton et al., 2018)
# by inverting the confusion matrix and multiplying by the test distribution.
def calc_class_weights(conf_matrix, test_preds, classes):
    """
    # Nicely balanced example; no label shift present
    >>> cm = np.array([[10,2],[2,10]])
    >>> calc_class_weights(cm, [0, 0, 1, 1], [0, 1])
    array([1., 1.])

    # Label shift present, so up-weight class 0 and down-weight class 1
    >>> cm = np.array([[10,2],[2,10]])
    >>> calc_class_weights(cm, [0, 0, 0, 1], [0, 1])
    array([1.75, 0.25])

    # Label shift present: class 1 never appears in test.
    >>> cm = np.array([[10,2],[2,10]])
    >>> calc_class_weights(cm, [0, 0, 0, 0], [0, 1])
    array([2.5, 0. ])

    # Reliable on class 0, unreliable on class 1,
    # test preds show increased predictions of class 0 (25->50%),
    # so up-weight class 0 a lot and ignore class 1 (?)
    >>> cm = np.array([[5,5],[0,10]])
    >>> calc_class_weights(cm, [0, 0, 0, 0, 1, 1, 1, 1], [0, 1])
    array([2., 0.])

    # Reliable on class 0, unreliable on class 1,
    # test preds show increased predictions of class 1 (75->90%),
    # so up-weight class 1 (p=0.75)
    >>> cm = np.array([[5,5],[0,10]])
    >>> calc_class_weights(cm, [1, 1, 1, 1, 1, 1, 1, 0], [0, 1])
    array([0.5, 1.5])

    # Reliable on class 0, unreliable on class 1,
    # test preds show increased predictions of class 1 (25->50%),
    # so up-weight class 1 even more (p=0.81)
    >>> cm = np.array([[15,2],[0,3]])
    >>> calc_class_weights(cm, [0, 0, 0, 0, 1, 1, 1, 1], [0, 1])
    array([0.66666667, 2.88888889])

    # Reliable on class 0 and class 1
    # test preds show increased predictions of class 0 (25->50%),
    # so up-weight class 0 a lot.
    >>> cm = np.array([[5,0],[0,15]])
    >>> calc_class_weights(cm, [0, 0, 0, 0, 1, 1, 1, 1], [0, 1])
    array([2.        , 0.66666667])

    # Label shift present, but one class never appears in val labels,
    # so cannot invert confusion matrix.
    >>> cm = np.array([[0,0],[10,20]])
    >>> calc_class_weights(cm, [0, 0, 1, 1], [0, 1])
    Failed to invert validation confusion matrix; some classes may be too infrequent.
    array([0., 0.])

    # Label shift present, but one class never appears in val preds,
    # so cannot invert confusion matrix.
    >>> cm = np.array([[10,0],[2,0]])
    >>> calc_class_weights(cm, [0, 0, 1, 1], [0, 1])
    Failed to invert validation confusion matrix; some classes may be too infrequent.
    array([0., 0.])

    ############ Multiclass tests ##############
    # Nicely balanced example; no label shift present
    >>> cm = np.array([[10,2,3],[2,10,2],[3,3,10]])
    >>> calc_class_weights(cm, [0, 0, 1, 1, 2, 2], [0, 1, 2])
    array([1., 1., 1.])

    # Label shift present, so up-weight class 0 and down-weight class 1, 2
    >>> cm = np.array([[10,2,3],[2,10,2],[3,3,10]])
    >>> calc_class_weights(cm, [0, 0, 0, 1, 2, 2], [0, 1, 2])
    array([1.96428571, 0.08928571, 0.89285714])

    # Label shift present: class 2 never appears in test, so give it weight 0
    >>> cm = np.array([[10,2,3],[2,10,2],[3,3,10]])
    >>> calc_class_weights(cm, [0, 0, 1, 1], [0, 1, 2])
    array([2.14285714, 2.14285714, 0.        ])
    """

    # Normalize the confusion matrix
    # and transpose it because Lipton et al. assum rows = preds, cols = true
    C_hat = conf_matrix.T*1.0 / np.sum(conf_matrix)

    # Compute the (normalized) class distribution of test set predictions
    n_classes = len(classes)
    n_preds   = len(test_preds)
    mu_hat_test = np.zeros((n_classes,))
    for (i, c) in enumerate(classes):
        mu_hat_test[i] = len([t for t in test_preds \
                              if t == c]) * 1.0 / n_preds

    # Invert the confusion matrix and multiply by mu_hat_test
    try:
        C_hat_inv = linalg.inv(C_hat)
    except:
        print('Failed to invert validation confusion matrix; some classes may be too infrequent.')
        return np.squeeze(np.zeros((n_classes, 1)))
    
    weights = C_hat_inv.dot(mu_hat_test)

    # Clip any negative weights to zero
    weights[weights < 0] = 0.0

    return np.squeeze(weights)


# Adjust test posterior probabilities given shift of class weights
# from original posteriors (BBSC without retraining)
# (SK20 = Shrikumar and Kundaje (2020))
def adjust_probs(val_weights, test_probs):

    new_test_probs = np.zeros_like(test_probs)

    (nitems, nclasses) = test_probs.shape
    # Section 2.2 in [SK20]
    for i in range(nitems):
        for c in range(nclasses):
            new_test_probs[i, c] = val_weights[c] * test_probs[i, c]
        # Normalize new test distribution
        # Override with class weights alone if they canceled out
        if np.sum(new_test_probs[i]) == 0:
            new_test_probs[i] = val_weights
        else:
            new_test_probs[i] = normalize_distribution(new_test_probs[i])

    return new_test_probs


