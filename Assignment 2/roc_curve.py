# -------------------------------------------------------------------------
# AUTHOR: Anjali Rai
# FILENAME: decision_tree.py
# SPECIFICATION: The program trains a decision tree classifier on 'cheat_data.csv', evaluates its performance using ROC AUC score, and saves the ROC curve plot as 'roc_curve_plot.png'.
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: 2 days
# -----------------------------------------------------------*/
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot
import numpy as np
import pandas as pd

marital_status = {"Single": [1, 0, 0], "Divorced": [0, 1, 0], "Married": [0, 0, 1]}
refund = {"Yes": 1, "No": 0}

def process_instance_row(instance):
    new_instance = []
    new_instance.append(refund[instance[0]])
    hot_encode = marital_status[instance[1]]
    new_instance.extend(hot_encode)
    new_instance.append(float(instance[2].replace('k', '')))
    new_instance.append(refund[instance[3]])
    return new_instance

# Read the dataset cheat_data.csv and prepare the data_training numpy array
cheat_data = pd.read_csv('cheat_data.csv', sep=',', header=0)
data_training = np.array(cheat_data.values)

# Check if both classes are present in the dataset
classes = np.unique(data_training[:, -1])
if len(classes) < 2:
    print("Error: Only one class present in the dataset.")
    exit()

# Transform the original training features to numbers and add them to X and Y
X = []
Y = []
for instance in data_training:
    new_instance = process_instance_row(instance)
    X.append(new_instance[:5])
    Y.append(new_instance[5])

# Split into train/test sets using 30% for test
trainX, testX, trainy, testy = train_test_split(X, Y, test_size=0.3)

# Check if both classes are present in the true labels
if len(np.unique(trainy)) < 2 or len(np.unique(testy)) < 2:
    print("Error: Only one class present in the true labels.")
    exit()

# Fit a decision tree model using entropy with max depth = 2
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2)
clf.fit(trainX, trainy)

# Predict probabilities for all test samples
dt_probs = clf.predict_proba(testX)[:, 1]

# Calculate scores
ns_auc = roc_auc_score(testy, [0 for _ in range(len(testy))])
dt_auc = roc_auc_score(testy, dt_probs)
print('No Skill: ROC AUC=%.3f' % ns_auc)
print('Decision Tree: ROC AUC=%.3f' % dt_auc)

# Calculate ROC curves
ns_fpr, ns_tpr, _ = roc_curve(testy, [0 for _ in range(len(testy))])
dt_fpr, dt_tpr, _ = roc_curve(testy, dt_probs)

# Plot the ROC curve for the model and save it as a PNG file
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(dt_fpr, dt_tpr, marker='.', label='Decision Tree')

# Axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')

# Show the legend
pyplot.legend()

# Save the plot as a PNG file
pyplot.savefig('roc_curve_plot.png')

# Close the plot to avoid displaying it in the console
pyplot.close()
