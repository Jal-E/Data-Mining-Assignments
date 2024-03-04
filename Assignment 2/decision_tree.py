# -------------------------------------------------------------------------
# AUTHOR: Anjali Rai
# FILENAME: decision_tree
# SPECIFICATION: description of the program
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv']

marital_status = {"Single": [1, 0, 0], "Divorced": [0, 1, 0], "Married": [0, 0, 1]}
refund = {"Yes": 1, "No": 0}


def process_instance_row(instance):
    new_instance = []
    new_instance.append(refund[instance[0]])
    hot_encode = marital_status[instance[1]]
    new_instance.extend(hot_encode)
    new_instance.append(float(instance[2].replace('k', '')))
    return new_instance


for ds in dataSets:
    X = []
    Y = []

    df = pd.read_csv(ds, sep=',', header=0)
    data_training = np.array(df.values)[:, 1:]

    for instance in data_training:
        X.append(process_instance_row(instance))
        Y.append(refund[instance[3]])

    dtest = pd.read_csv('cheat_test.csv', sep=',', header=0)
    data_test = np.array(dtest.values)[:, 1:]

    accuracies = []
    for i in range(10):
        clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=None)
        clf = clf.fit(X, Y)

        tp, tn, fp, fn = 0, 0, 0, 0
        for data in data_test:
            transformed_test_instance = process_instance_row(data)
            class_predicted = clf.predict([transformed_test_instance])[0]
            test_class_value = refund[data[3]]

            if class_predicted == test_class_value == 1:
                tp += 1
            elif class_predicted == 1 and test_class_value == 0:
                fp += 1
            elif class_predicted == 0 and test_class_value == 1:
                fn += 1
            else:
                tn += 1

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        accuracies.append(accuracy)

    avg_accuracy = np.average(accuracies)
    print("Average accuracy when training on", ds, ":", avg_accuracy)
