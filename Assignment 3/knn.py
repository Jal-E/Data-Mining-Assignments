# -------------------------------------------------------------------------
# AUTHOR: Anjali Rai
# FILENAME: knn.py
# SPECIFICATION: Complete the KNN program to read the weather_training.csv and weather_test.csv data,
#                discretize the output into 11 classes, and perform a grid search for the best KNN hyperparameters.
# FOR: CS 5990- Assignment #3
# TIME SPENT: 7 hours
# -------------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Read the training data
df_training = pd.read_csv('weather_training.csv')
X_training = df_training.drop(['Temperature (C)', 'Formatted Date'], axis=1).values     #Exclude the 'Formatted Date' column
y_training = df_training['Temperature (C)'].values.astype('float')

# Read the test data
df_test = pd.read_csv('weather_test.csv')
X_test = df_test.drop(['Temperature (C)', 'Formatted Date'], axis=1).values     #Exclude the 'Formatted Date' column
y_test = df_test['Temperature (C)'].values.astype('float')

# Define the discretization of the output into 11 classes using the given temperature range and step
classes = np.arange(-22, 40, 6)

# Apply the discretization to the training and test target values using np.digitize
y_training_discrete = np.digitize(y_training, bins=classes)
y_test_discrete = np.digitize(y_test, bins=classes)

# Define the hyperparameter values of KNN
k_values = [i for i in range(1, 20)]
p_values = [1, 2]
w_values = ['uniform', 'distance']

# Initialize the highest accuracy
highest_accuracy = 0
best_k = best_p = best_w = None

# Loop over the hyperparameter values (k, p, and weight) of KNN
for k in k_values:
    for p in p_values:
        for w in w_values:
            # Create and fit the KNN model
            clf = KNeighborsClassifier(n_neighbors=k, p=p, weights=w)
            clf.fit(X_training, y_training_discrete)

            # Make the KNN prediction for each test sample and start computing its accuracy
            correct_predictions = 0
            for x_testSample, y_true in zip(X_test, y_test_discrete):
                y_pred = clf.predict([x_testSample])[0]
                # Use a tolerance of 15% around the true class label for correctness
                if abs(y_pred - y_true) <= round(0.15 * y_true):
                    correct_predictions += 1

            # Calculate the accuracy
            accuracy = correct_predictions / len(X_test)

            # Check if the calculated accuracy is higher than the previously calculated one
            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
                best_k, best_p, best_w = k, p, w
                # Print the highest accuracy so far with the hyperparameters
                print(f'\nHighest KNN accuracy so far: {highest_accuracy:.2f}')
                print(f'Parameters: k={best_k}, p={best_p}, weight={best_w}')