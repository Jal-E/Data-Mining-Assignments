# -------------------------------------------------------------------------
# AUTHOR: Anjali Rai
# FILENAME: naive_bayes.py
# SPECIFICATION: Complete the Naive Bayes program to read training and test data, discretize the output,
#                and calculate accuracy after all predictions.
# FOR: CS 5990- Assignment #3
# TIME SPENT: 5 hours
# -------------------------------------------------------------------------

# importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd

# 11 classes after discretization
classes = [i for i in range(-22, 40, 6)]

# Read the training data
df_training = pd.read_csv('weather_training.csv')
X_training = df_training.drop(['Temperature (C)', 'Formatted Date'], axis=1).values  # Exclude the 'Formatted Date' column
y_training = df_training['Temperature (C)'].values.astype('float')

# Discretize the training class values
y_training = np.digitize(y_training, bins=classes) - 1

# Read the test data
df_test = pd.read_csv('weather_test.csv')
X_test = df_test.drop(['Temperature (C)', 'Formatted Date'], axis=1).values   #Exclude the 'Formatted Date' column
y_test = df_test['Temperature (C)'].values.astype('float')

# Discretize the test class values 
y_test = np.digitize(y_test, bins=classes) - 1

# Fitting the naive_bayes to the data
clf = GaussianNB()
clf = clf.fit(X_training, y_training)

# Make the naive_bayes prediction for each test sample and start computing its accuracy
correct_predictions = 0
total_predictions = len(X_test)
for x_testSample, real_class_index in zip(X_test, y_test):
    predicted_class_index = clf.predict([x_testSample])[0]
    # Convert the class index to the midpoint of class range to approximate real value
    predicted_value = (classes[predicted_class_index] + classes[min(predicted_class_index + 1, len(classes) - 1)]) / 2
    real_value = (classes[real_class_index] + classes[min(real_class_index + 1, len(classes) - 1)]) / 2
    percentage_difference = 100 * abs(predicted_value - real_value) / abs(real_value)
    if percentage_difference <= 15:
        correct_predictions += 1

# Calculate the Naive Bayes accuracy
accuracy = correct_predictions / total_predictions

# Print the Naive Bayes accuracy
print("Naive Bayes accuracy: " + str(accuracy))
