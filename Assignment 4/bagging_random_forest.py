# -------------------------------------------------------------------------
# AUTHOR: Anjali Rai
# FILENAME: bagging_random_forest.py
# SPECIFICATION: Complete the Python program to build classifiers to recognize handwritten digits
# FOR: CS 5990- Assignment #4
# TIME SPENT: 5 hours
# -----------------------------------------------------------*/

# importing some Python libraries
from sklearn import tree
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import csv

# initialize the training and test sets
dbTraining = []
dbTest = []
X_training = []
y_training = []
X_test = []
y_test = []
classVotes = []  # this array will be used to count the votes of each classifier

# reading the training data
with open('optdigits.tra', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        dbTraining.append(row)
        X_training.append(row[:-1])
        y_training.append(row[-1])

# reading the test data
with open('optdigits.tes', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        dbTest.append(row)
        X_test.append(row[:-1])
        y_test.append(row[-1])

# initialize class votes for each testing sample
for _ in range(len(dbTest)):
    classVotes.append([0] * 10)

# convert feature values to integers as they are currently strings
X_training = [[int(feature) for feature in sample] for sample in X_training]
y_training = [int(label) for label in y_training]
X_test = [[int(feature) for feature in sample] for sample in X_test]
y_test = [int(label) for label in y_test]

print("Started my base and ensemble classifier ...")

# Variables to store accuracy of the base classifier and ensemble classifier
base_accuracy = 0
ensemble_accuracy = 0

# We will create 20 bootstrap samples here (k = 20). One classifier will be created for each bootstrap sample
for k in range(20):
    bootstrapSample = resample(dbTraining, n_samples=len(dbTraining), replace=True)
    X_bootstrap = [sample[:-1] for sample in bootstrapSample]
    y_bootstrap = [sample[-1] for sample in bootstrapSample]

    # convert to integers
    X_bootstrap = [[int(feature) for feature in sample] for sample in X_bootstrap]
    y_bootstrap = [int(label) for label in y_bootstrap]

    # fitting the decision tree to the data
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=None)  # a single decision tree without pruning it
    clf = clf.fit(X_bootstrap, y_bootstrap)

    for i, testSample in enumerate(X_test):
        testSample = [int(feature) for feature in testSample]

        # make the classifier prediction for each test sample and update classVotes
        class_predicted = clf.predict([testSample])[0]
        classVotes[i][class_predicted] += 1

        if k == 0 and class_predicted == y_test[i]:
            base_accuracy += 1

# Calculate the accuracy for the first base classifier
if base_accuracy > 0:
    base_accuracy = base_accuracy / len(y_test)
    print("Finished my base classifier (fast but relatively low accuracy) ...")
    print("My base classifier accuracy: " + str(base_accuracy))
    print("")

# Compare the final ensemble prediction for each test sample with the true label to calculate the accuracy
correct_ensemble_predictions = 0
for i in range(len(X_test)):
    # The predicted class is the one with the most votes
    predicted_class = classVotes[i].index(max(classVotes[i]))
    if predicted_class == int(y_test[i]):
        correct_ensemble_predictions += 1
ensemble_accuracy = correct_ensemble_predictions / len(y_test)

# Printing the ensemble accuracy here
print("Finished my ensemble classifier (slow but higher accuracy) ...")
print("My ensemble accuracy: " + str(ensemble_accuracy))
print("")

# Create a Random Forest Classifier
print("Started Random Forest algorithm ...")
clf = RandomForestClassifier(n_estimators=20)  # The number of decision trees in Random Forest

# Fit Random Forest to the training data
clf.fit(X_training, y_training)

# Make the Random Forest prediction for each test sample
correct_rf_predictions = 0
for i, testSample in enumerate(X_test):
    class_predicted_rf = clf.predict([testSample])[0]
    if class_predicted_rf == int(y_test[i]):
        correct_rf_predictions += 1

# Calculate Random Forest accuracy
rf_accuracy = correct_rf_predictions / len(y_test)

# Printing Random Forest accuracy here
print("Random Forest accuracy: " + str(rf_accuracy))
print("Finished Random Forest algorithm (much faster and higher accuracy!) ...")
