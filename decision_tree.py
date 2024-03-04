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
from sklearn.metrics import accuracy_score


# Function to preprocess data: One-hot encode 'Marital Status' and convert 'Taxable Income' to float
def preprocess_data(df):
    # One-hot encoding for 'Marital Status'
    df = pd.get_dummies(df, columns=['Marital Status'])
    # Convert 'Taxable Income' from format '125k' to float (125.0)
    df['Taxable Income'] = df['Taxable Income'].str.replace('k', '').astype(float) * 1.0
    return df


# Function to convert 'Cheat' column to numerical values
def convert_cheat_to_numeric(cheat_value):
    return 1 if cheat_value == 'Yes' else 2


dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv']
testData = 'cheat_test.csv'

# Loop over datasets
for ds in dataSets:
    accuracies = []

    df = pd.read_csv(ds, sep=',', header=0)
    df = preprocess_data(df)

    # Preparing X and Y
    X = df.drop(columns=['Cheat']).values[:, 1:]  # Exclude the 'Tid' column from features
    Y = df['Cheat'].apply(convert_cheat_to_numeric).values

    # Loading and preprocessing the test data
    df_test = pd.read_csv(testData, sep=',', header=0)
    df_test = preprocess_data(df_test)
    X_test = df_test.drop(columns=['Cheat']).values[:, 1:]  # Exclude the 'Tid' column from features
    Y_test = df_test['Cheat'].apply(convert_cheat_to_numeric).values

    # Training the decision tree
    clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=None)
    clf.fit(X, Y)

    # Making predictions on the test set
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    accuracies.append(accuracy)

    # Plotting the decision tree
    tree.plot_tree(clf, feature_names=df.drop(columns=['Cheat']).columns[1:], class_names=['No', 'Yes'], filled=True,
                   rounded=True)  # Exclude 'Tid' from feature names
    plt.show()

    # Calculating the average accuracy over the 10 runs (here, we only have one run for simplicity)
    average_accuracy = np.mean(accuracies)
    print(f'Final accuracy when training on {ds}: {average_accuracy}')


'''
#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv']

for ds in dataSets:

    X = []
    Y = []

    df = pd.read_csv(ds, sep=',', header=0)   #reading a dataset eliminating the header (Pandas library)
    data_training = np.array(df.values)[:,1:] #creating a training matrix without the id (NumPy library)

    #transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
    #Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [2, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
    #be converted to a float.
    # X =

    #transform the original training classes to numbers and add them to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    # Y =

    #loop your training and test tasks 10 times here
    for i in range (10):

       #fitting the decision tree to the data by using Gini index and no max_depth
       clf = tree.DecisionTreeClassifier(criterion = 'gini', max_depth=None)
       clf = clf.fit(X, Y)

       #plotting the decision tree
       tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'], class_names=['Yes','No'], filled=True, rounded=True)
       plt.show()

       #read the test data and add this data to data_test NumPy
       #--> add your Python code here
       # data_test =

       for data in data_test:
           #transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
           #class_predicted = clf.predict([[1, 0, 1, 0, 115]])[0], where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           #--> add your Python code here

           #compare the prediction with the true label (located at data[3]) of the test instance to start calculating the model accuracy.
           #--> add your Python code here

       #find the average accuracy of this model during the 10 runs (training and test set)
       #--> add your Python code here

    #print the accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on cheat_training_1.csv: 0.2
    #--> add your Python code here'''




