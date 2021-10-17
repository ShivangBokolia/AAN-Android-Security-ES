# @author: Shivang Bokolia
# This file consists of a machine learning model ie, MLP Classifier that has been trained
# over a training dataset and provides predictions that are compared with the testing datasets.
# The dataset is obtained from the ES Data.xlsx file that is read on line 36.
# The output includes the Training set Accuracy, Confusion Matrix for test dataset, ANN model
# Accuracy and the loss curve graph for the same model.

# The number of neurons for the MLP Classifier can be changed on line 48. For adding another
# hidden layer, add the number of neurons to hidden_layer_sizes in the following manner:
# (first_hidden_layer_neutrons, second_hidden_layer_neutrons, ... )

from memory_profiler import profile
import warnings
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report


# Function for calculating the accuracy through the confusion matrix obtained.
def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements


@profile
def function():
    warnings.filterwarnings(action='ignore')

    # Read the excel file and collect the data
    excel_file = pd.read_excel("ES Data.xlsx", sheet_name="Sheet1")
    excel_content = excel_file.drop('Final Score', axis=1)

    # Setting the X values of the data and the Y values of the data
    X = excel_content.drop('Security Level', axis=1)
    Y = excel_content[['Security Level']]

    # Split the data into training set (0.67%) and testing set (0.33%)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=0)

    # Setting the MLP Classifier
    # The number of hidden layer neutrons can be changed in the hidden_layer_sizes argument.
    mlp = MLPClassifier(hidden_layer_sizes=192, max_iter=1000, activation='tanh', solver='adam',
                        learning_rate='constant', learning_rate_init=0.001, random_state=0)

    # Training the set and checking the accuracy for the training set
    start_training = time.time()
    scores = cross_val_score(mlp, X_train, Y_train.values.ravel(), cv=5)
    print("-------------- Training Accuracy --------------")
    print("%0.2f accuracy with a standard deviation of %0.2f\n" % (scores.mean(), scores.std()))
    end_training = time.time()

    # Fitting the data into the model
    start_testing = time.time()
    mlp.fit(X_train, Y_train.values.ravel())

    # Predicting the values for the test set
    y_pred = mlp.predict(X_test)

    # Checking if the predictions match the test set and presenting the results
    cm = confusion_matrix(Y_test.values, y_pred)
    end_testing = time.time()
    print('-------------- Confusion Matrix --------------')
    print(cm)
    print('\n-------------- ANN Model Accuracy --------------')
    print(accuracy(cm) * 100)
    print('\n-------------- ANN Model Report --------------')
    print(classification_report(Y_test, y_pred))

    # Plotting the loss curve of the MLP Classifier model
    plt.plot(mlp.loss_curve_)
    plt.xlabel("Number of steps")
    plt.ylabel("loss function")
    plt.show()

    # Time Consumption of the MLP Classifier model
    print("\nTest timing (time units): %0.2fs" % (end_testing - start_testing))
    print("Test timing (epoch numbers): %0.5f" % end_testing)
    print("Train timing (time units): %0.2fs" % (end_training - start_training))
    print("Train timing (epoch numbers): %0.5f" % end_training)


if __name__ == '__main__':
    function()
