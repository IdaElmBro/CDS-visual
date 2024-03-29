import os
import sys
sys.path.append("../../..") 
import numpy as np
import cv2
from joblib import dump, load
import argparse
import utils.classifier_utils as clf_util
# Import sklearn metrics
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
# Visualisation
import matplotlib.pyplot as plt
# Data
from tensorflow.keras.datasets import cifar10


# defining the arguments, so it can be specified in the console if we want to run the neural network or logistic regression classifier
def classifier_type():
    parser = argparse.ArgumentParser(description="Make classification report")
    parser.add_argument("--classifier",
                        "-c",
                        required=True,
                        choices=["NN", "LR"],
                        help="Type of classifier: NN (Neural Network) or LR (logistic)")
    args = parser.parse_args()
    return args



def process(X_train, X_test):
    '''
    This function takes the training set and greyscales the images, scales the values
    and lastly flattens the images.
    '''
    # initialize empty lists
    X_train_results = []
    X_test_results = []
    # preprocessing
    for image in X_train:
        greyed_X_train = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # greyscale
        scaled_X_train = greyed_X_train/255.0 # scale
        X_train_results.append(scaled_X_train) # append to empty list
    reshaped_X_train = np.array(X_train_results).reshape(-1, 1024) # reshape the dimensions

    for image in X_test:
        greyed_X_test = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scaled_X_test = greyed_X_test/255.0
        X_test_results.append(scaled_X_test)
    reshaped_X_test = np.array(X_test_results).reshape(-1, 1024)

    return reshaped_X_train, reshaped_X_test



# firstly a function for training a neural network classifier
def train_classifier_NN(reshaped_X_train, y_train, reshaped_X_test, y_test): 
    '''
    This function takes the reshaped training data and the original test data as parameters.
    It trains a logistic regression classifier, saves a loss curve
    and the classification report with the correct label names. 
    '''
# logistic regression 
    clf = MLPClassifier(activation = "relu", # try relu, cause logistic gave pretty bad results - well, this does too, but it's a bit better... 
                           hidden_layer_sizes = (188,), 
                           max_iter=1000,
                           random_state = 42, 
                           verbose = True).fit(reshaped_X_train, y_train)


     #plot loss curve
    plt.plot(clf.loss_curve_)
    plt.title("Loss curve during training", fontsize=14)
    plt.xlabel('Iterations')
    plt.ylabel('Loss score')
    #save
    plt.savefig("../out/loss_curve.png")

    #calculate predictions for all data the scaled test data.
    y_pred = clf.predict(reshaped_X_test)

# get label names
    label_names = ["airplane", 
                    "automobile", 
                    "bird", 
                    "cat", 
                    "deer", 
                    "dog", 
                    "frog", 
                    "horse", 
                    "ship", 
                    "truck"] 
# report
    NN_report = metrics.classification_report(y_test, y_pred, target_names = label_names)
    
# save report 
    f = open('../out/classification_report_NN.txt', 'w') # open in write mode
    f.write(NN_report) # write the variable into the txt file 
    f.close() 
 


# next for training a logistic regression classifer
def train_classifier_LR(reshaped_X_train, y_train, reshaped_X_test, y_test): 
    '''
    This function takes the reshaped training data and the original test data as parameters.
    It trains a logistic regression classifier and saves the classification report with the correct label names. 
    '''
# logistic regression 
    clf = LogisticRegression(tol=0.1, 
                         solver='saga',
                         multi_class='multinomial').fit(reshaped_X_train, y_train)


    #calculate predictions for all data the scaled test data.
    y_pred = clf.predict(reshaped_X_test)

# get label names
    label_names = ["airplane", 
                    "automobile", 
                    "bird", 
                    "cat", 
                    "deer", 
                    "dog", 
                    "frog", 
                    "horse", 
                    "ship", 
                    "truck"] 
# report
    LR_report = metrics.classification_report(y_test, y_pred, target_names = label_names)
    
# save report 
    f = open('../out/classification_report_LR.txt', 'w') # open in write mode
    f.write(LR_report) # write the variable into the txt file 
    f.close() 



def main():
    '''
    The main function loads the data and uses the functions from above to preprocess the data
    and make the classification report based on the arguments given in the console. 
    '''
    args = classifier_type()

    #load data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    #process
    X_train_scaled, X_test_scaled = process(X_train, X_test)

    if args.classifier == "NN":
        # train neural network classifier, make report and loss curve
        train_classifier_NN(X_train_scaled, y_train, X_test_scaled, y_test)

    elif args.classifier == "LR":
        # train logistic regression classifier and make report 
        train_classifier_LR(X_train_scaled, y_train, X_test_scaled, y_test)


if __name__=="__main__": # if it's executed from the command line run the function "main", otherwise don't do anything 
    main()
