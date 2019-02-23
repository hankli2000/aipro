#*****************************************************
# This file process data "car.csv". It predicts whether a car is "good","very good" or "unacceptable"
# It takes the last column as the output class, and the rest columns the input features
#******************************************************************************

#**************************************
#This part imports all the important libraries such as sklearn and numpy
#**************************************
import numpy as np   
import sklearn
# This specifies a particular sub-folder in sklearn that we will use
from sklearn import tree
#Triple quote can comment out multiple lines
""" We don't need the library os in this class  
import os
"""

#************************************
#This part loads csv file, and convert it into two matrices
# The first matrix X is our input data, the second matrix target is the output, or "class"
#****************************************
import csv
def load_data(filename):
    x=[]
    target=[]
    a=[]
    with open(filename) as csv_file:
        data_file = csv.reader(csv_file)
        for row in data_file:
            a.append(row)
    #print a[0]
    np_a=np.array(a)
    x=np_a[:,:-1]
    target=np_a[:,-1]
    return x, target
X, target=load_data('car.csv')

#*** Inspect data
n_samples=len(X)
n_features = len(X[0])
print n_samples
print n_features

#*** processing categorical data
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()

def convert_categorical(X):
    n_features = len(X[0])
    X_new=le.fit_transform(X[:,0])  # Take the first column
    for i in range (1, n_features):  #add remaining columns
        Xi=le.fit_transform(X[:,i]) 
        X_new=np.column_stack((X_new,Xi))
    return X_new
X_num=convert_categorical(X)
y_num=le.fit_transform(target)
print X
print X_num
print target
print y_num
print le.inverse_transform(y_num)

#*** Split into training and testing data
from sklearn.cross_validation import train_test_split
XTrain, XTest, yTrain, yTest = train_test_split(X_num, y_num, test_size=0.1, random_state=0)
print len(XTrain)
print len(XTest)
#print le.inverse_transform(yTest)


#*** start classification 
model=tree.DecisionTreeClassifier()   #initialize model
model.fit(XTrain, yTrain)  #It updates model

#*** Test
yPred=model.predict(XTest)
print le.inverse_transform(yPred)  #Get the original label name

#*** Get Accuracy
from sklearn.metrics import accuracy_score
accu=accuracy_score(yTest, yPred)
print accu

"""
"""
"""
#*** Visualize the model
tree.export_graphviz(model, out_file='car.dot')
cmd='dot -Tpdf car.dot -o car.pdf'
os.system(cmd)
"""
