import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
import statistics

# reading csv files
def read_data():
    newHeader = ("letter","x-box", "y-box", "width", "high", "onpix","x-bar", "y-bar", "x2bar", "y2bar", "xybar", "x2ybr","xy2br", "x-ege", "xegvy", "y-ege","yegvx")
    data =  pd.read_csv('data/letter-recognition.data', header=None, names=newHeader)
    return data

# A) Data preprocessing
# This dataset contains 26 classes to separate, but for this assignment, we’ll simplify to three
# binary classification problems.
# For each pair, set aside 10% of the relevant samples to use as a final validation set.

# Pair 1: H and K
# Pair 2: M and Y
# Pair 3: Your choice : V and Us


# find all the relevant samples
def findSamples(data, A, B):
    search_values1 = [A,B]
    newData =data[data.letter.str.contains('|'.join(search_values1))]
    return newData

# set aside 10% of those samples for final validation of the models
def setSamples(data):
    array = data.values
    X = array[:,1:17]
    y = array[:,0]
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.10)
    return X_train, X_validation, Y_train, Y_validation

data = read_data()
data1 = findSamples(data,'H','K')
data2 = findSamples(data,'M','Y')
data3 = findSamples(data,'U','V')
X_train1, X_validation1, Y_train1, Y_validation1 = setSamples(data1)
X_train2, X_validation2, Y_train2, Y_validation2 = setSamples(data2)
X_train3, X_validation3, Y_train3, Y_validation3 = setSamples(data3)

# B) Model fitting
# For this project, you must consider the following classification models:
# 1. k-nearest neighbors
# 2. Decision tree
# 3. Random Forest
# 4. SVM
# 5. Artificial Neural Network
# For each model, choose a hyperparameter to tune using 5-fold cross-validation. You must test at
# least 3 values for a categorical hyperparameter, and at least 5 values for a numerical one.
# Hyperparameter tuning should be done separately for each classification problem; you might end
# up with different values for classifying H from K than for classifying M from Y.

models = []
def addModels():
    # # kNN (KNeighborsClassifier)
    # n_nerighbors = [3,4,5,6,7]
    # for n in n_nerighbors:
    #     models.append(('k = %s' % n, KNeighborsClassifier(n_neighbors = n)))
    # algorithms = ["auto", "ball_tree", "kd_tree", "brute"]
    # for n in algorithms:
    #     models.append(('%s' % n, KNeighborsClassifier(algorithm = n)))

    # # Decision Tree
    # max_depth = [10, 20, 30, 40]
    # for n in max_depth:
    #     models.append(('MD=%s' % n, DecisionTreeClassifier(max_depth=n)))
    # min_samples_leaf = [1,2,3,4,5]
    # for n in min_samples_leaf:
    #     models.append(('ML=%s' % n, DecisionTreeClassifier(min_samples_leaf=n)))

    # # Random Forest
    # models.append(('#=50 ', RandomForestClassifier(n_estimators=50)))
    # models.append(('#=75 ', RandomForestClassifier(n_estimators=75)))
    # models.append(('#=100 ', RandomForestClassifier(n_estimators=100)))
    # models.append(('#=125 ', RandomForestClassifier(n_estimators=125)))
    # models.append(('#=150 ', RandomForestClassifier(n_estimators=150)))
    # models.append(('MD=10', RandomForestClassifier(n_estimators=100, max_depth=10)))
    # models.append(('MD=20', RandomForestClassifier(n_estimators=100, max_depth=20)))
    # models.append(('MD=30', RandomForestClassifier(n_estimators=100, max_depth=30)))
    # models.append(('MD=40', RandomForestClassifier(n_estimators=100, max_depth=40)))
    # models.append(('MD=50', RandomForestClassifier(n_estimators=100, max_depth=50)))

    # # SVM
    # models.append(('C=0.6 ', SVC(gamma='auto', C=0.6)))
    # models.append(('C=0.8 ', SVC(gamma='auto', C=0.8)))
    # models.append(('C=1.0 ', SVC(gamma='auto', C=1.0)))
    # models.append(('C=1.2 ', SVC(gamma='auto', C=1.2)))
    # models.append(('C=1.4 ', SVC(gamma='auto', C=1.4)))
    # models.append(('linear', SVC(gamma='auto', kernel='linear')))
    # models.append(('poly', SVC(gamma='auto', kernel='poly')))
    # models.append(('rbf', SVC(gamma='auto', kernel='rbf')))
    # models.append(('sigmoid', SVC(gamma='auto', kernel='sigmoid')))

    # NN
    # models.append(('HL=100 ', MLPClassifier(max_iter=1000, hidden_layer_sizes=(100,))))
    # models.append(('HL=200 ', MLPClassifier(max_iter=1000, hidden_layer_sizes=(200,))))
    # models.append(('HL=300 ', MLPClassifier(max_iter=1000, hidden_layer_sizes=(300,))))
    # models.append(('HL=400 ', MLPClassifier(max_iter=1000, hidden_layer_sizes=(400,))))
    # models.append(('HL=500 ', MLPClassifier(max_iter=1000, hidden_layer_sizes=(500,))))
    # models.append(('identity', MLPClassifier(max_iter=1000, activation="identity")))
    # models.append(('logistic', MLPClassifier(max_iter=1000, activation="logistic")))
    # models.append(('tanh', MLPClassifier(max_iter=1000, activation="tanh")))
    # models.append(('relu', MLPClassifier(max_iter=1000, activation="relu")))

    ## ETC
    # models.append(('#=50', ExtraTreesClassifier(n_estimators=50)))
    # models.append(('#=75', ExtraTreesClassifier(n_estimators=75)))
    # models.append(('#=100', ExtraTreesClassifier(n_estimators=100)))
    # models.append(('#=125', ExtraTreesClassifier(n_estimators=125)))
    # models.append(('#=150', ExtraTreesClassifier(n_estimators=150)))

    # QDA
    models.append(('RP=0.0', QuadraticDiscriminantAnalysis(reg_param=0.0)))
    models.append(('RP=0.2', QuadraticDiscriminantAnalysis(reg_param=0.2)))
    models.append(('RP=0.4', QuadraticDiscriminantAnalysis(reg_param=0.4)))
    models.append(('RP=0.6', QuadraticDiscriminantAnalysis(reg_param=0.6)))
    models.append(('RP=0.8', QuadraticDiscriminantAnalysis(reg_param=0.8)))

addModels()

def pred(title, models, X_train, Y_train, X_validation, Y_validation):
    meanList = []
    nameList = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)
        result = accuracy_score(Y_validation, predictions)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        # print("%s: %f (std: %f)" % (name, cv_results.mean(), cv_results.std()))
        meanList.append(cv_results.mean())
        nameList.append(name)
        # print("Without 5-fold cross-validation:   %f" % result)
        # print('After 5-fold cross-validation:     %f (std: %f)' % ( cv_results.mean(), cv_results.std()))
    plt.scatter(nameList, meanList)
    plt.ylabel('accuracy score')
    plt.title(title)
    for i,j in zip(nameList,meanList):
        plt.annotate(str(format(j, '.3f')),xy=(i,j))
    # plt.show()
    print("---------------------------")
    print(statistics.mean(meanList))


# C) Dimension reduction
# For each of the models, implement a method of dimension reduction from the following:
# 1. Simple Quality Filtering
# 2. Filter Methods
# 3. Wrapper Feature Selection
# 4. Embedded Methods
# 5. Feature Extraction
# Please refer to the lecture slides for more details on the methods. Implement a total of at least 3
# different methods to reduce the number of features from 16 to 4. Retrain your models using
# reduced datasets, including hyperparameter tuning.

# 1. Simple Quality Filtering - Low variance
from sklearn.feature_selection import VarianceThreshold
def SQF_lowVar(X_data):
    constant_filter = VarianceThreshold(threshold=2)
    constant_filter.fit(X_data)
    X_data = constant_filter.transform(X_data)
    return X_data

# 2. Filter Methods - Unsupervised approach
def UnsupervisedFilter(data):
    correlation=data.corr(method="pearson")
    max = [-1,-1,-1]
    for i in range(16):
        for j in range(16):
            if abs(correlation.values[i][j]) > max[0] and i!=j:
                max[0] = abs(correlation.values[i][j])
                max[1] = i
                max[2] = j
    # print('[%i, %i]: %f' % (max[1], max[2], max[0]))
    data.drop(data.columns[[max[1]+1]], axis = 1, inplace = True)
    return data

# 4. Embedded Methods
from sklearn.feature_selection import SelectKBest, chi2
def EmbeddedMethods(X, Y):
    X_newTrain = SelectKBest(chi2, k=4).fit_transform(X, Y)
    return X_newTrain

def NewsetSamples(data):
    array = data.values
    X = array[:,1:17]
    X = SQF_lowVar(X)
    y = array[:,0]
    X = EmbeddedMethods(X, y)
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.10)
    return X_train, X_validation, Y_train, Y_validation

import time
start = time.time()

# print("H and K before Dimension reduction: ")
# start1 = time.time()
# pred("H and K before Dimension reduction", models, X_train1, Y_train1, X_validation1, Y_validation1)
# end1 = time.time()
# print(end1 - start1)
#
# print("M and Y before Dimension reduction: ")
# start2 = time.time()
# pred("M and Y before Dimension reduction", models, X_train2, Y_train2, X_validation2, Y_validation2)
# end2 = time.time()
# print(end2 - start2)
#
# print("U and V before Dimension reduction: ")
# start3 = time.time()
# pred("U and V before Dimension reduction", models, X_train3, Y_train3, X_validation3, Y_validation3)
# end3 = time.time()
# print(end3 - start3)

print("H and K before Dimension reduction: ")
pred("H and K before Dimension reduction", models, X_train1, Y_train1, X_validation1, Y_validation1)
print("M and Y before Dimension reduction: ")
pred("M and Y before Dimension reduction", models, X_train2, Y_train2, X_validation2, Y_validation2)
print("U and V before Dimension reduction: ")
pred("U and V before Dimension reduction", models, X_train3, Y_train3, X_validation3, Y_validation3)

data = read_data()
data = UnsupervisedFilter(data)
data1 = findSamples(data,'H','K')
data2 = findSamples(data,'M','Y')
data3 = findSamples(data,'U','V')
X_train1, X_validation1, Y_train1, Y_validation1 = NewsetSamples(data1)
X_train2, X_validation2, Y_train2, Y_validation2 = NewsetSamples(data2)
X_train3, X_validation3, Y_train3, Y_validation3 = NewsetSamples(data3)


print("H and K after Dimension reduction: ")
pred("H and K after Dimension reduction", models, X_train1, Y_train1, X_validation1, Y_validation1)
print("M and Y after Dimension reduction: ")
pred("M and Y after Dimension reduction", models, X_train2, Y_train2, X_validation2, Y_validation2)
print("U and V after Dimension reduction: ")
pred("U and V after Dimension reduction", models, X_train3, Y_train3, X_validation3, Y_validation3)


# print("H and K after Dimension reduction: ")
# start4 = time.time()
# pred("H and K after Dimension reduction", models, X_train1, Y_train1, X_validation1, Y_validation1)
# end4 = time.time()
# print(end4 - start4)
#
# print("M and Y after Dimension reduction: ")
# start5 = time.time()
# pred("M and Y after Dimension reduction", models, X_train2, Y_train2, X_validation2, Y_validation2)
# end5 = time.time()
# print(end5 - start5)
#
# print("U and V after Dimension reduction: ")
# start6 = time.time()
# pred("U and V after Dimension reduction", models, X_train3, Y_train3, X_validation3, Y_validation3)
# end6 = time.time()
# print(end6 - start6)
#
# end = time.time()
# print(f"Runtime of the program is {end - start}")
