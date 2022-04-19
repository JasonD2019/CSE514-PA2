# CSE 514 Data Mining 
## _Xingjian Ding 502558 JasonD2019_
# Program Assignment 2
##### How to run
After download all files
```sh
python PA2.py
```

# Different modules 
To impletement different classification modules in this program, comment and uncomment code in function addModule(). Each time uncomment one kind of the modules to make the polt looks good. Too many modules will make the plot hard to read and understand.
```
# kNN (KNeighborsClassifier)
n_nerighbors = [3,4,5,6,7]
for n in n_nerighbors:
    models.append(('k = %s' % n, KNeighborsClassifier(n_neighbors = n)))
algorithms = ["auto", "ball_tree", "kd_tree", "brute"]
for n in algorithms:
    models.append(('%s' % n, KNeighborsClassifier(algorithm = n)))

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

# # NN
# models.append(('HL=100 ', MLPClassifier(max_iter=1000, hidden_layer_sizes=(100,))))
# models.append(('HL=200 ', MLPClassifier(max_iter=1000, hidden_layer_sizes=(200,))))
# models.append(('HL=300 ', MLPClassifier(max_iter=1000, hidden_layer_sizes=(300,))))
# models.append(('HL=400 ', MLPClassifier(max_iter=1000, hidden_layer_sizes=(400,))))
# models.append(('HL=500 ', MLPClassifier(max_iter=1000, hidden_layer_sizes=(500,))))
# models.append(('identity', MLPClassifier(max_iter=1000, activation="identity")))
# models.append(('logistic', MLPClassifier(max_iter=1000, activation="logistic")))
# models.append(('tanh', MLPClassifier(max_iter=1000, activation="tanh")))
# models.append(('relu', MLPClassifier(max_iter=1000, activation="relu")))

# # ETC
# models.append(('#=50', ExtraTreesClassifier(n_estimators=50)))
# models.append(('#=75', ExtraTreesClassifier(n_estimators=75)))
# models.append(('#=100', ExtraTreesClassifier(n_estimators=100)))
# models.append(('#=125', ExtraTreesClassifier(n_estimators=125)))
# models.append(('#=150', ExtraTreesClassifier(n_estimators=150)))

# # QDA
# models.append(('RP=0.0', QuadraticDiscriminantAnalysis(reg_param=0.0)))
# models.append(('RP=0.2', QuadraticDiscriminantAnalysis(reg_param=0.2)))
# models.append(('RP=0.4', QuadraticDiscriminantAnalysis(reg_param=0.4)))
# models.append(('RP=0.6', QuadraticDiscriminantAnalysis(reg_param=0.6)))
# models.append(('RP=0.8', QuadraticDiscriminantAnalysis(reg_param=0.8)))
```
