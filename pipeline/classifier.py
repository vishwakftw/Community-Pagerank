"""
Input:
* trainset.txt:
    0 <IDofPerson1ofcommunity0> <IDofPerson2ofcommunity0> ..
    1 <IDofPerson1ofcommunity1> <IDofPerson2ofcommunity1> ..
    ...
* testset.txt (Assuming 45 people go to trainset per community)
    0 <IDofPerson46ofcommunity0> <IDofPerson47ofcommunity0> ..
    1 <IDofPerson46ofcommunity1> <IDofPerson47ofcommunity1> ..
    ...
path of clean data

Note: The order of communities in trainset / testset does not matter,
i.e. first line can be : 25 ID1.txt ID2.txt etc. but the numbering of
communites must follow the order 0, 1, .. n, where (n + 1 would be
the number of lines in file)

Output:
Accuracy on test set
"""
from __future__ import print_function
from __future__ import division
import os
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
from argparse import ArgumentParser as AP
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

def _lines_in_file(file_path):
    with open(file_path, 'r') as f:
        for i, _ in enumerate(f):
            pass
    return i + 1

def plot_confusion_matrix(cm, n_classes):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.bwr)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(0, n_classes, 5)
    plt.xticks(tick_marks, np.arange(0, n_classes, 5), rotation=45)
    plt.yticks(tick_marks, np.arange(0, n_classes, 5))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

p = AP()
p.add_argument('--clean_data_root', type=str, default='./clean_data',
               help='Dir path for output')
p.add_argument('--train', type=str, default='./trainset.txt',
               help='Trainset file')
p.add_argument('--test', type=str, default='./testset.txt',
               help='Testset file')
p.add_argument('--op', type=str, default='../OUTPUTS/classifier_acc.txt', help='Path of output file')
p.add_argument('--show_confusion_matrix', action='store_true', help='Option to show confusion matrix')
p.add_argument('--verbose', action='store_true',
               help='option to print information at regular intervals | not helpful for large graphs')
p.add_argument('--classifier', type=str, choices=['naive-bayes', 'mlp', 'random-forest', 'adaboost'],
               default='naive-bayes', help='classifier option')
p.add_argument('--nfeatures', type=int, default=10000, help='Number of top features to extract')
p = p.parse_args()

verbose = p.verbose
trainset = p.train
testset = p.test
clean_data = p.clean_data_root
train = [[] for i in range(_lines_in_file(trainset))]
test = [[] for i in range(_lines_in_file(testset))]

with open(trainset, 'rt') as trainfile:
    for line in trainfile:
        line = line.strip('\n')
        people = line.split()
        train[int(people[0])] = people[1:]

train_x = []
train_y = []
for i in range(len(train)):
    dir_path = os.path.join(clean_data, "community" + str(i))
    for j in train[i]:
        filepath = os.path.join(dir_path, j + ".txt")
        if os.path.exists(filepath):
            train_x.append(filepath)
            train_y.append(i)
        else:
            print("[TRAIN SET] community" + str(i), j + ".txt", "not found.")

with open(testset, 'rt') as testfile:
    for line in testfile:
        line = line.strip('\n')
        people = line.split()
        test[int(people[0])] = people[1:]

# print(train)
test_x = []
test_y = []
for i in range(len(test)):
    dir_path = os.path.join(clean_data, "community" + str(i))
    for j in test[i]:
        filepath = os.path.join(dir_path, j + ".txt")
        if os.path.exists(filepath):
            test_x.append(filepath)
            test_y.append(i)
        else:
            print("[TEST SET] community" + str(i), j + ".txt", "not found.")

if verbose:
    print("Reading train and test file complete")

vectorizer = TfidfVectorizer(input='filename')
selector = SelectKBest(mutual_info_classif, k=p.nfeatures)

train_x_tf = vectorizer.fit_transform(train_x)
train_x_stf = selector.fit_transform(train_x_tf, train_y)
if verbose:
    print("Shape of training data: {}".format(train_x_stf.shape))

# Classifier region
if p.classifier == 'naive-bayes':
    clf = MultinomialNB().fit(train_x_stf, train_y)

elif p.classifier == 'mlp':
    clf = MLPClassifier(solver='lbfgs',
                        hidden_layer_sizes=(int(sqrt(p.nfeatures)),
                                            len(train))
                       ).fit(train_x_stf, train_y)
elif p.classifier == 'random-forest':
    base_clf = RandomForestClassifier()
    param_grid = {"n_estimators": [10, 20, 40, 100, 200],
                  "max_depth": [3, 5, None],
                  "max_features": [1, 3, 10],
                  "min_samples_split": [2, 3, 5],
                  "min_samples_leaf": [3, 5, 10],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}
    clf = GridSearchCV(base_clf, param_grid=param_grid).fit(train_x_stf, train_y)

elif p.classifier == 'adaboost':
    base_clf = AdaBoostClassifier().fit(train_x_stf, train_y)

if(verbose):
    print("Training complete")

test_x_tf = vectorizer.transform(test_x)
test_x_stf = selector.transform(test_x_tf)
if verbose:
    print(" Shape of testing data: {}".format(test_x_stf.shape))
pred = clf.predict(test_x_stf)

print("Testing set accuracy obtained: ", end='')
acc = np.mean(pred == test_y)
print(acc)
with open(p.op, 'wt') as f:
    f.write("Testset accuracy obtained: " + str(acc))

if p.show_confusion_matrix:
    cnf_mat = confusion_matrix(test_y, pred)
    plt.figure()
    plot_confusion_matrix(cnf_mat, len(train))
    plt.savefig('confusion_matrix.png', dpi=100)
