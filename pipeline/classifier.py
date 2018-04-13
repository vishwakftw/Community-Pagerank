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
import itertools
from math import sqrt
from argparse import ArgumentParser as AP

import numpy as np
import scipy.sparse as ssp
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

def _check_exists(feat, number):
    a1 = os.path.exists('./train_features_x-{}-{}.npz'.format(feat, number))
    a2 = os.path.exists('./test_features_x-{}-{}.npz'.format(feat, number))
    a3 = os.path.exists('./train_features_y-{}-{}.npy'.format(feat, number))
    a4 = os.path.exists('./test_features_y-{}-{}.npy'.format(feat, number))
    return a1 and a2 and a3 and a4

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
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.seismic)
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
p.add_argument('--op', type=str, default='../OUTPUTS/classifier_acc.txt',
               help='Path of output file')
p.add_argument('--show_confusion_matrix', action='store_true',
               help='Option to show confusion matrix')
p.add_argument('--verbose', action='store_true',
               help='option to print information at regular intervals')
p.add_argument('--feature', type=str,
                choices=['best', 'random', 'tfbest', 'tfrandom', 'lda'], default='best', 
                help='Feature type to use. Default: K Best from TFIDF')
p.add_argument('--classifier', type=str,
               choices=['naive-bayes', 'mlp-1', 'mlp-2', 'knn', 'logistic',
                        'ovo-svm', 'ovo-logistic', 'ovr-svm', 'ovr-logistic'],
               default='naive-bayes', help='classifier option')
p.add_argument('--nfeatures', type=int, default=10000, help='Number of top features to extract')
p.add_argument('--save_load', action='store_true',
               help='Toggle to save / load features to prevent redundant computation')
p = p.parse_args()

verbose = p.verbose
trainset = p.train
testset = p.test
clean_data = p.clean_data_root
feature = p.feature
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

if _check_exists(feature, p.nfeatures) and p.save_load:
    if verbose:
        print("Loading from files...")
    train_x_stf = ssp.load_npz('./train_features_x-{}-{}.npz'.format(feature, p.nfeatures))
    train_x_stf = train_x_stf.toarray()
    train_y = np.load('./train_features_y-{}-{}.npy'.format(feature, p.nfeatures))
    test_x_stf = ssp.load_npz('./test_features_x-{}-{}.npz'.format(feature, p.nfeatures))
    test_x_stf = test_x_stf.toarray()
    test_y = np.load('./test_features_y-{}-{}.npy'.format(feature, p.nfeatures))
else:
    if feature != 'lda':
        is_idf = False if feature == 'tfbest' or feature == 'tfrandom' else True
        vectorizer = TfidfVectorizer(input='filename', use_idf=is_idf)
        selector = SelectKBest(mutual_info_classif, k=p.nfeatures)
        
        train_x_tf = vectorizer.fit_transform(train_x)
        if 'random' in feature:
            train_y = [1 for x in range(len(train_x))]
        train_x_stf = selector.fit_transform(train_x_tf, train_y)
        test_x_tf = vectorizer.transform(test_x)
        test_x_stf = selector.transform(test_x_tf)
        if p.save_load:
            if verbose:
                print("Saving to files...")
            ssp.save_npz('./train_features_x-{}-{}.npz'.format(feature, p.nfeatures), train_x_stf)
            train_y = np.array(train_y)
            train_y.dump('./train_features_y-{}-{}.npy'.format(feature, p.nfeatures))
            ssp.save_npz('./test_features_x-{}-{}.npz'.format(feature, p.nfeatures), test_x_stf)
            test_y = np.array(test_y)
            test_y.dump('./test_features_y-{}-{}.npy'.format(feature, p.nfeatures))
    else:
        print('ERROR: lda not yet implemented')
        exit(0)

if verbose:
    print("Shape of training data: {}".format(train_x_stf.shape))
    print("Shape of testing data: {}".format(test_x_stf.shape))

# Classifier region
if p.classifier == 'naive-bayes':
    clf = MultinomialNB().fit(train_x_stf, train_y)

elif p.classifier == 'mlp-1':
    clf = MLPClassifier(solver='lbfgs',
                        hidden_layer_sizes=(int(sqrt(p.nfeatures)),)
                       ).fit(train_x_stf, train_y)

elif p.classifier == 'mlp-2':
    clf = MLPClassifier(solver='lbfgs',
                        hidden_layer_sizes=(2 * int(sqrt(p.nfeatures)), int(sqrt(p.nfeatures)))
                       ).fit(train_x_stf, train_y)

elif p.classifier == 'knn':
    clf = KNeighborsClassifier(n_neighbors=11).fit(train_x_stf, train_y)

elif p.classifier == 'logistic':
    clf = LogisticRegression(class_weight='balanced',
                             multi_class='multinomial', solver='sag').fit(train_x_stf, train_y)

elif p.classifier == 'ovo-svm':
    base_clf = SVC(kernel='rbf', class_weight='balanced')
    clf = OneVsOneClassifier(base_clf).fit(train_x_stf, train_y)

elif p.classifier == 'ovo-logistic':
    base_clf = LogisticRegression(class_weight='balanced')
    clf = OneVsOneClassifier(base_clf).fit(train_x_stf, train_y)

elif p.classifier == 'ovr-svm':
    base_clf = SVC(kernel='rbf', class_weight='balanced')
    clf = OneVsRestClassifier(base_clf).fit(train_x_stf, train_y)

elif p.classifier == 'ovr-logistic':
    base_clf = LogisticRegression(class_weight='balanced')
    clf = OneVsRestClassifier(base_clf).fit(train_x_stf, train_y)

if(verbose):
    print("Training complete")

pred = clf.predict(test_x_stf)

print("Testing set accuracy obtained: ", end='')
acc = np.mean(pred == test_y)
print(acc)
with open(p.op, 'a') as f:
    f.write('{}\t{}\t{}\n'.format(p.classifier, p.nfeatures, round(acc, 7)))

if p.show_confusion_matrix:
    cnf_mat = confusion_matrix(test_y, pred)
    plt.figure()
    plot_confusion_matrix(cnf_mat, len(train))
    plt.savefig('confusion_matrix[clf={},feature={},nfeatures={}].png'.format(p.classifier, feature, p.nfeatures),
                dpi=100)
