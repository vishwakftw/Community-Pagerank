""" 
Input:
* trainset.txt:
    <IDofPerson1ofcommunity1> <IDofPerson2ofcommunity1> ..
    <IDofPerson1ofcommunity2> <IDofPerson2ofcommunity2> ..
    ...
* testset.txt (Assuming 45 people go to trainset per community)
    <IDofPerson46ofcommunity1> <IDofPerson47ofcommunity1> ..
    <IDofPerson46ofcommunity2> <IDofPerson47ofcommunity2> ..
    ...
path of clean data

Output:
Accuracy on test set
"""

from __future__ import print_function
from __future__ import division
import os
import string
from argparse import ArgumentParser as AP
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

p = AP()
p.add_argument('--clean_data_root', type=str, default='./clean_data',
                help='Dir path for output')
p.add_argument('--train', type=str, default='./trainset.txt',
                help='Trainset file')
p.add_argument('--test', type=str, default='./testset.txt',
                help='Testset file')
p.add_argument('--op', type=str, default='sys.stdout', help='Path of output file')
p.add_argument('--verbose', action='store_true',
                help='option to print information at regular intervals | not helpful for large graphs')
p = p.parse_args()

verbose = p.verbose
trainset = p.train
testset = p.test
clean_data = p.clean_data_root
train = []
test = []

with open(trainset, 'rt') as trainfile:
    count = 0
    for line in trainfile:
        line = line.strip('\n')
        train.append([])
        train[count] = line.split()
        # train[count][-1] = train[count][-1]
        count += 1

# print(train)
train_x = []
train_y = []   
for i in range(len(train)):
    dir_path = os.path.join(clean_data, "community"+str(i))
    for j in train[i]:
        train_x.append(os.path.join(dir_path, j+".txt"))
        train_y.append(i)
        
with open(testset, 'rt') as testfile:
    count = 0
    for line in testfile:
        line = line.strip('\n')
        test.append([])
        test[count] = line.split()
        count += 1

# print(train)
test_x = []
test_y = []   
for i in range(len(test)):
    dir_path = os.path.join(clean_data, "community"+str(i))
    for j in train[i]:
        test_x.append(os.path.join(dir_path, j+".txt"))
        test_y.append(i)

vectorizer = TfidfVectorizer(input='filename')
train_x_tf = vectorizer.fit_transform(train_x)
print(train_x_tf.shape)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(train_x_tf, train_y)

if(verbose):
    print("Training complete")
    
test_x_tf = vectorizer.transform(test_x)
pred = clf.predict(test_x_tf)
print("Testset accuracy obtained: ", end='', file=p.op_file)
print(np.mean(pred == test_y))
