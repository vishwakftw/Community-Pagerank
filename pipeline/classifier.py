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
"""

from __future__ import print_function
from __future__ import division
import os
import string
from argparse import ArgumentParser as AP
from sklearn.feature_extraction.text import TfidfTransformer

p = AP()
p.add_argument('--clean_data_root', type=str, default='./clean_data',
                help='Dir path for output')
p.add_argument('--train', type=str, default='./trainset.txt',
                help='Trainset file')
p.add_argument('--test', type=str, default='./testset.txt',
                help='Testset file')
p.add_argument('--verbose', action='store_true',
                help='option to print information at regular intervals | not helpful for large graphs')
p = p.parse_args()

verbose = p.verbose
trainset = p.train
clean_data = p.clean_data_root
train = []

with open(trainset, 'r') as trainfile:
    count = 0
    for line in trainfile:
        train.append([])
        train[count] = line.split()
    count += 1

train_x = []
train_y = []   
for i in range(len(train)):
    dir_path = os.path.join(clean_data, "community"+(i+1))
    for j in train[i]:
        train_x.append(os.path.join(dir_path, j))
        train_y.append(i)

vectorizer = TfidfTransformer(input='filename')

