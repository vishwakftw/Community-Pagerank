""" Expected directory strucutre:
    raw_data/
        community1/
            person1.txt (any file name is fine)
            person2.txt
        community2/
            person3.txt
            ...
        ...
Processed and stored as:
    clean_data/
        same structure here
Processing:
    Conversion to lower case
    Removal of punctuations
    Removal of stop words
    Stemming
"""

from __future__ import print_function
from __future__ import division
import os
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from argparse import ArgumentParser as AP

p = AP()
p.add_argument('--dir_root', type=str, default='./raw_data', help='Root location of the raw data')
p.add_argument('--clean_data_root', type=str, default='./clean_data',
                help='Dir path for output')
p.add_argument('--verbose', action='store_true',
                help='option to print information at regular intervals | not helpful for large graphs')
p = p.parse_args()

verbose = p.verbose

porter = PorterStemmer()
stop_words = set(stopwords.words('english'))

for community in os.listdir(p.dir_root):
    print("Processing directory: " + community)
    community_path = os.path.join(p.dir_root, community)
    if os.path.isdir(community_path):
        new_dir = os.path.join(p.clean_data_root, community)
        if not os.path.exists(new_dir):
            os.mkdir(new_dir, mode=0o755)
        for person in os.listdir(community_path):
            print("\rProcessing file: " + person)
            file_path = os.path.join(community_path, person)
            if os.path.isfile(file_path):
                f = open(file_path, 'rt')
                text = f.read()
                f.close()
                
                tokens = word_tokenize(text)
                # convert to lower case
                tokens = [w.lower() for w in tokens]
                
                # remove punctuation from each word
                table = str.maketrans('', '', string.punctuation)
                stripped = [w.translate(table) for w in tokens]
                
                words = [w for w in stripped if not w in stop_words]
                
                stemmed = [porter.stem(word) for word in words]
                
                cleaned = [w for w in stemmed if w is not '']
                
                cleaned_file_path = os.path.join(new_dir, person)
                f = open(cleaned_file_path, 'wt')
                for w in cleaned:
                    f.write(w+" ")
                f.close()
                
                #print(cleaned)
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
