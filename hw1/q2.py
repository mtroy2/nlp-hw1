from scipy.misc import logsumexp
import os
import math
from autograd import grad
import autograd.numpy as np
from collections import namedtuple
from nltk import bigrams

symbols = set(['?','!',';',',','.'])
common_words = set( ['we','the','i','a','an','and','are','as','at','be','by','for','from','has','he','in','is','it','its','of','on','that','to','was','were','will','with'])

speaker_words   = {}  # speaker -> [word -> count]
class_instances = {}  # speaker -> instances of that class
class_words     = {}  # speaker -> all words in class
word_probs 	     = {}  # speaker -> [word -> p( w | k )]
class_probs	     = {}  # class 	-> p(k)
log_p_k_d       = {}

#for log_reg
word_lookup = {}
class_lookup = {}
total_classes = 0
vocabulary = []
train_set = open( os.getcwd() + '/hw1-data/train' )
for line in train_set:
    total_classes += 1

    line = line.split()
    line = [word for word in line if word not in common_words]
    line = [word for word in line if word not in symbols]
    speaker = line[0]
    if speaker not in speaker_words
    vocabulary.extend(line[1:])

vocabulary = set(vocabulary)

