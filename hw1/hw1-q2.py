from autograd.scipy.misc import logsumexp
import os
import math
from autograd import grad
import autograd.numpy as np
from collections import namedtuple
from nltk import bigrams
from random import shuffle
symbols = set(['?','!',';',',','.'])
common_words = set( ['we','the','i','a','an','and','are','as','at','be','by','for','from','has','he','in','is','it','its','of','on','that','to','was','were','will','with'])


#for log_reg
word_lookup = {}
class_lookup = {}

vocabulary = []
classes = []

with open( os.getcwd() + '/hw1-data/train' ) as train_file:
    train_set = train_file.read().splitlines()
with open( os.getcwd() + '/hw1-data/dev' ) as dev_file:  
    dev_set = dev_file.read().splitlines()
with open( os.getcwd() + '/hw1-data/test' ) as test_file:
    test_set = test_file.read().splitlines()
def read_train(train_set, classes, vocabulary):
    """ Generates vocabulary and speakers  """
    for line in train_set:
        #tokenize line
        line = line.split()
        #remove stop-words and symbols
        line = [word for word in line if word not in common_words]
        line = [word for word in line if word not in symbols]
       
        speaker = line[0]

        # Build speaker list and vocabulary
        if speaker not in classes:
            # add to classes list
            classes.append(speaker)
        for word in line[1:]:
            # new word
            if word not in vocabulary:
                #add to vocab
                vocabulary.append(word)


def test_file(lambda_k_w, in_file):
    num_correct = 0.
    num_incorrect = 0.
    for line in in_file:
        #tokenize line
        line = line.split()
        #remove stop-words and symbols
        line = [word for word in line if word not in common_words]
        line = [word for word in line if word not in symbols]
        
        speaker = line[0]
        wc = np.zeros(len(vocabulary) + 1)
        for word in line[1:]:
            if word in vocabulary:
                wc[word_lookup[word]] += 1.
        wc[word_lookup['<b>']] = 1
        prob_array = np.dot(lambda_w_k, wc)
        max_index = 0
        lambda_max = 0
        index = 0
        for val in prob_array:
            if val > lambda_max:
                lambda_max = val
                max_index = index
            index += 1
        guess = classes[max_index]
        if guess == speaker:
            num_correct +=1
        else:
            num_incorrect += 1
    print('Accuracy was ' + str( num_correct / (num_incorrect+num_correct)))

def train_model(iterations,learning_rate,lambda_w_k):
    for i in range(0,iterations):
        lambda_sum=0
        for line in train_set:
            # count occurances of word in line
            word_counts = np.zeros(len(vocabulary)+1)
            line = line.split()
            #remove stop-words and symbols
            line = [word for word in line if word not in common_words]
            line = [word for word in line if word not in symbols]
            speaker = line[0]
            for word in line[1:]:
                if word in vocabulary:
                    word_counts[word_lookup[word]] += 1

            word_counts[word_lookup['<b>']] = 1
            lambda_w_k -= learning_rate * g_lambda_pk(lambda_w_k, word_counts, speaker)
            lambda_sum += n_log_pkd(lambda_w_k,word_counts,speaker)

        test_file(lambda_w_k, dev_set)
        print('negative log prob on iteration ' + str(i) + ' = ' + str(lambda_sum))
        shuffle(train_set)

def n_log_pkd(model, doc, correct_class):
    s = np.dot(model,doc)
    return -1 * ( s[class_lookup[correct_class]] - logsumexp(s) )

def apply_model(model,document):
    """ apply trained model to single line of file (document) """
    document = document.split()
    document = [word for word in document if word not in common_words]
    document = [word for word in document if word not in symbols]
    speaker = document[0]
    line = document[1:]
    prob_list = []
    wc = np.zeros(len(vocabulary)+1)
    for word in line:
        if word in vocabulary:
            wc[word_lookup[word]] += 1
    wc[word_lookup['<b>']] = 1
    for speaker in classes:
        prob_list.append(  math.exp(-1 * n_log_pkd(model,wc,speaker)) )
    return prob_list
if __name__ == "__main__":
    # Inital read of train document, sets vocabulary and # classes   
    read_train(train_set,classes,vocabulary)
    # array containing lamba(w | k )s for each class
    lambda_w_k = np.zeros((len(classes),len(vocabulary)+1),dtype=np.float64)

    c_map_val = 0
    w_map_val = 0
    # map classes and words to rows/ columns
    for c in classes:
        class_lookup[c] = c_map_val
        c_map_val += 1
    
    for w in vocabulary:
        word_lookup[w] = w_map_val
        w_map_val += 1

    # set bias word to be last value
    word_lookup['<b>'] = w_map_val



    g_lambda_pk = grad(n_log_pkd)
    learning_rate = 0.01
    iters = 25
    train_model(iters, learning_rate,lambda_w_k)
    print( "\n--------------------- Training Complete ---------------------\n")
    print('lambda(clinton) = ' + str(lambda_w_k[class_lookup['clinton']][word_lookup['<b>']]) )
    print('lambda(trump) = ' + str( lambda_w_k[ class_lookup['trump'] ][ word_lookup['<b>'] ] ) )
    print('lambda(clinton,country) = ' + str( lambda_w_k[ class_lookup['clinton']][ word_lookup['country'] ] ))
    print('lambda(trump,country) = ' + str(lambda_w_k[class_lookup['trump']][word_lookup['country']]))
    print('lambda(clinton,president) = ' + str(lambda_w_k[class_lookup['clinton']][word_lookup['president']]))
    print('lambda(trump,president) = ' + str(lambda_w_k[class_lookup['trump']][word_lookup['president']]))

    print('\n---------------------- Question 2c -------------------------\n')
    prob_list = apply_model(lambda_w_k, dev_set[0])
    for speaker in classes:
        print('p( ' + speaker + '| doc ) = ' + str(prob_list[class_lookup[speaker]]) )
    print ('sum of all probs = ' + str(sum(prob_list) ))
    print ('\n-----------------------Question 2d -------------------------\n')


    test_file(lambda_w_k, test_set)









