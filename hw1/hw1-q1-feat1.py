from scipy.misc import logsumexp
import os
import math
from autograd import grad
import autograd.numpy as np
from collections import namedtuple
from nltk import bigrams
train_set = open( os.getcwd() + '/hw1-data/train' )
symbols = set(['?','!',';',',','.'])
common_words = set( ['we','the','i','a','an','and','are','as','at','be','by','for','from','has','he','in','is','it','its','of','on','that','to','was','were','will','with'])

speaker_words   = {}  # speaker -> [word -> count]
class_instances = {}  # speaker -> instances of that class
class_words     = {}  # speaker -> all words in class
word_probs 	     = {}  # speaker -> [word -> p( w | k )]
class_probs	     = {}  # class 	-> p(k)
log_p_k_d       = {}
speaker_bigrams = {}  # speaker -> [bigram -> count]
bigram_vocab = []
bigram_probs = {}
bigram_list = {}

for line in train_set:

    # Remove symbols
    for char in line:
        if char in symbols:
            line = line.replace(char, '')
    words = line.split()
    speaker = words[0]	

    if speaker in class_instances:
        class_instances[speaker] += 1.	#increment class instance

    else:
        class_instances[speaker] = 1.	# init num classes
        speaker_words[speaker]   = {}  	# init words dict
        class_words[speaker]     = []	# init word list
        class_probs[speaker]     = 0.	# init prob to 0
        word_probs[speaker] 	 = {}
        speaker_bigrams[speaker] = {}
        bigram_probs[speaker]	= {}
        bigram_list[speaker ]   = []
    # Remove common words        
    words = [x for x in words if x not in common_words]
    class_words[speaker].extend(words[1:])

    # empty line
    if len(words) < 2:
        continue
    s_symbol_line = words[1:]
    s_symbol_line.insert(0,'@')
    s_symbol_line.extend('@')
    bigramed_line = list(bigrams(s_symbol_line))
    for word in words[1:]:
        # if word has been seen already
        if word in speaker_words[speaker]:
            speaker_words[speaker][word] += 1.
            # add word to unique dict
        else:
            speaker_words[speaker][word] = 1.
    bigram_list[speaker].extend(bigramed_line)
    for bg in bigramed_line:
        if bg not in bigram_vocab:
            bigram_vocab.append(bg)
        if bg in speaker_bigrams[speaker]:
            speaker_bigrams[speaker][bg] += 1
        else:
            speaker_bigrams[speaker][bg] = 1
		
# Compute Probabilities
total_classes = sum(class_instances.values())
total_p = 0.
# p(k)
for speaker, instances in class_instances.items():
    class_probs[speaker] = instances / total_classes

#get cumulative vocabulary
transcript = []

for speaker, lines in class_words.items():
    transcript.extend(lines)
vocabulary = set(transcript)


#p( w | k) w/ add .25 smoothing
for speaker, word_dict in speaker_words.items():
    speaker_words[speaker]['unk'] = 0.
    for word, count in word_dict.items():
        word_probs[speaker][word] = ( count + .25 ) / (len( class_words[speaker]) + len(vocabulary)*.25 + 1  )

for speaker,bigram_dict in speaker_bigrams.items():
    speaker_bigrams[speaker][('unk','unk')] = 0.
    for bg, count in bigram_dict.items():
        bigram_probs[speaker][bg] = ( count + .25) / (len( bigram_list[speaker] ) + len(bigram_vocab)*.25 + 1 )


dev_file = open( os.getcwd() + '/hw1-data/test' )
num_correct = 0.
num_incorrect = 0.
for line in dev_file:
    # remove symbols from line
    for char in line:
        if char in symbols:
            line = line.replace(char, '')

    line = line.split()
    line = [word for word in line if word not in common_words]

    correct_class=line[0]
    p_sum = 0
    max_prob = 0
    # compute log p(k,d)'s
    s_symbol_line = line[1:]
    s_symbol_line.insert(0,'@')
    s_symbol_line.extend('@')
    bigramed_line = list(bigrams(s_symbol_line))

    for speaker, word_dict in word_probs.items():

        prob_k_d = 0.
        # get summation of log p(w|k)
        for word in line[1:]:

            if word in word_dict:
                prob_k_d += math.log(word_dict[word])
            else:
                prob_k_d += math.log(word_dict['unk'])
	
        for bg in bigramed_line:             
            if bg in speaker_bigrams[speaker]:
                prob_k_d += math.log(bigram_probs[speaker][bg])
            else:
                prob_k_d += math.log(bigram_probs[speaker][('unk','unk')])
        prob_k_d += math.log(class_probs[speaker])  
        log_p_k_d[speaker] = prob_k_d
    p_k_give_d = 0
    p_sum = 0
    for speaker, log_prob in log_p_k_d.items():  
        p_k_give_d = math.exp(log_prob - logsumexp(list(log_p_k_d.values())))
        p_sum+=p_k_give_d
        if p_k_give_d > max_prob:
            max_prob = p_k_give_d
            class_guess = speaker
    if class_guess == correct_class:
        num_correct += 1.
    else:
        num_incorrect += 1.

print ('final p_sum = '+ str(p_sum))
print ('Guess percentage = ' + str(num_correct / (num_correct+num_incorrect)))

print ('End Question 4 ------------------------------------')

