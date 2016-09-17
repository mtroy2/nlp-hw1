from scipy.misc import logsumexp
import os
import math
from autograd import grad
import autograd.numpy as np
train_set = open( os.getcwd() + '/hw1-data/train' )
symbols = set(['?','!',';',',','.'])
common_words = set( ['we', 'the','i','a','an','and','are','as','at','be','by','for','from','has','he','in','is','it','its','of','on','that','to','was','were','will','with'])

speaker_words   = {}  # speaker -> [word -> count]
class_instances = {}  # speaker -> instances of that class
class_words     = {}  # speaker -> all words in class
word_probs 	     = {}  # speaker -> [word -> p( w | k )]
class_probs	     = {}  # class 	-> p(k)
log_p_k_d       = {}
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

    # Remove common words        
    words = [x for x in words if x not in common_words]
    class_words[speaker].extend(words[1:])

    for word in words[1:]:

        # if word has been seen already
        if word in speaker_words[speaker]:
            speaker_words[speaker][word] += 1.
            # add word to unique dict
        else:
            speaker_words[speaker][word] = 1.
			
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

#p( w | k) w/ add 1 smoothing
for speaker, word_dict in speaker_words.items():
    speaker_words[speaker]['unk'] = 0.
    for word, count in word_dict.items():
        word_probs[speaker][word] = ( count + .25 ) / (len( class_words[speaker]) + len(vocabulary)*.25 + 1  )
print ('Question 1: ----------------------------------------')
print ('c( Clinton ) = '  + str(class_instances['clinton']))
print ('c( Trump   ) = '  + str(class_instances['trump']))
		
print ('c( Clinton, country ) = ' + str(speaker_words['clinton']['country']))
print ('c( Trump  , country ) = ' + str(speaker_words[ 'trump' ]['country']))

print ('End Question 1: -----------------------------------')

print ('\nQuestion 2: -------------------------------------')
print ('p( Clinton ) = ' + str(class_probs['clinton']))
print ('p( Trump ) = ' + str(class_probs['trump']))
print ('p( country | Clinton ) = '   + str(word_probs['clinton']['country']))
print ('p( country | Trump   ) = '   + str(word_probs[ 'trump' ]['country']))
print ('p( president | Clinton ) = ' + str(word_probs['clinton']['president']))
print ('p( president | Trump   ) = ' + str(word_probs[ 'trump' ]['president']))
print ('End Question 2: ------------------------------------')

print ('\nBegin Question 3: --------------------------------')

dev_file = open( os.getcwd() + '/hw1-data/dev' )
content = dev_file.readlines()

line = content[0]
# remove symbols from line
for char in line:
    if char in symbols:
        line = line.replace(char, '')

line = line.split()
line = [word for word in line if word not in common_words]



# compute log p(k,d)'s
for speaker, word_dict in word_probs.items():

    prob_k_d = 0.
    # get summation of log p(w|k)
    for word in line[1:]:
        if word in word_dict:
            prob_k_d += math.log(word_dict[word])
        else:
            prob_k_d += math.log(word_dict['unk'])
    
    
    prob_k_d += math.log(class_probs[speaker])  
    log_p_k_d[speaker] = prob_k_d
	
p_k_give_d = 0

p_sum = 0
for speaker, log_prob in log_p_k_d.items():
      
    p_k_give_d = math.exp(log_prob - logsumexp(list(log_p_k_d.values())))
    p_sum += p_k_give_d
    print ('p( ' + speaker + ' | d ) = ' + str(p_k_give_d))
print ('total probability = ' + str(p_sum))
	
print ('End Question 3: -----------------------------------')

print ('\nBegin Question 4 ----------------------------------')

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
    for speaker, word_dict in word_probs.items():

        prob_k_d = 0.
        # get summation of log p(w|k)
        for word in line[1:]:
            if word in word_dict:
                prob_k_d += math.log(word_dict[word])
            else:
                prob_k_d += math.log(word_dict['unk'])
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

print ('Guess percentage = ' + str(num_correct / (num_correct+num_incorrect)))

print ('End Question 4 ------------------------------------')





















