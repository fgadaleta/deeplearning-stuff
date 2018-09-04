# from wildml.com

import random
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Dropout, Activation
import numpy as np
import utils as ut
import nltk
import itertools
import time
import sys
from datetime import datetime
from rnn_theano import RNNTheano

def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=1, evaluate_loss_after=5, verbose = True):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):

        if verbose:
            print("Epoch %d " %epoch)

        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5  
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
            # ADDED! Saving model parameters
            ut.save_model_parameters_theano("./data/rnn-theano-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, time), model)
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1


vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"
 
ready = True
if ready:  # just load stuff 
    pars_idx = ut.loadStuff('./data/pars_idx.pkl')
    pars = ut.loadStuff('./data/pars.pkl')
    voc = ut.loadStuff('./data/voc.pkl')
    X,y = ut.loadStuff('./data/dataset.pkl')  


# Split full comments into sentences
sentences = []
for x in pars:
    par = nltk.sent_tokenize(x.lower())
    # Append SENTENCE_START and SENTENCE_END
    sentences.append("%s %s %s" % (sentence_start_token, par[0], sentence_end_token))

print "Parsed %d sentences." % (len(sentences))
    
# Tokenize the sentences into words
# TODO remove html if any
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
 
# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print "Found %d unique words tokens." % len(word_freq.items())
 
# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
 
print "Using vocabulary size %d." % vocabulary_size
print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])
 
# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
 
print "\nExample sentence: '%s'" % sentences[0]
print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]

# Create the training data
print("Creating training data")
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])


# model RNN here
_HIDDEN_DIM = 80
_LEARNING_RATE = 0.005
_NEPOCH = 100
_MODEL_FILE = None

model = RNNTheano(vocabulary_size, hidden_dim=_HIDDEN_DIM)
t1 = time.time()
model.sgd_step(X_train[10], y_train[10], _LEARNING_RATE)
t2 = time.time()
print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)

if _MODEL_FILE != None:
    ut.load_model_parameters_theano(_MODEL_FILE, model)

train_with_sgd(model, X_train, y_train, nepoch=_NEPOCH, learning_rate=_LEARNING_RATE)




# load and generate
model = RNNTheano(vocabulary_size, hidden_dim=80)
ut.load_model_parameters_theano('./data/rnn-theano-80-8000-2015-11-18-06-01-13.npz', model)

def generate_sentence(model):
    # We start the sentence with the start token
    new_sentence = [word_to_index[sentence_start_token]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        next_word_probs = model.forward_propagation(new_sentence)
        sampled_word = word_to_index[unknown_token]
        # We don't want to sample unknown words
        while sampled_word == word_to_index[unknown_token]:
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str


def generate_text(model):
    #stop = False
    # We start the sentence with the start token
    new_sentence = [word_to_index[sentence_start_token]]
    
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
       	next_word_probs = model.forward_propagation(new_sentence)
       	sampled_word = word_to_index[unknown_token]
       	# We don't want to sample unknown words
       	while sampled_word == word_to_index[unknown_token]:
       		samples = np.random.multinomial(1, next_word_probs[-1])
       		sampled_word = np.argmax(samples)
       	# first known word generated
       	new_sentence.append(sampled_word)
       	print('generated %d words\n' %len(new_sentence))
    
    new_sentence.append(word_to_index[sentence_end_token])    
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str

 
num_sentences = 10
senten_min_length = 7
senten_max_length = 256

for i in range(num_sentences):
    sent = []
    # We want long sentences, not sentences with one or two words
    while len(sent) < senten_min_length:
        sent = generate_sentence(model)
    print " ".join(sent)
