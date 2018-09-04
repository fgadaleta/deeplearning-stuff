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
from gru_theano import GRUTheano


def train_with_sgd(model, X_train, y_train, learning_rate=0.001, nepoch=20, decay=0.9,
                   callback_every=10000, callback=None):
    num_examples_seen = 0
    for epoch in range(nepoch):
        # For each training example...
        for i in np.random.permutation(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate, decay)
            num_examples_seen += 1
            # Optionally do callback
            if (callback and callback_every and num_examples_seen % callback_every == 0):
                callback(model, num_examples_seen)            
    return model


# We do this every few examples to understand what's going on
def sgd_callback(model, num_examples_seen):
    dt = datetime.now().isoformat()
    loss = model.calculate_loss(x_train[:5000], y_train[:5000])
    print("\n%s (%d)" % (dt, num_examples_seen))
    print("--------------------------------------------------")
    print("Loss: %f" % loss)
    #generate_sentences(model, 10, index_to_word, word_to_index)
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
    MODEL_OUTPUT_FILE = "GRU-%s-%s-%s-%s.dat" % (ts, vocabulary_size, _HIDDEN_DIM, HIDDEN_DIM)
    ut.save_model_parameters_theano(model, MODEL_OUTPUT_FILE)
    print("\n")
    sys.stdout.flush()


vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

# model RNN here
_HIDDEN_DIM = 80
_LEARNING_RATE = 0.005
_NEPOCH = 20
_MODEL_FILE = None

 
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


model = GRUTheano(vocabulary_size, hidden_dim=_HIDDEN_DIM, bptt_truncate=-1)

t1 = time.time()
model.sgd_step(X_train[10], y_train[10], _LEARNING_RATE)
t2 = time.time()
print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)

if _MODEL_FILE != None:
    ut.load_model_parameters_theano(_MODEL_FILE, model)

for epoch in range(_NEPOCH):
    train_with_sgd(model, X_train, y_train, nepoch=1, learning_rate=_LEARNING_RATE, decay=0.9, 
    callback_every=PRINT_EVERY, callback=sgd_callback)




##################################################################
##    Generating stuff from the model
##################################################################
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

 
# load and generate
generate = False

if generate == True:
    model = RNNTheano(vocabulary_size, hidden_dim=80)
    ut.load_model_parameters_theano('./data/rnn-lstm-theano-80-8000-2015-11-18-06-01-13.npz', model)

    num_sentences = 10
    senten_min_length = 7
    senten_max_length = 256

    for i in range(num_sentences):
        sent = []
        # We want long sentences, not sentences with one or two words
        while len(sent) < senten_min_length:
            sent = generate_sentence(model)
        print " ".join(sent)
