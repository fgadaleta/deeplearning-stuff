#! /usr/bin/env python

import sys
import os
import time
import numpy as np
from utils import *
from datetime import datetime
from gru_theano import GRUTheano

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
VOCABULARY_SIZE = int(os.environ.get("VOCABULARY_SIZE", "3700"))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "48"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "128"))
NEPOCH = int(os.environ.get("NEPOCH", "80"))
MODEL_OUTPUT_FILE = os.environ.get("MODEL_OUTPUT_FILE")
INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "./data/reddit-comments-2015.csv")
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "3000"))

if not MODEL_OUTPUT_FILE:
  ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
  MODEL_OUTPUT_FILE = "GRU-%s-%s-%s-%s.dat" % (ts, VOCABULARY_SIZE, EMBEDDING_DIM, HIDDEN_DIM)

# Load data
#x_train, y_train, word_to_index, index_to_word = load_data(INPUT_DATA_FILE, VOCABULARY_SIZE)


xx, yy, vocab, word_to_index, index_to_word = loadText('pg11.txt', \
                                                       origin="http://www.gutenberg.org/cache/epub/11/pg11.txt",\
                                                       vocsize=VOCABULARY_SIZE)

# need to break x_train 
x_train = []
y_train = []
maxlen = 25
for i in range(xx.shape[0]-maxlen):
  x_train.append(xx[i:i+maxlen])
  y_train.append(yy[i:i+maxlen])

print(len(x_train), ' training samples')

# Build model
#model = GRUTheano(VOCABULARY_SIZE, hidden_dim=HIDDEN_DIM, bptt_truncate=-1)


# We do this every few examples to understand what's going on
def sgd_callback(model, epoch, num_examples_seen):
  #dt = datetime.now().isoformat()
  ts = datetime.now().strftime("%Y-%m-%d-%H-%M")  
  loss = model.calculate_loss(x_train[:10], y_train[:10])
  print("\n%s (%d)" % (ts, num_examples_seen))
  print("Loss: %f" % loss)
  print("--------------------------------------------------")
  #generate_sentences(model, 5, index_to_word, word_to_index)
  
  print('Saving model')
  MODEL_OUTPUT_FILE = "GRU-epo%s-voc%s-hid%s.dat" % (epoch,VOCABULARY_SIZE,HIDDEN_DIM)
  save_model_parameters_theano(model, MODEL_OUTPUT_FILE)
  
  # embedding layer
  print('Saving word embedding')
  emb = model.E.get_value().transpose()
  saveStuff(emb, 'results/alice_embeddings_epo%s.pkl'%(epoch))
  saveStuff(vocab, 'results/alice_voc.pkl')

  print("\n")
  sys.stdout.flush()


start_epoch = 41
generateOnly = True

if generateOnly == True:
  path = 'GRU-epo41-voc3700-hid128.dat.npz'
  model = load_model_parameters_theano(path, modelClass=GRUTheano)
  generate_sentences(model, 5, index_to_word, word_to_index)
  exit


for epoch in range(start_epoch, NEPOCH):
  
  if epoch>0:
    path = 'results/GRU-epo%s-voc%s-hid%s.dat.npz' %(epoch, VOCABULARY_SIZE, HIDDEN_DIM)
    print('Loading file %s' %path)
    print('Training  %s of %s epochs' %(epoch,NEPOCH))
    model = load_model_parameters_theano(path, modelClass=GRUTheano)
  else:
    # Build model from scratch
    model = GRUTheano(VOCABULARY_SIZE, hidden_dim=HIDDEN_DIM, bptt_truncate=-1)

  
  train_with_sgd(model, x_train, y_train, \
                 learning_rate=LEARNING_RATE, \
				 nepoch=80, \
                 startfrom=epoch, \
                 decay=0.9, \
                 callback_every=PRINT_EVERY, \
                 callback=sgd_callback)



"""
# load vocabulary 
xx, yy, vocab, word_to_index, index_to_word = loadText('pg11.txt', \
                                                       origin="http://www.gutenberg.org/cache/epub/11/pg11.txt",\
                                                       vocsize=VOCABULARY_SIZE)


path = 'GRU-2016-02-21-16-28-3700-48-128.dat.npz'
loaded_model = load_model_parameters_theano(path, modelClass=GRUTheano)

# embedding layer
emb = loaded_model.E.get_value().transpose()
saveStuff(emb, 'alice_embeddings.pkl')
saveStuff(vocab, 'alice_voc.pkl')

"""
