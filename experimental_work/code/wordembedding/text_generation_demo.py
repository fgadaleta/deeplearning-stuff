from __future__ import print_function
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense
from keras.layers import Embedding
from keras.layers.recurrent import LSTM
from keras.datasets.data_utils import get_file

import numpy as np
import random
import sys
import getopt
import os.path

import utils as ut


path = 'data/alice.txt'

maxlen = 20 
    
# load data and build vocab
voc = ut.loadText(path, bychar=True)
syms = set(voc[0]) 
X,y = ut.buildTrainingSet(voc, maxlen=maxlen, step=3) 
            
# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
#model.add(Embedding(len(syms), 128))
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(syms))))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(syms)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

def sample(a, temperature=1.0):
        # helper function to sample an index from a probability array
        a = np.log(a) / temperature
        a = np.exp(a) / np.sum(np.exp(a))
        return np.argmax(np.random.multinomial(1, a, 1))

    
def generate(model, voc, maxlen=20, diversity=0.5, numchars=100):
    """ Generate text from model """

    text, char_indices, indices_char = voc
    chars = set(text)
    start_index = random.randint(0, len(text) - maxlen - 1)  
    generated = ''
    sentence = text[start_index: start_index + maxlen]
    generated += sentence
    
    print('----- Generating with seed: "' + sentence + '"')
    print(generated, end='')
    sys.stdout.write(generated)

    for i in range(numchars):
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.
            
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]
        generated += next_char
        sentence = sentence[1:] + next_char
        sys.stdout.write(next_char)
        sys.stdout.flush()
        #print(next_char, end='') 
        
    print('...')
    print()

age = 1
numsyms = 128

while(True):
    print('\n\n')
    print('=' * 30) 
    print('Me at the age of?')
    print('=' * 30) 
    
    age = str(raw_input()).lower()
    
    if age == 'stop':
        break
    
    filename = 'results/lstm_char_based/lstm_char_based_epo_%d'%(int(age))
    if os.path.isfile(filename):
        model.load_weights(filename)
        generate(model, voc, diversity=0.9, numchars=numsyms)
    else:
        print('I was never %s epochs old' %age) 
        print('Try another one.')

