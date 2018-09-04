'''
Example script to generate text char-by-char.

At least 60 epochs are required before the generated text
starts sounding coherent.

Min corpus size: ~100k characters. ~1M is better.

Usage:
$ THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python lstm_text_generation.py -i data/alice.txt -s 100 -g 256

'''

#!/usr/bin/env python
from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Embedding
from keras.layers.recurrent import LSTM
from keras.datasets.data_utils import get_file
import numpy as np
import random
import sys
import getopt
import utils as ut


def main(argv):
    # default values
    #path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
    path = ''
    outputfile = ''  
    startfrom = 1
    numsyms=100
    generateonly = False

    try:
        opts, args = getopt.getopt(argv,"hi:o:s:g:x",["ifile=","ofile="])
    except getopt.GetoptError:
        print('program.py -i <inputfile> -o <outputfile> -g <numchars> -s <startfrom>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('-'*50)
            print('\n')
            print('test.py -i <inputfile> -o <outputfile> -g <numchars> -s <startfrom>\n')
            print('-i <inputfile>\t text file to train the network')
            print('-g <numchars>\t number of characters to generate in generate-only mode')
            print('-s <startfrom>\t training epoch to start from (file must exist)')
            print('\n\n')
            print('-'*50)
            sys.exit()
        elif opt in ("-i", "--ifile"):
            path = arg
        elif opt in ("-g", "--generate"):
            generateonly = True
            numsyms = int(arg)
        #elif opt in ("-x", "--generateonly"):
        #    generateonly = True
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-s", "--startepo"):
            startfrom = int(arg)
        
    print('Input file is %s'          %path)
    print('Output file is %s'         %outputfile)
    print('Starting from epoch %s'    %startfrom)
    print('Generative model only %s ' %generateonly)
    print('Symbols to generate %d '%numsyms)

    maxlen = 20  # alice 
    
    # load data and build vocab
    voc = ut.loadText(path, bychar=True, lower=True)
    syms = set(voc[0])      # voc = (text, sym_indices, indices_sym) 
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

    if generateonly:
        print('Loading model from epoch %d' %(startfrom))
        #model.load_weights('results/lstm_word_based_epo_%d'%(startfrom))
        model.load_weights('results/lstm_char_based/tweet_lstm_char_based_epo_%d'%(startfrom))
        
        yn = 'y'   # default answer

        while(yn == 'y'):
            ut.generate(model, voc, numchars=numsyms)    
            #ut.generateByWord(model, voc, numwords=numsyms)    
            yn = ''
            while(yn not in ['y', 'n']):
                print("Generate more? [Y/n]: ")
                yn = str(raw_input()).lower()
    else:  # not generate-only mode 
    
        # train the model, output generated text after each iteration
        for iteration in range(startfrom, 400):
            print()
            print('-' * 50)
            print('Starting from epoch %d' %iteration)
            
            if iteration >= 2:
                print('-' * 50)
                print('Loading model from epoch %d' %(iteration-1))
                model.load_weights('results/lstm_char_based/tweet_lstm_char_based_epo_%d'%(iteration-1))
                
            model.fit(X, y, batch_size=128, nb_epoch=1)
            model.save_weights('results/lstm_char_based/tweet_lstm_char_based_epo_%d'%iteration, overwrite=True)
            # are we learning well? let's print some
            ut.generate(model, voc, numchars=42)    
            #ut.generateByWord(model, voc)    
        
if __name__ == "__main__":
   main(sys.argv[1:])








