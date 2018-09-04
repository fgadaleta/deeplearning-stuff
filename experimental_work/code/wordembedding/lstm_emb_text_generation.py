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
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense
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

    maxlen = 5 
    
    # load data and build vocab
    voc = ut.loadText(path, bychar=False, lower=True)
    syms = set(voc[0])      # voc = (text, sym_indices, indices_sym) 
    
    X,y = ut.buildSkipgram(voc, maxlen=20, step=5)
    #X,y = ut.buildTrainingSequences(voc, maxlen=maxlen, step=1)
    print('input shape (%s, %s)'%(X.shape, y.shape))
    
    vocsize = len(syms)
    emb_dims = 128

    # build the model: 2 stacked LSTM
    print('Build model...')
    model = Sequential()
    layers = [maxlen, 100, 512,1]           # (X,y) from buildSkipgram
    #layers = [maxlen, 100, 512, vocsize]   # (X,y) from buildSequences

    model.add(Embedding(vocsize, emb_dims))
    
    #model.add(LSTM(input_dim=layers[0], output_dim=layers[1], return_sequences=True))  # with next lstm layer
    model.add(LSTM(input_dim=layers[0], output_dim=layers[1], return_sequences=False))  # with no more layers
    model.add(Dropout(0.5))
    
    #model.add(LSTM(layers[2], return_sequences=False))
    #model.add(Dropout(0.2))

    model.add(Dense(output_dim = layers[3]))  # for skipgram
    model.add(Activation('sigmoid'))          # for skipgram
    #model.add(Activation('softmax'))         # for sequences
    

    #model.compile(loss='categorical_crossentropy', optimizer='rmsprop')  # buildSequences
    model.compile(loss='mse', optimizer='rmsprop')        # buildSkipgram

   
    if generateonly:
        print('Loading model from epoch %d' %(startfrom))
        model.load_weights('results/lstm_word_based_epo_%d'%(startfrom))
        
        yn = 'y'   # default answer

        while(yn == 'y'):
            #ut.generate(model, voc, numchars=numsyms)    
            ut.generateByWord(model, voc, numwords=numsyms)    
            yn = ''
            while(yn not in ['y', 'n']):
                print("Generate more? [Y/n]: ")
                yn = str(raw_input()).lower()
    else:  # not generate-only mode 
    
        # train the model, output generated text after each iteration
        for iteration in range(startfrom, 400, 10):
            print()
            print('-' * 50)
            print('Starting from epoch %d' %iteration)
            
            if iteration >= 2:
                print('-' * 50)
                print('Loading model from epoch %d' %(iteration-1))
                model.load_weights('results/lstm_emb_word_based_epo_%d'%(iteration-1))
                
            model.fit(X, y, batch_size=128, nb_epoch=10)
            model.save_weights('results/lstm_emb_word_based_epo_%d'% (10+iteration-1), overwrite=True)
            print('Extracting embeddings')
            emb = model.layers[0]
            embeddings = emb.W.get_value()
            print('embeddings shape', embeddings.shape)
            print('Saving embeddings and vocabulary for t-SNE')
            np.save('results/lstm_embeddings', embeddings)
            #np.save('results/vocab_embeddings', voc[1])
            ut.saveStuff(voc[1], 'results/vocab_embeddings')
            # are we learning well? let's print some
            #ut.generate(model, voc, numchars=42)    
            #ut.generateByWord(model, voc)    
        
if __name__ == "__main__":
   main(sys.argv[1:])
