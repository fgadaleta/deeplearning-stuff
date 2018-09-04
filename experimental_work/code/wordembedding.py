from __future__ import with_statement
from collections import defaultdict
from collections import OrderedDict

import copy
import cPickle
import gzip
import os
import urllib
import random
import stat
import subprocess
import sys
import timeit

import numpy as np

import theano
from theano import tensor as T

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN, SimpleDeepRNN
from keras.datasets import imdb
from keras.callbacks import EarlyStopping

# Otherwise the deepcopy fails
import sys
sys.setrecursionlimit(1500)

PREFIX = '../data'

# utils functions
def shuffle(lol, seed):
    '''
    lol :: list of list as input
    seed :: seed the shuffling

    shuffle inplace each list in the same order
    '''
    for l in lol:
        random.seed(seed)
        random.shuffle(l)


# start-snippet-1
def contextwin(l, win):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence

    l :: array containing the word indexes

    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >= 1
    l = list(l)

    lpadded = win // 2 * [-1] + l + win // 2 * [-1]
    out = [lpadded[i:(i + win)] for i in range(len(l))]

    assert len(out) == len(l)
    return out


# data loading functions
def atisfold(fold):
    assert fold in range(5)
    filename = os.path.join(PREFIX, 'atis.fold'+str(fold)+'.pkl.gz')
    f = gzip.open(filename, 'rb')
    train_set, valid_set, test_set, dicts = cPickle.load(f)
    return train_set, valid_set, test_set, dicts


# metrics function using conlleval.pl
def conlleval(p, g, w, filename, script_path):
    '''
    INPUT:
    p :: predictions
    g :: groundtruth
    w :: corresponding words

    OUTPUT:
    filename :: name of the file where the predictions
    are written. it will be the input of conlleval.pl script
    for computing the performance in terms of precision
    recall and f1 score

    OTHER:
    script_path :: path to the directory containing the
    conlleval.pl script
    '''
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, w in zip(sl, sp, sw):
            out += w + ' ' + wl + ' ' + wp + '\n'
        out += 'EOS O O\n\n'

    f = open(filename, 'w')
    f.writelines(out)
    f.close()

    return get_perf(filename, script_path)


def download(origin, destination):
    '''
    download the corresponding atis file
    from http://www-etud.iro.umontreal.ca/~mesnilgr/atis/
    '''
    print 'Downloading data from %s' % origin
    urllib.urlretrieve(origin, destination)


def get_perf(filename, folder):
    ''' run conlleval.pl perl script to obtain
    precision/recall and F1 score '''
    _conlleval = os.path.join(folder, 'conlleval.pl')
    if not os.path.isfile(_conlleval):
        url = 'http://www-etud.iro.umontreal.ca/~mesnilgr/atis/conlleval.pl'
        download(url, _conlleval)
        os.chmod(_conlleval, stat.S_IRWXU)  # give the execute permissions

    proc = subprocess.Popen(["perl",
                            _conlleval],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)

    stdout, _ = proc.communicate(''.join(open(filename).readlines()))
    for line in stdout.split('\n'):
        if 'accuracy' in line:
            out = line.split()
            break

    precision = float(out[6][:-2])
    recall = float(out[8][:-2])
    f1score = float(out[10])

    return {'p': precision, 'r': recall, 'f1': f1score}


def saveStuff(stuff, filename):
    np.save(filename, stuff)

def loadStuff(stuff):
    res = np.load(stuff)
    return res
    
#def loadEmbeddings(folder):
#    f = os.path.join(folder, 'embeddings.npy')
#    res = np.load(f)   
#    return res 
    

def main(param=None):
    if not param:
        param = {
            'lr': 0.0970806646812754,
            'win': 9,
            # number of words in the context window
            'nhidden': 200,
            # number of hidden units
            'emb_dimension': 50,
            # dimension of word embedding
            'nepochs': 60,
            # 60 is recommended
            'savemodel': True}
    print param

    # load the dataset
    train_set, valid_set, test_set, dic = atisfold(3)
    
    idx2label = dict((k, v) for v, k in dic['labels2idx'].iteritems())
    idx2word = dict((k, v) for v, k in dic['words2idx'].iteritems())
    
    train_lex, train_ne, train_y = train_set
    valid_lex, valid_ne, valid_y = valid_set
    test_lex, test_ne, test_y = test_set

    vocsize = len(set(reduce(lambda x, y: list(x) + list(y),
                             train_lex + valid_lex + test_lex)))

    nclasses = len(set(reduce(lambda x, y: list(x)+list(y),
                              train_y + test_y + valid_y)))
    nsentences = len(train_lex)
    
    vocwords = []
    for i in range(vocsize):
        vocwords.append(idx2word[i])
        
    saveStuff(vocwords, 'rnnslu/words')
    print('Vocabulary size %d' %vocsize)
    print('Number of labels %d' %nclasses)
    print('Number of sentences %d' %nsentences)
    
    windowsize = 9
    max_features = vocsize 
    maxlen = windowsize  # context window size
    batch_size = 16

    train_data_x = []
    train_data_y = []
    
    # transform data (context window)
    for i in range(len(train_lex)):
        tmp = contextwin(train_lex[i], windowsize)
        train_data_x.append(tmp)
        train_data_y.append(train_y[i])
        
    X  = [val for sublist in train_data_x for val in sublist]
    y  = [val for sublist in train_data_y for val in sublist]
    
    # prepare data split
    test_split = 0.2
    X_train = X[:int(len(X)*(1-test_split))]
    y_train = y[:int(len(X)*(1-test_split))]  # keras is happy with list here
    X_test = X[int(len(X)*(1-test_split)):]
    y_test = y[int(len(X)*(1-test_split)):]
    X_train = np.asarray(X_train)  # keras wants arrays here
    X_test = np.asarray(X_test)
    
    # keras implementation of RNN
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    
    print('Build model...')
    model = Sequential()
    model.add(Embedding(max_features, 50, input_length=windowsize))
    #model.add(LSTM(output_dim=200, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(SimpleRNN(output_dim=200, activation='softmax'))  
    #model.add(SimpleDeepRNN(output_dim=200, depth=3, activation='sigmoid')) 
    # try using a GRU instead, for fun
    #model.add(GRU(output_dim=200, activation='sigmoid'))  
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('softmax'))

    # try using different optimizers and different optimizer configs
    #sgd = SGD(lr=0.0970806646812754, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')

    print("Train...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    model.fit(X_train, y_train, 
              batch_size=batch_size, 
              nb_epoch=30, 
              validation_data=(X_test, y_test), 
              show_accuracy=True,
              callbacks=[early_stopping])

    print('Extracting embeddings')
    emb = model.layers[0]
    embeddings = emb.W.get_value()
    print('Saving embeddings and vocabulary for processing with t-SNE')
    saveStuff(embeddings, '/home/frag/Documents/python-code/tsne/data/imdb_embeddings.pkl')
    saveStuff(idx2word.values(), '/home/frag/Documents/python-code/tsne/data/imdb_voc.pkl')


if __name__ == '__main__':
    main()
