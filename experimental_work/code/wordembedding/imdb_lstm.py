from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.datasets import imdb

'''
    Train a LSTM on the IMDB sentiment classification task.
    The dataset is actually too small for LSTM to be of any advantage
    compared to simpler, much faster methods such as TF-IDF+LogReg.
    Notes:
    - RNNs are tricky. Choice of batch size is important,
    choice of loss and optimizer is critical, etc.
    Some configurations won't converge.
    - LSTM loss decrease patterns during training can be quite different
    from what you see with CNNs/MLPs/etc.
    GPU command:
     THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_lstm.py
'''
import sys
import cPickle

def loadStuff(path=None):
    """ 
    Loads stuff from disk as pickle object to memory
    :type stuff: any type
    :param stuff: data to be loaded
    
    Return: object loaded (same type as original object) 
    """

    if path == None:
        print("No path specified")
        return 
      
    try:
        pkl_file = open(path, 'rb')
        obj = cPickle.load(pkl_file)
        pkl_file.close()
        print('Data correctly loaded and returned')
        return obj

    except IOError as e:
        #print "I/O error({0}):{1}".format(e.errno, e.strerror)
        print('error')
    except:
        print("Unexpected error", sys.exc_info()[0])
        raise


def saveStuff(stuff, path=None):
    """
    Saves stuff to disk as pickle object
    :type stuff: any type
    :param stuff: data to be stored
    
    Return: create pickle file at path
    """ 
    if path == None:
        # TODO take name from something
        output = open('i-will-be-overwritten.pkl', 'wb')
    else:
        output = open(path, 'wb')
    
    # Pickle the list using the highest protocol available.
    cPickle.dump(stuff, output, -1)
    output.close()




max_features = 20000
maxlen = 100  # cut texts after this number of words 
              #(among top max_features most common words)
batch_size = 32

print("Loading data...")
test_split = 0.2
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features, test_split=0.2)
#X, labels = loadStuff('imdb.pkl')
#X_train = X[:int(len(X)*(1-test_split))]
#y_train = labels[:int(len(X)*(1-test_split))]
#X_test = X[int(len(X)*(1-test_split)):]
#y_test = labels[int(len(X)*(1-test_split)):]

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 50, input_length=maxlen))
model.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'))  # try using a GRU instead, for fun
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', optimizer='rmsprop')

print("Train...")
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=4, validation_data=(X_test, y_test), show_accuracy=True)

saveStuff(model, './model.pkl')

score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, show_accuracy=True)
print('Test score:', score)
print('Test accuracy:', acc)


