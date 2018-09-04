import numpy as np
import theano as theano
import theano.tensor as T
import time
import operator
from utils import load_data, load_model_parameters_theano, generate_sentences, train_with_sgd
from gru_theano import *
import sys
import datetime


# Load data (this may take a few minutes)
VOCABULARY_SIZE = 500
X_train, y_train, word_to_index, index_to_word = load_data("data/reddit-comments-2015-08.csv", VOCABULARY_SIZE)

# Load parameters of pre-trained model
#model = load_model_parameters_theano('./data/pretrained.npz')

# Build your own model (not recommended unless you have a lot of time!)
LEARNING_RATE = 1e-3
NEPOCH = 20
HIDDEN_DIM = 128

print('building GRU model...')
model = GRUTheano(VOCABULARY_SIZE, HIDDEN_DIM)
print("done.\n")
sys.stdout.flush()

t1 = time.time()
model.sgd_step(X_train[0], y_train[0], LEARNING_RATE)
#model.sgd_step(X_train[:10], y_train[:10], LEARNING_RATE)
t2 = time.time()
print "SGD Step time: ~%f milliseconds" % ((t2 - t1) * 1000.)


def sgd_callback(model, num_examples_seen):
    #dt = datetime.now().isoformat()
    #loss = model.calculate_loss(x_train[:5000], y_train[:5000])
    print("\n(%d)" % (num_examples_seen))
    print("--------------------------------------------------")
    #print("Loss: %f" % loss)
    #generate_sentences(model, 10, index_to_word, word_to_index)
    #ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
    outfile = "GRU-%s.dat" % (str(num_examples_seen))
    #ut.save_model_parameters_theano(model, MODEL_OUTPUT_FILE)
    
    save_model_parameters_theano(model, outfile)
    print("\n")
    sys.stdout.flush()


#train_with_sgd(model, X_train, y_train, LEARNING_RATE, NEPOCH, decay=0.9, callback=sgd_callback, callback_every=1000)

#train_batch(model, X_train, y_train, LEARNING_RATE, NEPOCH, decay=0.9)

#generate_sentences(model, 10, index_to_word, word_to_index)



