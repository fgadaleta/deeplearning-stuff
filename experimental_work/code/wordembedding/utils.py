from __future__ import with_statement
from collections import defaultdict
import os,sys,re,cPickle,random
import numpy as np
import nltk
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.datasets import imdb
#from keras.preprocessing import sequence
#from keras.models import Sequential
#from keras.layers.core import Dense, Dropout, Activation
#from keras.layers.embeddings import Embedding
#from keras.layers.recurrent import LSTM, GRU


def saveStuff(stuff, path=None):
    """
    Saves stuff to disk as pickle object
    :type stuff: any type
    :param stuff: data to be stored
    
    Return: create pickle file at path
    """
    if path == None:
        # TODO take name from something
        output = open('results/i-will-be-overwritten.pkl', 'wb')
    else:
        output = open(path, 'wb')

    # Pickle the list using the highest protocol available.
    cPickle.dump(stuff, output, -1)
    output.close()


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
        print('I/O error')
    except:
        print("Unexpected error" % sys.exc_info()[0])
        raise



def loadTextFiles(path):
    """
    Load all text files in path and returns list of content
    Cleans text of special characters 

    Usage: raw = loadTextFiles('/home/frag/Documents/python-code/word2vec/aclImdb/train/pos/') 
    """

    data = []
    
    for filename in os.listdir(path):
        f=open(path+filename, 'r')
        content = f.read()
        # clean special characters and append
        data.append(re.sub('\W+',' ', content))

    return data



def buildVocabulary(paragraphs, verbose=True):
    """ 
    Build vocabulary of unique words
    Returns: list of unique words
    """
    vocabulary = []
    
    for p in paragraphs:
        for word in p.split():
            vocabulary.append(word)

    vocabulary = set(vocabulary)
    if verbose:
        print('Built vocabulary of %d unique words'%len(vocabulary))
    
    return list(vocabulary)



def pars2idx(pars, vocabulary, verbose = True):
    """
    Convert paragraphs to word indexes of a vocabulary 

    :param pars: list of paragraphs 
    :type pars: list

    Return:
    list of word-indexed paragraphs
    """
    pars_idx = []
    npars = len(pars)

    for i in range(npars):
        p_idx = []
        
        for w in pars[i].split():
            w_idx = vocabulary.index(str(w))
            if type(w_idx) == int:
                p_idx.append(w_idx)
            
        if verbose:
            sys.stdout.write("Converting paragraph %d of %d \r" %(i, npars))
            sys.stdout.flush() 
        
        pars_idx.append(p_idx)

    return pars_idx


def idx2word(idx, voc):
    return voc[idx]

def pad_sequences(x, maxlen=100):
    nrows = len(x)
    ncols = maxlen
    res = np.zeros((nrows, ncols))
    
    for i,e in enumerate(x):
        nwords = len(e)
        if nwords > maxlen:
            res[i,:] = e[:maxlen]
            
        else:
            res[i, :len(e)] = e
            res[i, len(e):] = 0 
    
    return res


    
def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)


def save_model_parameters_theano(outfile, model):
    U, V, W = model.U.get_value(), model.V.get_value(), model.W.get_value()
    np.savez(outfile, U=U, V=V, W=W)
    print "Saved model parameters to %s." % outfile
   
def load_model_parameters_theano(path, model):
    npzfile = np.load(path)
    U, V, W = npzfile["U"], npzfile["V"], npzfile["W"]
    model.hidden_dim = U.shape[0]
    model.word_dim = U.shape[1]
    model.U.set_value(U)
    model.V.set_value(V)
    model.W.set_value(W)
    print "Loaded model parameters from %s. hidden_dim=%d word_dim=%d" % (path, U.shape[0], U.shape[1])
    

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
    #print("Insert text to start from [min 20 chars]:")
    #sentence = str(raw_input())
    #sentence = sentence[:maxlen]
    generated += sentence
    print('----- Generating with seed: "' + sentence + '"')
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
    print()    


def generateByWord(model, voc, maxlen=20, diversity=0.5, numwords=42):
    """ Generate text from model """

    text, sym_indices, indices_sym = voc
    syms = set(text)
    start_index = random.randint(0, len(text) - maxlen - 1)  
    generated = ''
    sentence = text[start_index: start_index + maxlen]
    
    #generated += sentence
    generated += ' '.join(sentence)
    print('----- Generating with seed: "' + ' '.join(sentence) + '"')
    sys.stdout.write(generated)

    for i in range(numwords):
        x = np.zeros((1, maxlen, len(syms)))
        for t, sym in enumerate(sentence):
            x[0, t, sym_indices[sym]] = 1.
            
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_sym = indices_sym[next_index]
        generated += ' '+next_sym
        sentence.append(next_sym)
        tmpsentence = sentence[1:]
        sentence = tmpsentence
        sys.stdout.write(next_sym+' ')
        sys.stdout.flush()
    print()    




def loadText(path, lower=False, decode=False, bychar=True):
    """
    Returns: (text, char_indices, indices_char) 
    """
    if lower:
        text = open(path).read().lower()
    else:
        text = open(path).read()
    
    if decode:
	text = text.decode('utf-8')

    if lower:
        text = text.lower()

    if bychar:
        print('corpus length:', len(text))
        chars = set(text)   # unique characters
	print chars	        
        print('Unique chars:', len(chars))
        char_indices = dict((c, i) for i, c in enumerate(chars))
        indices_char = dict((i, c) for i, c in enumerate(chars))
        voc = (text, char_indices, indices_char)    # return char-tokenized text        
        return voc

    else:
        tokens = nltk.word_tokenize(text)
        syms = set(tokens)    # unique symbols
        vocsize = len(syms)   # vocabulary size
        print('Unique words:', vocsize)
        word_freq = nltk.FreqDist(tokens)    
        vocab = word_freq.most_common(vocsize)
        # re-index by most frequent words 
        indices_word = [x[0] for x in vocab]
        unk_token = 'UNK'
        indices_word.append(unk_token)
        word_indices = dict([(w,i) for i,w in enumerate(indices_word)])   
        voc = (tokens, word_indices, indices_word)  # return word-tokenized text
        return voc



def buildTrainingSet(voc, bychar=True, maxlen=20, step=3):
    """
    create training dataset (x,y) -> (20-chars, next-char)
    cut the text in semi-redundant sequences of maxlen characters
    Return: (X,y)
    """
    
    text, sym_indices, _ = voc
    sentences = []
    next_syms = []
    
    syms = set(text)   # unique symbols (chars or words)
    
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_syms.append(text[i + maxlen])
    print('nb sequences:', len(sentences))
    
    print('Vectorization...')
    X = np.zeros((len(sentences), maxlen, len(syms)), dtype=np.bool)
    y = np.zeros((len(sentences), len(syms)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, sym in enumerate(sentence):
            X[i, t, sym_indices[sym]] = 1
        y[i, sym_indices[next_syms[i]]] = 1

    return (X,y)




def buildTrainingSequences(voc, maxlen=50, step=3):
    """
    create training dataset (x,y) -> (maxlen-sym-indices, next-sym)
    cut the text in semi-redundant sequences of maxlen characters
    
    Return: (sequence_of_int, bool)
    """
    
    text, sym_indices, _ = voc
    sentences = []
    next_syms = []
    
    syms = set(text)   # unique symbols (chars or words)
    
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_syms.append(text[i + maxlen])
    print('nb sequences:', len(sentences))
 
    X = np.zeros((len(sentences), maxlen), dtype=np.int)
    y = np.zeros((len(sentences), len(syms)), dtype=np.bool)

    for i, sentence in enumerate(sentences):
        for j, sym in enumerate(sentence):
            X[i,j] = sym_indices[sym] 
        
        y[i, sym_indices[next_syms[i]]] = 1  # one-hot enconding

    return (X,y)



def buildSkipgram(voc, maxlen=50, step=3):
    """
    Loop over words in a dataset, and for each word, we look 
    at a context window around the word. 
    Generate pairs of (pivot_word, other_word_from_same_context) with label 1,
    and pairs of (pivot_word, random_word) with label 0 (skip-gram method).
    Return: (x,y) -> ([word, context_words], 1) || ([word, out_of_context_words], 0)  
    """
    
    text, sym_indices, _ = voc
    sentences = []
    y = []
    syms = set(text)   # unique symbols (chars or words)

    # build correct sequences of words in context
    for i in range(maxlen, len(text) - maxlen, step):
        context = text[i-maxlen/2: i+maxlen/2]
        sentences.append(context)
        y.append(1)

    # build out of context sequences
    for i in range(maxlen, len(text) - maxlen, step):
        random_idx = np.random.random_integers(1, len(text)-1, maxlen)
        out_of_context = [text[x] for x in random_idx]
        sentences.append(out_of_context)
        y.append(0)

    print('nb sequences:', len(sentences))
 
    X = np.zeros((len(sentences), maxlen), dtype=np.int)

    for i, sentence in enumerate(sentences):
        for j, sym in enumerate(sentence):
            X[i,j] = sym_indices[sym] 
        
    y = np.asarray(y)

    # shuffle and return
    idx = np.random.permutation(X.shape[0])
    X = X[idx,:]
    y = y[idx]

    return (X,y)



def buildBrokenGram(voc, maxlen=50, step=3, breaks=10):
    """
    Loop over words in a dataset, and for each word, we look 
    at a context window around the word. 
    Generate pairs of (pivot_word, other_word_from_same_context) with label 1,
    and pairs of (pivot_word, random_word) with label 0 (skip-gram method).
    Return: (x,y) -> ([word, context_words], 1) || ([word, out_of_context_words], 0)  
    """
    
    text, sym_indices, _ = voc
    sentences = []
    y = []
    syms = set(text)   # unique symbols (chars or words)

    # build correct sequences of words in context
    for i in range(maxlen, len(text) - maxlen, step):
        context = text[i-maxlen/2: i+maxlen/2]
        sentences.append(context)
        y.append(1)

    # build out of context sequences
    for i in range(maxlen, len(text) - maxlen, step):
        numbreaks = np.random.random_integers(1, breaks, 1) # how many we break this time?
        
        random_idx = np.random.random_integers(1, len(text)-1, numbreaks)
        out_of_context = [text[x] for x in random_idx]
        sentences.append(context)
        y.append(0)

    print('nb sequences:', len(sentences))
 
    X = np.zeros((len(sentences), maxlen), dtype=np.int)

    for i, sentence in enumerate(sentences):
        for j, sym in enumerate(sentence):
            X[i,j] = sym_indices[sym] 
        
    y = np.asarray(y)
    return (X,y)
