
'''
    We loop over words in a dataset, and for each word, we look at a context window around the word. 
    We generate pairs of (pivot_word, other_word_from_same_context) with label 1,
    and pairs of (pivot_word, random_word) with label 0 (skip-gram method).

    We use the layer WordContextProduct to learn embeddings for the word couples,
    and compute a proximity score between the embeddings (= p(context|word)),
    trained with our positive and negative labels.

    We then use the weights computed by WordContextProduct to encode words 
    and demonstrate that the geometry of the embedding space 
    captures certain useful semantic properties.

    Read more about skip-gram in this particularly gnomic paper by Mikolov et al.: 
        http://arxiv.org/pdf/1301.3781v3.pdf

    Note: you should run this on GPU, otherwise training will be quite slow. 
    On a EC2 GPU instance, expect 3 hours per 10e6 comments (~10e8 words) per epoch with dim_proj=256.
    Should be much faster on a modern GPU.

    GPU command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python skipgram_word_embeddings.py

    Dataset: 5,845,908 Hacker News comments. 
    Obtain the dataset at: 
        https://mega.co.nz/#F!YohlwD7R!wec0yNO86SeaNGIYQBOR0A 
        (HNCommentsAll.1perline.json.bz2)
'''
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import theano
import six.moves.cPickle
import os, re, json

from keras.preprocessing import sequence, text
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils, generic_utils
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense
from keras.layers import Embedding
from keras.layers.recurrent import LSTM

#from keras.layers.embeddings import Embedding #,WordContextProduct
from six.moves import range
from six.moves import zip

import utils as ut

max_features = 50000 # vocabulary size: top 50,000 most common words in data
vocsize = 50000
emb_dims = 256
maxlen = 10

skip_top = 100 # ignore top 100 most common words
nb_epoch = 1
dim_proj = 256 # embedding space dimension

save = True
test_model = True
train_model = False 
load_model = True
load_tokenizer = True 

 
save_dir = os.path.expanduser("results/")
model_load_fname = "HN_skipgram_model.pkl"
model_save_fname = "HN_skipgram_model.pkl"
tokenizer_fname = "HN_tokenizer.pkl"

data_path = 'data/HNCommentsAll.1perline.json'

# text preprocessing utils
html_tags = re.compile(r'<.*?>')
to_replace = [('&#x27;', "'")]
hex_tags = re.compile(r'&.*?;')

def clean_comment(comment):
    c = str(comment.encode("utf-8"))
    c = html_tags.sub(' ', c)
    for tag, char in to_replace:
        c = c.replace(tag, char)
    c = hex_tags.sub(' ', c)
    return c

def text_generator(path=data_path):
    f = open(path)
    for i, l in enumerate(f):
        comment_data = json.loads(l)
        comment_text = comment_data["comment_text"]
        comment_text = clean_comment(comment_text)
        if i % 50000 == 0:
            print(i)
        yield comment_text
    f.close()

# model management
if load_tokenizer:
    print('Load tokenizer...')
    tokenizer = six.moves.cPickle.load(open(os.path.join(save_dir, tokenizer_fname), 'rb'))
else:
    print("Fit tokenizer...")
    tokenizer = text.Tokenizer(nb_words=max_features)
    tokenizer.fit_on_texts(text_generator())
    if save:
        print("Save tokenizer...")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        six.moves.cPickle.dump(tokenizer, open(os.path.join(save_dir, tokenizer_fname), "wb"))


# build a model in any case (train or test)
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
print('Compile model...')
model.compile(loss='mse', optimizer='rmsprop')        # buildSkipgram
    

# training process
if train_model:
    if load_model:
        print('Load model...')
        model.load_weights(os.path.join(save_dir, model_load_fname))

    sampling_table = sequence.make_sampling_table(max_features)

    for e in range(nb_epoch):
        print('-'*40)
        print('Epoch', e)
        print('-'*40)

        progbar = generic_utils.Progbar(tokenizer.document_count)
        samples_seen = 0
        losses = []
        
        for i, seq in enumerate(tokenizer.texts_to_sequences_generator(text_generator())):
            # get skipgram couples for one text in the dataset
            couples, labels = sequence.skipgrams(seq, max_features, window_size=4, negative_samples=1., sampling_table=sampling_table)
            if couples:
                # one gradient update per sentence (one sentence = a few 1000s of word couples)
                X = np.array(couples, dtype="int32")
                
                #loss = model.train(X, labels)
                loss = model.train_on_batch(X, labels)
                
                losses.append(loss)
                if len(losses) % 1000 == 0:
                    progbar.update(i, values=[("loss", np.mean(losses))])
                    losses = []
                    
                    print('\n')
                    print("Saving temp model...")
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    model.save_weights(os.path.join(save_dir, model_save_fname+'_tmp'), overwrite=True) 
                samples_seen += len(labels)

        print('Samples seen:', samples_seen)
    print("Training completed!")

    if save:
        print("Saving model...")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model.save_weights(os.path.join(save_dir, model_save_fname), overwrite=True ) 
        


if test_model:
    print("It's test time!")
    print('Load model...')
    model.load_weights(os.path.join(save_dir, model_load_fname))
    
    # recover the embedding weights trained with skipgram:
    weights = model.layers[0].get_weights()[0]
    
    # we no longer need this
    del model
    
    weights[:skip_top] = np.zeros((skip_top, dim_proj))
    norm_weights = np_utils.normalize(weights)
    
    word_index = tokenizer.word_index
    reverse_word_index = dict([(v, k) for k, v in list(word_index.items())])
    word_index = tokenizer.word_index

    def embed_word(w):
        i = word_index.get(w)
        if (not i) or (i<skip_top) or (i>=max_features):
            return None
        return norm_weights[i]
    
    def closest_to_point(point, nb_closest=10):
        proximities = np.dot(norm_weights, point)
        tups = list(zip(list(range(len(proximities))), proximities))
        tups.sort(key=lambda x: x[1], reverse=True)
        return [(reverse_word_index.get(t[0]), t[1]) for t in tups[:nb_closest]]  

    def closest_to_word(w, nb_closest=10):
        i = word_index.get(w)
        if (not i) or (i<skip_top) or (i>=max_features):
            return []
        return closest_to_point(norm_weights[i].T, nb_closest)

        
    ''' the resuls in comments below were for: 
    5.8M HN comments
    dim_proj = 256
    nb_epoch = 2
    optimizer = rmsprop
    loss = mse
    max_features = 50000
    skip_top = 100
    negative_samples = 1.
    window_size = 4
    and frequency subsampling of factor 10e-5. 
    '''

    words = ["article", # post, story, hn, read, comments
             "3", # 6, 4, 5, 2
             "two", # three, few, several, each
             "great", # love, nice, working, looking
             "horrible",
             "insane",
             "love",
             "rtfm",
             "data", # information, memory, database
             "money", # company, pay, customers, spend
             "years", # ago, year, months, hours, week, days
             "android", # ios, release, os, mobile, beta
             "javascript", # js, css, compiler, library, jquery, ruby
             "look", # looks, looking
             "business", # industry, professional, customers
             "company", # companies, startup, founders, startups
             "after", # before, once, until
             "own", # personal, our, having
             "us", # united, country, american, tech, diversity, usa, china, sv
             "using", # javascript, js, tools (lol)
             "here", # hn, post, comments
         ]


    
    while(True):
        print("Give me a word. I'll find what you mean ['stop' to quit]")
        word = str(raw_input()).lower()
        if word == 'stop':
            break
        else:
            res = closest_to_word(word, 5)
            print("I think that with '%s' you also mean" %word)
            for r in res:
                print(r)
    """
    for w in words:
        res = closest_to_word(w, 5)
        print('====', w)
        for r in res:
            print(r)
    """
