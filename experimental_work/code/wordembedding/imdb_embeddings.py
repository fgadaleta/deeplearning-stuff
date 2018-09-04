import random
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Dropout, Activation
import utils as ut

############################# main #########################
ready = True

if ready:  # just load stuff 
    pars_idx = ut.loadStuff('./data/pars_idx.pkl')
    pars = ut.loadStuff('./data/pars.pkl')
    voc = ut.loadStuff('./data/voc.pkl')
    X,y = ut.loadStuff('./data/dataset.pkl')  

else:       # compute all and store 
    pars = ut.loadTextFiles('/home/frag/Documents/python-code/word2vec/aclImdb/train/pos/') 
    voc = ut.buildVocabulary(pars)
    #pars = pars[:100]   # remove me 
    pars_idx = ut.pars2idx(pars, voc)
    
    ut.saveStuff(pars_idx, './data/pars_idx.pkl')
    ut.saveStuff(pars, './data/pars.pkl')
    ut.saveStuff(voc, './data/voc.pkl')
    
    #map(lambda i: voc[i], X[0])
    
    ####################################################################
    # preparing dataset 
    # randomly break a paragraph with one random word and annotate the 
    # respose as non-valid
    ####################################################################
    data = pars_idx       # copy paragraphs and break 
    y = [0]*len(data)     # list of responses (1 valid, 0 not valid) 
    
    for i,par in enumerate(data):
        # decide to break or not
        bw = random.randint(0,1)   # broken words
        
        if bw == 0:         # if not
            y[i] = 1
        else:
            breakfrom = random.randint(0, len(par)-1)
            breakto   = random.randint(0, len(voc)-1)
            
            print('Changing "%s" to "%s" of paragraph [%d]' 
                  %(voc[breakfrom], voc[breakto], i))
            
            par[breakfrom] = breakto
            y[i] = 0
        
    dataset = (data, y)
    ut.saveStuff(dataset, './data/dataset.pkl')
    
    (X,y) = dataset



########################################################
## Train from X, y
########################################################
# padding to maximum lenght of paragraph
# TODO use a sliding window to handle paragraphs with different lenght
maxlen = 256
batch_size = 64
test_split = 0.2

X = sequence.pad_sequences(X, maxlen=maxlen) 

# print sentence in human language
#sentence = ""
#for i in X[1]:
#    sentence += "".join(" "+voc[i])
#print sentence 

# prepare data split
X_train = X[:int(len(X)*(1-test_split))]
y_train = y[:int(len(X)*(1-test_split))]
X_test = X[int(len(X)*(1-test_split)):]
y_test = y[int(len(X)*(1-test_split)):]

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

max_features = len(voc) 

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
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=15, validation_data=(X_test, y_test), show_accuracy=True)


print('Extracting embeddings')
emb = model.layers[0]
embeddings = emb.W.get_value()

print('Saving embeddings and vocabulary for t-SNE')
saveStuff(embeddings, '/home/frag/Documents/python-code/tsne/data/imdb_embeddings.pkl')
saveStuff(voc, '/home/frag/Documents/python-code/tsne/data/imdb_voc.pkl')
