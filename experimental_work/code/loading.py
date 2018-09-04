import cPickle
import gzip
import os

import theano
import theano.tensor as T
 
import numpy as np



def load_data(dataset, datasource='mnist'):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    
    :type datasource: string
    :param datasouce: the type of dataset (mnist, matrix, csv)
    '''

    #############
    # LOAD DATA #
    #############
    if datasource == 'mnist':
        # Download the MNIST dataset if it is not present
        data_dir, data_file = os.path.split(dataset)
        if data_dir == "" and not os.path.isfile(dataset):
            # Check if dataset is in the data directory.
            new_path = os.path.join(
                os.path.split(__file__)[0],
                "..",
                "data",
                dataset
            )
            if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
                dataset = new_path
                
        if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
            import urllib
            origin = (
                'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
            )
            print 'Downloading data from %s' % origin
            urllib.urlretrieve(origin, dataset)
            
        print '... loading data'
        
        # Load the dataset
        f = gzip.open(dataset, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

    if datasource == 'csv':
        #print 'Generating dataset'
        #generateData(n=50000, savefile=dataset) 
        print 'Loading synthetic dataset'
        data = np.genfromtxt(dataset, delimiter=',')
        train_set, valid_set, test_set = splitData(data)
        
    
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    #shared dataset foo here    

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval



def splitData(dataset, weights=[.8, .1, .1]):
    """
    Split dataset in train, valid, test sets
    
    :type dataset: numpy array
    :param dataset: input dataset to split 

    :type weights: list
    :param weights: list of percentage of samples in train, valid, test respectively

    Return: x,y,z where x is tuple (input, target)
    """
    # rows are observations, cols are features
    #dataset = dataset.T      

    nrows,ncols = dataset.shape
    indexes = np.arange(nrows)
    np.random.shuffle(indexes)
    
    start = 0
    end   = weights[0]*nrows
    train = dataset[indexes[start:end]]
    train = train[~np.isnan(train[:,:-1]).any(axis=1)]
    train_x = train[:, :-1]
    train_y = train[:,-1]
    
    start = end+1
    end   = start+weights[1]*nrows    
    valid = dataset[indexes[start:end]]
    valid = valid[~np.isnan(valid[:,:-1]).any(axis=1)]
    valid_x = valid[:,:-1]
    valid_y = valid[:,-1]
    
    start = end+1
    test = dataset[indexes[start:]]
    test = test[~np.isnan(test[:,:-1]).any(axis=1)]
    test_x = test[:, :-1]
    test_y = test[:,-1]
    
    return (train_x, train_y), (valid_x,valid_y), (test_x, test_y)
    
    


def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables
    
    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                        dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                        dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')


def generateData(n=200, savefile=''):
    np.random.seed(420) # random seed for consistency
    
    mu_vec1 = np.array([0,0,0,0])
    cov_mat1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0], [0,0,0,1]])
    class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, n)
    class1_label  = np.random.sample(n)
    class1_sample = np.column_stack((class1_sample, class1_label))
    #assert class1_sample.shape == (3,n), "The matrix has not the dimensions 3x20"
   
    mu_vec2 = np.array([1,1,1,1])
    cov_mat2 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, n)
    class2_label = np.random.sample(n)
    class2_sample = np.column_stack((class2_sample, class2_label))                           
    #assert class1_sample.shape == (3,n), "The matrix has not the dimensions 3x20"
    
    all_samples = np.concatenate((class1_sample, class2_sample), axis=0)
    #assert all_samples.shape == (3,n*2), "The matrix has not the dimensions 3x40"
    
    if (len(savefile)):
        path = '../data/'+savefile
        np.savetxt(path, all_samples, delimiter=',')
        #all_samples.tofile(path, sep=",")
    
    

    #%pylab inline
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d import proj3d

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    plt.rcParams['legend.fontsize'] = 10
    
    ax.plot(class1_sample[0,:-1], class1_sample[1,:-1], class1_sample[2,:-1],
            'o', markersize=8, color='blue', alpha=0.5, label='class1')
    ax.plot(class2_sample[0,:-1], class2_sample[1,:-1], class2_sample[2,:-1],
            '^', markersize=8, alpha=0.5, color='red', label='class2')
    
    plt.title('Samples for class 1 and class 2')
    ax.legend(loc='upper right')
    
    plt.show()

    return all_samples
    
