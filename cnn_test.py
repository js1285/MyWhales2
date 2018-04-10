# Deep learning lab course final project.  Kaggle whale
# classification.

# Test the model builder on an MNIST architecture.


import tensorflow as tf
import os, gzip
import pickle as cPickle
from cnn import build_model


def mnist(datasets_dir='./data/mnist'):
    """Download MNIST data set. (course repo)."""
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'mnist.pkl.gz')
    if not os.path.exists(data_file):
        print('... downloading MNIST from the web')
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    # Load the dataset
    f = gzip.open(data_file, 'rb')
    try:
        train_set, valid_set, test_set = cPickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    test_x, test_y = test_set
    test_x = test_x.astype('float32')
    test_x = test_x.astype('float32').reshape(test_x.shape[0], 1, 28, 28)
    test_y = test_y.astype('int32')
    valid_x, valid_y = valid_set
    valid_x = valid_x.astype('float32')
    valid_x = valid_x.astype('float32').reshape(valid_x.shape[0], 1, 28, 28)
    valid_y = valid_y.astype('int32')
    train_x, train_y = train_set
    train_x = train_x.astype('float32').reshape(train_x.shape[0], 1, 28, 28)
    train_y = train_y.astype('int32')
    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    print('... done loading data')
    return rval


def build_model_test():
    # model specifications according to Fabian's lab course assignment 2:
    layers = [
        ("conv", 17, 4, None, tf.nn.relu),
        ("pool", None, 5, None, None),
        ("conv", 16, 3, None, tf.nn.relu),
        ("pool", None, 200, None, None),
        ("flatten", None, None, None, None),
        ("dense", None, None, 127, tf.nn.relu)        
    ]

    params = {"layers_list": layers,
              "optimizer": "SGD",
              "optimizer_params": (0.001,),
              "n_classes": 10,
              "image_x": 28,
              "image_y": 28}
    
    mnist_classifier = tf.estimator.Estimator(
        model_fn=build_model,
        model_dir="tmp/mnist_model3",
        params=params)

    train_data = mnist()[0]
    input_fn = lambda: (train_data[0][:3000], train_data[1][:3000])
    
    mnist_classifier.train(input_fn=input_fn, steps=1)
    print("done training. evaluating:")
    res= mnist_classifier.evaluate(input_fn=input_fn, steps=1)
    print(res)
                           

build_model_test()
