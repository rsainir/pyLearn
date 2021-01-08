import numpy as np
# import sequential model type from keras: linear stack of neural net layers: used for feed forward cnn (convoluted nn)
from keras.models import Sequential
# neural network layers (used by most NNs)
from keras.layers import Dense, Dropout, Activation, Flatten
# convolutional layers
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
# utilities, used to transform data
from keras.utils import np_utils, to_categorical
# Load image data
# image data consists of handwritten digits
from keras.datasets import mnist
# plot sample 1
from matplotlib import pyplot as plt
# print(img_train.shape)  # prints (60000, 28, 28): 60k samples that are 28x28 pixels each
from tensorflow.python.keras.optimizers import SGD


def display_mnist(i, plotter, dataset):  # function to display ith element of data
    plotter.figure()
    plotter.imshow(dataset[i])
    plotter.colorbar()
    plotter.grid(False)
    plotter.show()


def load_mnist():  # Load MNIST data into train and test sets
    (img_train, label_train), (img_test, label_test) = mnist.load_data()
    # single color channel added
    img_train = img_train.reshape((img_train.shape[0], 28, 28, 1))
    img_test = img_test.reshape((img_test.shape[0], 28, 28, 1))
    # 10 different digits for integers, can transform into 10 element binary vector w 1 for value, 0 if not
    label_train = to_categorical(label_train)
    label_test = to_categorical(label_test)
    return img_train, label_train, img_test, label_test

    # print(img_train.shape)  # prints (60000, 28, 28), RGB 3 CHANNELS
    # display_mnist(10, plt, img_train)
    # transform our dataset from having shape (n, width, height) to (n, depth, width, height).

# see if we can preprocess the input data
# transform our dataset from having shape (n, width, height) to (n, depth, width, height).
# using tensorflow, preprocess by dividing both training set and testing set by 255.0


def prep_img(trains, tests):
    # need color depth of one for the channel, not 255, normalize
    prep_train = trains / 255.0
    prep_test = tests / 255.0
    display_mnist(10, plt, trains)
    return prep_train, prep_test  # normalized versions of our data sets


def def_model():
    # 2 aspects to model, front end (convolutional and pooling layers) and back end (classifier to make prediction)
    model = Sequential()
    # convolutional layer with (3,3) filter size, 32 filters
    # first ConvLayer is responsible for capturing the Low-Level features such as edges, color, gradient orientation
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    # max pooling layer: Pooling layer is responsible for reducing the spatial size of the Convolved Feature.
    # This is to decrease the computational power required to process the data through dimensionality reduction.
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    # interpretational dense layer
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    # output layer, use softmax to represent probabilities between 0 to 1
    model.add(Dense(10, activation='softmax'))
    # compilation of model
    opt = SGD(lr=0.01, momentum=0.9)  # config: stochastic gradient descent optimizer: learning rate and momentum
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def run_test_1():
    img_train, label_train, img_test, label_test = load_mnist()  # load
    img_train, img_test = prep_img(img_train, img_test)  # normalize
    display_mnist(0, plt, img_train)


np.random.seed(123)  # reproducibility
run_test_1()

# model:use sequential
# model = Sequential()
# model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(3, 28, 28)))

# print(model.output_shape)
