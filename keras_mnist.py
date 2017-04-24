# external
import numpy as np
import matplotlib.pyplot as plt
# keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

# make the plot figures bigger
plt.rcParams['figure.figsize'] = (7,7)

# loading the training data
nb_classes = 10
# we shuffle and split the data between training and testing setts
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# (uncomment to ensure you've gotten to this point correctly)
# print('x_train dimensions', x_train.shape) # should print (60000, 28, 28)
# print('y_train dimensions', y_train.shape) # should print (60000,)

# looking at some examples of training data (uncomment to see this)
# for i in range(9):
#     plt.subplot(3,3,i+1)
#     plt.imshow(x_train[i], cmap='gray', interpolation='none')
#     plt.title("Class {}".format(y_train[i])) # print's label above the data image

# reshaping the inout so that each 28x28 pixel image becomes a single
# 784 dimensional vector; each of the inputs is also scaled to be in the
# range [0,1] instead of [0,255]
x_train = x_train.reshape(60000, 784)
x_train = x_train.astype('float32')
x_train /= 255

x_test = x_test.reshape(10000, 784)
x_test = x_test.astype('float32')
x_test /= 255

# modifying the target matrices to be in the one-hot format:
# 0 -> [1, 0, 0, 0, 0, 0, 0, 0, 0]
# 1 -> [0, 1, 0, 0, 0, 0, 0, 0, 0], etc.
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)


# building a simple 3-layer fully-connected network
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu')) # activation is a nonlinear function applied to the output later
                              # 'relu' makes all values below 0 -> 0
model.add(Dropout(0.2)) # dropout helps protect from overfitting to the training data
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10)) # for the 10 different digits we want a prob dist. around
model.add(Activation('softmax')) # 'softmax' ensures the output is a valid probability dist.

# compile the model
# categorical crossentropy is a loss function well-suited for comparing two probability
# distributions; our predictions are prob. dists. over the 10 digits an image can be
# cross-entropy is a measure of how different your predicted dist. is from the target dist.
# the optimizer helps determine how quickly the model learns and how resistant it is to
# diverging or never-converging; 'adam' is often a good choice
model.compile(loss='categorical_crossentropy', optimizer='adam')

# train the model
model.fit(x_train,
          y_train,
          batch_size=128,
          epochs=4,
          verbose=1,
          validation_data=(x_test, y_test))

# evaluate it's performance
score = model.evaluate(x_test,
                       y_test,
                       verbose=0)
print('test score: ', score)

# inspecting the output
# predict_classes function outputs the highest probability class according to the trained
# classifier for each input example
predicted_classes = model.predict_classes(x_test)

# check which items we labeled correctly and incorrectly
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]

# print some correctly predicted images and some incorrectly predicted images
plt.figure()
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))
    plt.show()

plt.figure()
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))
    plt.show()
