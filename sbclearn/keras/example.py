'''
sbclearn (c) University of Manchester 2017

sbclearn is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=no-member
# pylint: disable=too-many-arguments
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn import model_selection

import numpy as np


def get_data(num_classes=10):
    '''example method.'''
    # Generate dummy data
    x_data = np.random.random((1000, 20))
    y_data = to_categorical(np.random.randint(num_classes, size=(1000, 1)),
                            num_classes=num_classes)

    return x_data, y_data


def classify((x_data, y_data), test_size=0.25, epochs=20, batch_size=128,
             learn_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True):
    '''classify method.'''
    x_train, x_test, y_train, y_test = \
        model_selection.train_test_split(x_data, y_data, test_size=test_size)

    model = _get_model(x_data, y_data)

    sgd = SGD(lr=learn_rate, decay=decay, momentum=momentum, nesterov=nesterov)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    return (model.metrics_names,
            model.evaluate(x_test, y_test, batch_size=batch_size))


def _get_model(x_data, y_data):
    '''Gets model.'''
    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    model.add(Dense(64, activation='relu', input_dim=x_data.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(set([tuple(val) for val in y_data])),
                    activation='softmax'))

    return model


def main():
    '''main method.'''
    print classify(get_data())


if __name__ == '__main__':
    main()
