'''
synbiochem (c) University of Manchester 2017

synbiochem is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=too-many-arguments
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD


class Classifier(object):
    '''Simple classifier in Theanets.'''

    def __init__(self, x_data, y_data):
        self.__num_outputs = len(set([tuple(val) for val in y_data]))
        self.__x_data = x_data
        self.__y_data = y_data
        self.__model = self.__get_model()

    def train(self, learn_rate=0.01, decay=1e-6, momentum=0.9, epochs=20,
              batch_size=128, nesterov=True):
        '''Train classifier.'''
        sgd = SGD(lr=learn_rate, decay=decay,
                  momentum=momentum, nesterov=nesterov)

        self.__model.compile(loss='categorical_crossentropy',
                             optimizer=sgd)

        self.__model.fit(self.__x_data, self.__y_data, epochs=epochs,
                         batch_size=batch_size)

    def predict(self, x_test, batch_size=128):
        '''Classifies test data.'''
        return self.__model.predict(x_test, batch_size=batch_size)

    def __get_model(self):
        '''Gets model.'''
        model = Sequential()
        # Dense(64) is a fully-connected layer with 64 hidden units.
        # in the first layer, you must specify the expected input data shape:
        # here, 20-dimensional vectors.
        model.add(Dense(64, activation='relu',
                        input_dim=self.__x_data.shape[1]))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.__num_outputs, activation='softmax'))

        return model
