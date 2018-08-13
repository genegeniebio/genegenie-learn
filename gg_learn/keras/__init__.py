'''
synbiochem (c) University of Manchester 2017

synbiochem is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=too-many-arguments
from keras import backend
from keras import layers, models
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from scipy import stats
from sklearn.model_selection import KFold

from sbclearn.utils.plot_utils import plot_stats, plot_linreg


def coeff_determ(y_true, y_pred):
    '''coeff_determination.'''
    ss_res = backend.sum(backend.square(y_true - y_pred))
    ss_tot = backend.sum(backend.square(y_true - backend.mean(y_true)))
    return 1 - ss_res / (ss_tot + backend.epsilon())


def get_classify_model(input_length,
                       output_dim=5,
                       layer_units=[32, 32],
                       dropout=0.2,
                       loss='binary_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy']):
    '''Get model.'''
    model = models.Sequential()

    model.add(layers.Embedding(input_dim=21,
                               output_dim=output_dim,
                               input_length=input_length))

    for idx, layer in enumerate(layer_units):
        model.add(layers.LSTM(layer,
                              return_sequences=idx != len(layer_units) - 1))
        model.add(layers.Dropout(dropout))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)

    return model


def get_regress_model(input_shape,
                      layer_units=[20, 7, 5],
                      activation='relu',
                      loss='mean_squared_error',
                      optimizer='adam',
                      metrics=[coeff_determ]):
    '''Get model.'''
    model = models.Sequential()

    for units in layer_units:
        model.add(layers.Dense(units=units,
                               activation=activation,
                               input_shape=(input_shape,)))
        input_shape = units

    model.add(layers.Dense(units=1))

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model


def k_folds_nn(x, y, param, n_splits=3, batch_size=None, epochs=500):
    '''Perform k-folds neural network.'''
    if not batch_size:
        batch_size = len(x)

    kfold = KFold(n_splits=n_splits, random_state=0, shuffle=True)

    for i, (train, test) in enumerate(kfold.split(x, y)):
        x_train, x_test = x[train], x[test]
        y_train, y_test = y[train], y[test]

        model = get_regress_model(len(x[0]))

        stats = model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=(x_test, y_test))

        plot_stats(stats.history, 'stats_' + str(i + 1) + '.svg', param)

        x_train_predict = model.predict(x_train).flatten()
        x_test_predict = model.predict(x_test).flatten()
        plot_linreg(x_train_predict, x_test_predict, y_train, y_test, i)


class Classifier(object):
    '''Simple classifier in keras.'''

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
        model.add(Dense(64,
                        input_dim=self.__x_data.shape[1],
                        activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.__num_outputs, activation='softmax'))

        return model
