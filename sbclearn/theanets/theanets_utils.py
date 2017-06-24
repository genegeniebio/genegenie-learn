'''
sbclearn (c) University of Manchester 2017

sbclearn is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=too-few-public-methods
from collections import defaultdict

from sklearn import model_selection
import theanets

import numpy as np


class TheanetsBase(object):
    '''Base class for Classifier and Regressor.'''

    def __init__(self, network, x_data, y_data):
        self._network = network
        self._x_data = x_data
        self._y_data = y_data
        self._exp = None

        assert len(self._x_data) == len(self._y_data)
        assert len(set([len(row) for row in self._x_data])) == 1

    def _train(self, num_outputs, valid_size, hidden_layers, hyperparams):
        '''Train the network.'''
        if hidden_layers is None:
            hidden_layers = [len(self._x_data)]

        if hyperparams is None:
            hyperparams = {}

        layers = [len(self._x_data[0])] + hidden_layers + [num_outputs]

        self._exp = theanets.Experiment(self._network, layers=layers)

        x_train, x_valid, y_train, y_valid = \
            model_selection.train_test_split(self._x_data, self._y_data,
                                             test_size=valid_size)

        self._exp.train((x_train, y_train), (x_valid, y_valid), **hyperparams)


class Classifier(TheanetsBase):
    '''Simple classifier in Theanets.'''

    def __init__(self, x_data, y_data):
        super(Classifier, self).__init__(theanets.Classifier, x_data,
                                         y_data.astype(np.int32))

    def train(self, valid_size=0.25, hidden_layers=None, hyperparams=None):
        '''Train classifier.'''
        num_outputs = len(set(self._y_data))

        super(Classifier, self)._train(num_outputs, valid_size, hidden_layers,
                                       hyperparams)

    def predict(self, x_test):
        '''Classifies test data.'''
        return self._exp.network.classify(x_test)


class Regressor(TheanetsBase):
    '''Simple regressor in Theanets.'''

    def __init__(self, x_data, y_data):
        super(Regressor, self).__init__(theanets.Regressor, x_data,
                                        np.array([[y] for y in y_data]))

    def train(self, valid_size=0.25, hidden_layers=None, hyperparams=None):
        '''Train regressor.'''
        num_outputs = len(self._y_data[0])

        super(Regressor, self)._train(num_outputs, valid_size, hidden_layers,
                                      hyperparams)

    def predict(self, x_test):
        '''Predicts test data.'''
        return self._exp.network.predict(x_test)


def k_fold_cross_valid((x_data, y_data), tests=50, test_size=0.05,
                       hidden_layers=None, hyperparams=None):
    '''k-fold cross validation.'''
    results = defaultdict(list)

    for _ in range(tests):
        x_train, x_test, y_train, y_test = \
            model_selection.train_test_split(x_data, y_data,
                                             test_size=test_size)

        regressor = Regressor(x_train, y_train)
        regressor.train(hidden_layers=hidden_layers, hyperparams=hyperparams)

        for test, pred in zip(y_test, regressor.predict(x_test)):
            results[test].append(pred[0])

    return results

# hyperparams = {
    # 'input_noise': [i / 10.0 for i in range(0, 10)],
    # 'hidden_noise': [i / 10.0 for i in range(0, 10)],
    # 'activ_func': 'relu',
    # 'learning_rate': 0.004,
    # 'momentum': 0.6,
    # 'patience': 3,
    # 'min_improvement': 0.1,
    # 'validate_every': range(1, 25),
    # 'batch_size': range(10, 50, 10),
    # 'hidden_dropout': [i * 0.1 for i in range(0, 10)],
    # 'input_dropout': [i * 0.1 for i in range(0, 10)]
#    }
