'''
synbiochem (c) University of Manchester 2015

synbiochem is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments

from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy
import scipy.stats
import theanets

import sbclearn


class TheanetsBase(object):
    '''Base class for Classifier and Regressor.'''

    def __init__(self, network, data, outputs):
        self._network = network
        self._x_data = sbclearn.pad(data[0])
        self._y_data = sbclearn.pad(data[1])
        self._outputs = outputs
        self._exp = None

        # Check lengths of x_data and y_data are equal:
        assert len(self._x_data) == len(self._y_data)

    def train(self, split=0.75, hidden_layers=None, hyperparams=None):
        '''Train the network.'''
        if hidden_layers is None:
            hidden_layers = [1024]

        if hyperparams is None:
            hyperparams = {}

        layers = [len(self._x_data[0])] + hidden_layers + [self._outputs]
        self._exp = theanets.Experiment(self._network, layers=layers)

        # Split data into training and validation:
        ind = int(split * len(self._x_data))
        self._exp.train((self._x_data[:ind], self._y_data[:ind]),
                        (self._x_data[ind:], self._y_data[ind:]),
                        **hyperparams)


class Classifier(TheanetsBase):
    '''Simple classifier in Theanets.'''

    def __init__(self, x_data, y_data):
        y_enum = sbclearn.enum_list(y_data)
        y_data = numpy.array([y[1] for y in y_enum], dtype=numpy.int32)
        self.__y_map = dict(set(y_enum))

        super(Classifier, self).__init__(theanets.Classifier, (x_data, y_data),
                                         len(self.__y_map))

    def predict(self, x_test, y_test):
        '''Classifies and analyses test data.'''
        y_pred = self._exp.network.classify(sbclearn.pad(x_test))

        y_test = numpy.array([self.__y_map[y]
                              for y in y_test], dtype=numpy.int32)

        inv_y_map = {v: k for k, v in self.__y_map.items()}

        return [inv_y_map[y] for y in y_pred], inv_y_map, \
            classification_report(y_test, y_pred), \
            confusion_matrix(y_test, y_pred), f1_score(y_test, y_pred,
                                                       average='macro')


class Regressor(TheanetsBase):
    '''Simple regressor in Theanets.'''

    def __init__(self, x_data, y_data):
        super(Regressor, self).__init__(theanets.Regressor, (x_data, y_data),
                                        len(y_data[0]))

    def predict(self, x_test, y_test=None):
        '''Classifies and analyses test data.'''
        y_preds = [val[0] for val in self._exp.network.predict(x_test)]
        error = None

        if y_test is not None:
            # R squared:
            _, _, r_value, _, _ = \
                scipy.stats.linregress(y_test, y_preds)

            error = 1 - r_value

        return y_preds, error
