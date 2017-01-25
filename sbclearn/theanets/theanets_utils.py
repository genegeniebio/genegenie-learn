'''
synbiochem (c) University of Manchester 2015

synbiochem is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=no-member
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments

from collections import defaultdict
from functools import partial
from itertools import count
import random

from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy
import theanets


class TheanetsBase(object):
    '''Base class for Classifier and Regressor.'''

    def __init__(self, network, data, outputs):
        self._network = network
        self._x_data = _pad(data[0])
        self._y_data = _pad(data[1])
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
        y_enum = _enumerate(y_data)
        y_data = numpy.array([y[1] for y in y_enum], dtype=numpy.int32)
        self.__y_map = dict(set(y_enum))

        super(Classifier, self).__init__(theanets.Classifier, (x_data, y_data),
                                         len(self.__y_map))

    def predict(self, x_test, y_test):
        '''Classifies and analyses test data.'''
        y_pred = self._exp.network.classify(_pad(x_test))

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

    def predict(self, x_test, y_test=None, results=None):
        '''Classifies and analyses test data.'''
        preds = [val[0] for val in self._exp.network.predict(x_test)]

        if y_test is None:
            return preds, None
        else:
            if results is None:
                results = defaultdict(list)

            for tup in zip(*[y_test, preds]):
                results[tup[0]].append(tup[1])

            # The mean squared error:
            error = \
                numpy.mean([(x - y) ** 2
                            for x, y in zip(results.keys(),
                                            [numpy.mean(pred)
                                             for pred in results.values()])])

            return results, error


def split_data(data, split=0.9):
    '''Split data.'''
    x_data_rand, y_data_rand = randomise_order(data)

    # Split data into training and classifying:
    ind = int(split * len(x_data_rand))

    return x_data_rand[:ind], \
        [[y] for y in y_data_rand[:ind]], \
        x_data_rand[ind:], y_data_rand[ind:]


def randomise_order(data):
    '''Assumes data are ordered by index and then randomises their orders such
    that this order is maintained.'''
    data = zip(*data)
    random.shuffle(data)
    return zip(*data)


def _pad(data):
    '''Pad data with average values if sublists of different lengths.'''
    try:
        max_len = max([len(x) for x in data])
        mean_val = numpy.mean([x for sublist in data for x in sublist])
        return numpy.array([x + [mean_val] * (max_len - len(x)) for x in data],
                           dtype=numpy.float32)
    except TypeError:
        # For 1D data, elements cannot be passed to len:
        return data


def _enumerate(lst):
    '''Returns enumeration of supplied list.'''
    label_to_number = defaultdict(partial(next, count()))
    return [(item, label_to_number[item]) for item in lst]
