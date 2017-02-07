'''
HolyGrail (c) University of Manchester 2015

HolyGrail is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments
from collections import defaultdict

import numpy
import scipy.stats

from sbclearn.theanets.theanets_utils import Regressor
import sbclearn
import synbiochem.optimisation.gen_alg as gen_alg


class LearnGeneticAlgorithm(gen_alg.GeneticAlgorithm):
    '''Class to optimise parameters for Classifier using a GA.'''

    def __init__(self, pop_size, data, split, tests, args,
                 retain=0.2, random_select=0.05, mutate=0.01, verbose=False):
        '''Constructor.'''
        super(LearnGeneticAlgorithm, self).__init__(pop_size, args,
                                                    retain, random_select,
                                                    mutate, verbose)

        self.__data = data
        self.__split = split
        self.__tests = tests

    def _fitness(self, individual):
        '''Determine the fitness of an individual.'''

        # Form hidden layers array:
        num_hidden_layers = individual.pop('num_hidden_layers', 1)
        activ_func = individual.pop('activ_func', 'relu')
        num_nodes = individual.pop('num_nodes', 128)
        hidden_layers = [(num_nodes, activ_func)] * num_hidden_layers

        return_vals = self.__predict(hidden_layers, individual)

        # Reform individual dict:
        individual['num_hidden_layers'] = num_hidden_layers
        individual['activ_func'] = activ_func
        individual['num_nodes'] = num_nodes

        if self._verbose:
            print ('%.3f\t' % return_vals[0]) + str(individual)

        return return_vals[0]

    def __predict(self, hidden_layers=None, hyperparams=None):
        '''Learn method.'''
        results = defaultdict(list)

        for _ in range(self.__tests):
            data_split = sbclearn.split_data(self.__data, self.__split)

            regressor = Regressor(data_split[0][0],
                                  [[val] for val in data_split[1][0]])

            regressor.train(hidden_layers=hidden_layers,
                            hyperparams=hyperparams)

            y_preds, _ = regressor.predict(data_split[0][1],
                                           data_split[1][1])

            for tup in zip(data_split[1][1], y_preds):
                results[tup[0]].append(tup[1])

        _, _, r_value, _, _ = \
            scipy.stats.linregress(results.keys(),
                                   [numpy.mean(pred)
                                    for pred in results.values()])

        return 1 - r_value, results
