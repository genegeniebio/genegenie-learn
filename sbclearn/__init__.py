'''
sbclearn (c) University of Manchester 2017

sbclearn is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
import collections
import itertools

from matplotlib import lines, pyplot
from scipy.stats import linregress

import numpy as np


# KD Hydrophobicity, EIIP, Helix, Sheet, Turn
AA_PROPS = {
    'A': [1.8, -0.0667, 32.9, -23.6, -41.6],
    'R': [-4.5, 0.2674, 0, -6.2, -5.1],
    'N': [-3.5, -0.2589, -24.8, -41.6, 44.5],
    'D': [-3.5, 0.4408, 5.8, -41.6, 37.8],
    'C': [2.5, 0.1933, -5.1, 6.8, 17.4],
    'Q': [-3.5, 0.1545, 11.3, 0, -2],
    'E': [-3.5, -0.2463, 36.5, -67.3, -30.1],
    'G': [-0.4, -0.2509, -46.2, -13.9, 44.5],
    'H': [-3.2, -0.1414, 11.3, -18.6, -5.1],
    'I': [4.5, -0.2794, -1, 45.1, -75.5],
    'L': [3.8, -0.2794, 26.2, 15.7, -52.8],
    'K': [-3.9, -0.0679, 19.1, -31.5, 1],
    'M': [1.9, 0.1899, 27.8, 1, -51.1],
    'F': [2.8, 0.26, 10.4, 20.7, -51.1],
    'P': [-1.6, -0.1665, -59.8, -47.8, 41.9],
    'S': [-0.8, 0.1933, -32.9, -6.2, 35.8],
    'T': [-0.7, 0.2572, -24.8, 28.5, -4.1],
    'W': [-0.9, 0.0331, 3, 21.5, -4.1],
    'Y': [-1.3, 0.0148, -31.5, 27, 13.1],
    'V': [4.2, -0.2469, -3, 49.5, -69.3]
}

NUM_AA_PROPS = len(AA_PROPS['A'])


def plot(results, title, color='b'):
    '''Plot results.'''
    pyplot.title(title)
    pyplot.xlabel('Measured')
    pyplot.ylabel('Predicted')

    pyplot.errorbar(results.keys(),
                    [np.mean(pred) for pred in results.values()],
                    yerr=[np.std(pred) for pred in results.values()],
                    markersize=3,
                    fmt='o',
                    color=color)

    slope, _, r_value, _, _ = \
        linregress(results.keys(),
                   [np.mean(pred) for pred in results.values()])

    label = 'm=%0.3f, r2=%0.3f' % (slope, r_value)

    fit = np.poly1d(np.polyfit(results.keys(),
                               [np.mean(pred)
                                for pred in results.values()], 1))

    ret = pyplot.plot(results.keys(),
                      fit(results.keys()),
                      label=label,
                      linewidth=1,
                      color=color)

    pyplot.legend(handles=[ret[0]])
    pyplot.show()


def get_aa_props(all_sequences, scale=(0.1, 0.9)):
    '''Returns input data for machine-learning problems.'''
    scaled = __scale(scale)
    mean_value = np.mean([x for sublist in scaled.values()
                          for x in sublist])
    return [list(itertools.chain.from_iterable([scaled[am_acid]
                                                if am_acid in scaled
                                                else [mean_value] *
                                                len(scaled['A'])
                                                for am_acid in sequences]))
            for sequences in all_sequences]


def __scale(scale):
    '''Scale amino acid properties.'''
    scaled = collections.defaultdict(list)

    for i in range(NUM_AA_PROPS):
        props = {key: value[i] for key, value in AA_PROPS.iteritems()}
        min_val, max_val = min(props.values()), max(props.values())

        for key, value in props.iteritems():
            scaled[key].append(scale[0] + (scale[1] - scale[0]) *
                               (value - min_val) / (max_val - min_val))

    return scaled
