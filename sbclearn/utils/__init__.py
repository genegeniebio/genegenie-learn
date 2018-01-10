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
