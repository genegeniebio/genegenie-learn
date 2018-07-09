'''
sbclearn (c) University of Manchester 2017

sbclearn is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
import collections
import itertools

from scipy.stats import linregress

import matplotlib.pyplot as plt
import numpy as np


# from matplotlib import lines
def plot(y_test, y_pred, title='Measured vs. predicted'):
    '''Plot results.'''
    plt.title(title)
    plt.xlabel('Measured')
    plt.ylabel('Predicted')

    y_test = [y for y in y_test]
    y_pred = [y for y in y_pred]

    slope, _, r_value, _, _ = linregress(y_test, y_pred)
    label = 'm=%0.3f, r2=%0.3f' % (slope, r_value)

    plt.scatter(y_test, y_pred)

    fit = np.poly1d(np.polyfit(y_test, y_pred, 1))

    ret = plt.plot(y_test,
                   fit(y_test),
                   label=label,
                   linewidth=1,
                   )

    plt.legend(handles=[ret[0]])
    plt.show()
