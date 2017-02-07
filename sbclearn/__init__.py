'''
synbiochem (c) University of Manchester 2015

synbiochem is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author: neilswainston
'''
# pylint: disable=no-member
from collections import defaultdict
from functools import partial
from itertools import count
import random

import numpy


def split_data(data, split=0.9):
    '''Split data.'''
    data_rand = randomise_order(data)

    # Split data into training and classifying:
    ind = int(split * len(data_rand[0]))

    return [(array[:ind], array[ind:]) for array in data_rand]


def randomise_order(data):
    '''Assumes data are ordered by index and then randomises their orders such
    that this order is maintained.'''
    data = zip(*data)
    random.shuffle(data)
    return zip(*data)


def pad(data):
    '''Pad data with average values if sublists of different lengths.'''
    try:
        max_len = max([len(x) for x in data])
        mean_val = numpy.mean([x for sublist in data for x in sublist])
        return numpy.array([x + [mean_val] * (max_len - len(x)) for x in data],
                           dtype=numpy.float32)
    except TypeError:
        # For 1D data, elements cannot be passed to len:
        return data


def enum_list(lst):
    '''Returns enumeration of supplied list.'''
    label_to_number = defaultdict(partial(next, count()))
    return [(item, label_to_number[item]) for item in lst]
