'''
sbclearn (c) University of Manchester 2017

sbclearn is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=invalid-name
from os import listdir
from os.path import isdir, join
import sys

from sklearn import model_selection

from sbclearn.theanets.theanets_utils import Classifier
import debrief_db
import numpy as np


def get_b_factors(filename):
    '''Gets b factors.'''
    b_factors = {}

    for dir_name in [fle for fle in listdir(filename)
                     if isdir(join(filename, fle))]:

        mutant_dir = join(filename, dir_name)

        for fle in listdir(mutant_dir):
            if fle.startswith('b-factor'):
                b_factors[dir_name] = _read_bfactors(join(mutant_dir, fle))

    return b_factors


def get_xcorrel(filename):
    '''Gets data.'''
    xcorrel = {}

    for dir_name in [fle for fle in listdir(filename)
                     if isdir(join(filename, fle))]:

        mutant_dir = join(filename, dir_name)

        for fle in listdir(mutant_dir):
            if fle.startswith('xcorrel'):
                xcorrel[dir_name] = _read_xcorrel(join(mutant_dir, fle))

    return xcorrel


def get_activity(name):
    '''Get activity.'''
    sheet = '1-dcR5dPaYwtH38HNYqBieOSaqMz-31N8aEdEb3IqRkw'
    client = debrief_db.DEBriefDBClient(sheet, 'MAO-N', 'A:R')
    return client.get_activity('' if name == 'wt' else name)


def learn(x_data, y_data):
    '''Learn.'''
    for _ in range(50):
        x_train, x_test, y_train, y_test = \
            model_selection.train_test_split(x_data, y_data, test_size=0.2)

        classifier = Classifier(x_train, y_train)
        classifier.train()
        y_preds = classifier.predict(x_test)

        for tup in zip(y_test, y_preds):
            print tup


def _read_bfactors(filename):
    '''Reads b-factors.'''
    with open(filename) as fle:
        return [float(line.split()[1]) for line in fle]


def _read_xcorrel(filename):
    with open(filename) as fle:
        return [[f(v) for (f, v) in zip((int, int, float), line.split())]
                for line in fle]


def main(args):
    '''main method.'''
    b_factors = get_b_factors(args[0])

    x_vals = np.array(b_factors.values())
    y_vals = np.array([get_activity(key) for key in b_factors])

    learn(x_vals, y_vals)


if __name__ == '__main__':
    main(sys.argv[1:])
