'''
sbclearn (c) University of Manchester 2017

sbclearn is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=invalid-name
from collections import defaultdict
import sys

from sklearn import model_selection
import scipy.stats

from sbclearn.theanets.theanets_utils import Regressor
import numpy as np
import pandas as pd


def get_data(filename):
    '''Gets data.'''
    df = pd.read_table(filename, sep=',')
    _preprocess(df)

    # TODO: deal with duplicate sequences (same seq, different vals)
    x_data = np.array(_encode_seqs(df['r1'] + df['r2']))
    y_data = df['FC']
    labels = df['Construct']

    return x_data, y_data, labels


def learn(x_data, y_data, labels):
    '''Learn.'''
    hyperparams = {
        # 'input_noise': [i / 10.0 for i in range(0, 10)],
        # 'hidden_noise': [i / 10.0 for i in range(0, 10)],
        'activ_func': 'relu',
        'learning_rate': 0.004,
        'momentum': 0.6,
        'patience': 3,
        'min_improvement': 0.1,
        # 'validate_every': range(1, 25),
        # 'batch_size': range(10, 50, 10),
        # 'hidden_dropout': [i * 0.1 for i in range(0, 10)],
        # 'input_dropout': [i * 0.1 for i in range(0, 10)]
    }

    results = defaultdict(list)

    for _ in range(50):
        x_train, x_test, y_train, y_test, _, labels_test = \
            model_selection.train_test_split(x_data, y_data, labels,
                                             test_size=0.1)

        regressor = Regressor(x_train, y_train)
        regressor.train(hidden_layers=[60, 60], hyperparams=hyperparams)
        y_preds = regressor.predict(x_test)

        for tup in zip(y_test, y_preds, labels_test):
            results[(tup[2], tup[0])].append(tup[1])

    return results


def _preprocess(df):
    '''Preprocess data.'''
    df['r1'] = _align(df['r1'])
    df['r2'] = _align(df['r2'])


def _align(col):
    '''Perform multiple sequence alignment.'''
    # TODO: Implement multiple sequence alignment,
    return col


def _encode_seqs(seqs):
    '''Encodes x data.'''
    encoding = {'A': [1, 0, 0, 0],
                'C': [0, 1, 0, 0],
                'G': [0, 0, 1, 0],
                'T': [0, 0, 0, 0],
                '-': [0.25, 0.25, 0.25, 0.25]}

    return [[item for sublist in [encoding[nucl] for nucl in seq]
             for item in sublist]
            for seq in seqs]


def _output(results):
    '''Output results.'''
    _, _, r_value, _, _ = \
        scipy.stats.linregress([key[1] for key in results.keys()],
                               [np.mean(pred) for pred in results.values()])

    print
    print
    print '--------'
    print 'R squared: %.3f' % r_value
    print

    res = zip([key[0] for key in results.keys()],
              [key[1] for key in results.keys()],
              [np.mean(pred) for pred in results.values()],
              [np.std(pred) for pred in results.values()])

    res.sort(key=lambda x: x[2], reverse=True)

    for result in res:
        print '\t'.join([str(r) for r in result])

    _plot(results)


def _plot(results):
    '''Plot results.'''
    import matplotlib.pyplot as plt

    plt.title('Prediction of limonene production from RBS seqs')
    plt.xlabel('Measured')
    plt.ylabel('Predicted')

    plt.errorbar([key[1] for key in results.keys()],
                 [np.mean(pred) for pred in results.values()],
                 yerr=[np.std(pred) for pred in results.values()],
                 fmt='o',
                 color='red')

    fit = np.poly1d(np.polyfit([key[1] for key in results.keys()],
                               [np.mean(pred)
                                for pred in results.values()], 1))

    plt.plot([key[1] for key in results.keys()],
             fit([key[1] for key in results.keys()]), 'r')

    plt.xlim(0, 1.6)
    plt.ylim(0, 1.6)

    plt.show()


def main(args):
    '''main method.'''
    x_data, y_data, labels = get_data(args[0])
    results = learn(x_data, y_data, labels)
    _output(results)


if __name__ == '__main__':
    main(sys.argv[1:])
