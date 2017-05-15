'''
sbclearn (c) University of Manchester 2017

sbclearn is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
from collections import defaultdict
import csv
import itertools
import sys

import numpy
import scipy.stats

from sbclearn.theanets.theanets_utils import Regressor
import sbclearn


def get_data(filename):
    '''Gets data.'''
    data = _read_data(filename)

    x_data = data.keys()
    y_data = [numpy.mean([val['val'] for val in vals])
              for vals in data.values()]

    variants = [list(set(val)) for val in zip(*x_data)]

    seqs = [''.join(vals) for vals in zip(*[val
                                            for val in zip(*x_data)
                                            if len(set(val)) > 1])]

    all_seqs = [''.join(combo)
                for combo in itertools.product(*[var
                                                 for var in variants
                                                 if len(var) > 1])]

    all_seqs = [seq for seq in all_seqs if seq not in seqs]

    x_data = [_encode_x_data(val) for val in seqs]
    x_all_data = [_encode_x_data(val) for val in all_seqs]

    return [x_data, y_data, seqs], [x_all_data, None, all_seqs], data


def _read_data(filename):
    '''Reads data.'''
    data = defaultdict(list)

    with open(filename, 'rU') as fle:
        reader = csv.DictReader(fle)

        for line in reader:
            data[line['sr1'] + line['sr2']].append({'id': line['Construct'],
                                                    'val': float(line['FC'])})
    return data


def _encode_x_data(x_data):
    '''Encodes x data.'''
    x_vals = {'A': (1, 0, 0, 0, 0),
              'C': (0, 1, 0, 0, 0),
              'G': (0, 0, 1, 0, 0),
              'T': (0, 0, 0, 1, 0),
              '-': (0, 0, 0, 0, 1)}

    return [val for nucl in x_data for val in x_vals[nucl]]


def _output(val_res, test_res):
    '''Output results.'''
    _, _, r_value, _, _ = \
        scipy.stats.linregress([key[2] for key in val_res.keys()],
                               [numpy.mean(pred) for pred in val_res.values()])

    print 'Validate'
    print '--------'
    print 'R squared: %.3f' % r_value
    print

    _print_results(val_res)

    print
    print
    print 'Test'
    print '----'
    print

    _print_results(test_res)

    _plot(val_res)


def _print_results(res):
    '''Prints results.'''
    results = zip([key[0] for key in res.keys()],
                  [', '.join(key[1]) for key in res.keys()],
                  [key[2] for key in res.keys()],
                  [numpy.mean(pred) for pred in res.values()],
                  [numpy.std(pred) for pred in res.values()])

    results.sort(key=lambda x: x[3], reverse=True)

    for result in results:
        print '\t'.join([str(res) for res in result])


def _plot(val_res):
    '''Plot results.'''
    import matplotlib.pyplot as plt

    plt.title('Prediction of limonene production from RBS seqs')
    plt.xlabel('Measured')
    plt.ylabel('Predicted')

    # Test results:
    # plt.errorbar([numpy.mean(pred) for pred in test_res],
    #             [numpy.mean(pred) for pred in test_res],
    #             yerr=[numpy.std(pred) for pred in test_res],
    #             fmt='o',
    #             color='red')

    # Validate results:
    plt.errorbar([key[2] for key in val_res.keys()],
                 [numpy.mean(pred) for pred in val_res.values()],
                 yerr=[numpy.std(pred) for pred in val_res.values()],
                 fmt='o',
                 color='black')

    fit = numpy.poly1d(numpy.polyfit([key[2] for key in val_res.keys()],
                                     [numpy.mean(pred)
                                      for pred in val_res.values()], 1))

    plt.plot([key[2] for key in val_res.keys()],
             fit([key[2] for key in val_res.keys()]), 'k')

    plt.xlim(0, 1.6)
    plt.ylim(0, 1.6)

    plt.show()


def main(args):
    '''main method.'''
    train_data, test_data, meta_data = get_data(args[0])

    hyperparams = {
        # 'aa_props_filter': range(1, (2**holygrail.NUM_AA_PROPS)),
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

    validate_results = defaultdict(list)
    test_results = defaultdict(list)

    # Validate:
    for _ in range(50):
        data_split = sbclearn.split_data(train_data, 0.9)

        regressor = Regressor(data_split[0][0],
                              [[val] for val in data_split[1][0]])

        regressor.train(hidden_layers=[60, 60], hyperparams=hyperparams)
        y_preds, _ = regressor.predict(data_split[0][1], data_split[1][1])

        for tup in zip(data_split[2][1], data_split[1][1], y_preds):
            validate_results[(tup[0], tuple([term['id']
                                             for term in meta_data[tup[0]]]),
                              tup[1])].append(tup[2])

    # Test:
    for _ in range(50):
        data_split = sbclearn.split_data(train_data, 1)

        regressor = Regressor(data_split[0][0],
                              [[val] for val in data_split[1][0]])

        regressor.train(hidden_layers=[60, 60], hyperparams=hyperparams)
        y_preds, _ = regressor.predict(test_data[0], None)

        for tup in zip(test_data[2], y_preds):
            test_results[(tup[0], (), -1)].append(tup[1])

    _output(validate_results, test_results)

if __name__ == '__main__':
    main(sys.argv[1:])
