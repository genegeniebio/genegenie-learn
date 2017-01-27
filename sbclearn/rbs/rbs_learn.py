'''
synbiochem (c) University of Manchester 2015

synbiochem is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
from collections import defaultdict
import itertools
import sys

import numpy

from sbclearn.theanets.theanets_utils import Regressor
import sbclearn


def get_data(filename):
    '''Gets data.'''
    data = _read_data(filename)

    x_data = data.keys()
    y_data = [numpy.mean(vals) for vals in data.values()]

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

    return [x_data, y_data, seqs], [x_all_data, None, all_seqs]


def _read_data(filename):
    '''Reads data.'''
    data = defaultdict(list)

    with open(filename, 'rU') as infile:
        for line in infile:
            tokens = line.strip().split('\t')
            data[tokens[0]].append(float(tokens[1]))

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
    print 'Validate'
    print '--------'
    print 'R squared: %.3f' % (1 - val_res[1])
    print

    _print_results(val_res[2], val_res[0].keys(), val_res[0].values())

    print
    print
    print 'Test'
    print '----'
    print

    _print_results(test_res[2], [float('NaN')] * len(test_res[0]), test_res[0])

    _plot(val_res[0])


def _print_results(seqs, vals, preds):
    '''Prints results.'''
    results = zip(seqs,
                  vals,
                  [numpy.mean(pred) for pred in preds],
                  [numpy.std(pred) for pred in preds])

    results.sort(key=lambda x: x[2], reverse=True)

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
    plt.errorbar(val_res.keys(),
                 [numpy.mean(pred) for pred in val_res.values()],
                 yerr=[numpy.std(pred) for pred in val_res.values()],
                 fmt='o',
                 color='black')

    fit = numpy.poly1d(numpy.polyfit(val_res.keys(),
                                     [numpy.mean(pred)
                                      for pred in val_res.values()], 1))

    plt.plot(val_res.keys(), fit(val_res.keys()), 'k')

    plt.xlim(0, 1.6)
    plt.ylim(0, 1.6)

    plt.show()


def main(args):
    '''main method.'''
    train_data, test_data = get_data(args[0])

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
    test_results = []

    # Validate:
    for _ in range(250):
        x_train, y_train, x_test, y_test = sbclearn.split_data(train_data[:2],
                                                               0.9)
        regressor = Regressor(x_train, y_train)
        regressor.train(hidden_layers=[60, 60], hyperparams=hyperparams)
        validate_results, error = regressor.predict(x_test, y_test,
                                                    results=validate_results)

    # Test:
    for _ in range(250):
        x_train, y_train, _, _ = sbclearn.split_data(train_data[:2], 1)
        regressor = Regressor(x_train, y_train)
        regressor.train(hidden_layers=[60, 60], hyperparams=hyperparams)
        test_result, _ = regressor.predict(test_data[0], None)
        test_results.append(test_result)

    _output([validate_results, error, train_data[2]],
            [zip(*test_results), float('NaN'), test_data[2]])

if __name__ == '__main__':
    main(sys.argv[1:])
