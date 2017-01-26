'''
synbiochem (c) University of Manchester 2015

synbiochem is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
from collections import defaultdict
import numpy

from sbclearn.theanets.theanets_utils import Regressor
import sbclearn


def get_data(filename):
    '''Gets data.'''
    x_data = []
    y_data = []

    with open(filename, 'rU') as infile:
        for line in infile:
            tokens = line.strip().split('\t')
            x_data.append(tokens[0])

            if len(tokens) > 1:
                y_data.append(float(tokens[1]))

    x_data = [''.join(vals) for vals in zip(*[val
                                              for val in zip(*x_data)
                                              if len(set(val)) > 1])]
    seqs = x_data
    x_data = [_encode_x_data(val) for val in x_data]

    return [x_data, y_data], seqs


def _encode_x_data(x_data):
    '''Encodes x data.'''
    x_vals = {'A': (1, 0, 0, 0, 0),
              'C': (0, 1, 0, 0, 0),
              'G': (0, 0, 1, 0, 0),
              'T': (0, 0, 0, 1, 0),
              '-': (0, 0, 0, 0, 1)}

    return [val for nucl in x_data for val in x_vals[nucl]]


def _output(results, error):
    '''Output results.'''
    print 'Mean squared error: %.3f' % error

    for result in zip(results.keys(),
                      [numpy.mean(pred) for pred in results.values()],
                      [numpy.std(pred) for pred in results.values()]):
        print '\t'.join([str(res) for res in result])

    _plot(results)


def _plot(results):
    '''Plot results.'''
    import matplotlib.pyplot as plt

    plt.title('Prediction of limonene production from RBS seqs')
    plt.xlabel('Measured')
    plt.ylabel('Predicted')

    plt.errorbar(results.keys(),
                 [numpy.mean(pred) for pred in results.values()],
                 yerr=[numpy.std(pred) for pred in results.values()],
                 fmt='o',
                 color='black')

    fit = numpy.poly1d(numpy.polyfit(results.keys(),
                                     [numpy.mean(pred)
                                      for pred in results.values()], 1))
    plt.plot(results.keys(), fit(results.keys()), 'k')

    plt.xlim(0, 1.6)
    plt.ylim(0, 1.6)

    plt.show()


def main():
    '''main method.'''
    data, _ = get_data('rbs.txt')

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

    results = defaultdict(list)

    for _ in range(50):
        x_train, y_train, x_test, y_test = sbclearn.split_data(data, 0.98)
        regressor = Regressor(x_train, y_train)
        regressor.train(hidden_layers=[60, 60], hyperparams=hyperparams)
        results, error = regressor.predict(x_test, y_test, results=results)

    _output(results, error)

if __name__ == '__main__':
    main()
