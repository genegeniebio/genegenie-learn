'''
(c) GeneGenie Bioinformatics Ltd. 2018

Licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
import math

from sklearn.linear_model import LinearRegression

from gg_learn.utils import coeff_corr
import matplotlib.pyplot as plt


def plot_histogram(data, xlabel, ylabel, title):
    '''Plot histogram.'''
    plt.hist(data, 50, edgecolor='black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_stats(stats, filename, param):
    '''Plot neural network performance statistics.'''
    fig = plt.figure(figsize=(8, 4))

    # subplots:
    _subplot_stats(fig.add_subplot(121), stats, 'loss')

    if param:
        _subplot_stats(fig.add_subplot(122), stats, param)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def plot_linreg(x_train_pred, x_test_pred, y_train_true, y_test_true, i):
    '''Plot linear regression.'''
    axes = plt.gca()
    min_val = math.floor(min(min(y_train_true), min(y_test_true)))
    max_val = math.ceil(max(max(y_train_true), max(y_test_true)))
    axes.set_xlim([min_val, max_val])
    axes.set_ylim([min_val, max_val])

    plt.title('Linear regression of training prediction ' + str(i + 1))
    plt.xlabel('measured')
    plt.ylabel('predicted')

    # Plot train:
    _subplot(y_train_true, x_train_pred, 'gray', 0.1, 'r-square')

    # Plot test:
    _subplot(y_test_true, x_test_pred, 'red', 0.75, 'q-square')

    plt.legend(loc=2)

    plt.savefig('linear_regression_' + str(i + 1) + '.svg',
                bbox_inches='tight')
    plt.close()


def _subplot_stats(ax, stats, param):
    '''Plot subplot of stats.'''
    ax.plot(stats[param], label=param)
    ax.plot(stats['val_' + param], label='val_' + param)
    ax.set_xlabel('Epochs')
    ax.set_ylabel(param)
    ax.set_title(param)
    ax.legend()


def _subplot(y_true, y_pred, color, alpha, label):
    '''Subplot.'''
    plt.scatter(y_true, y_pred, c=color, alpha=alpha)

    model = LinearRegression()
    model.fit(y_pred.reshape(-1, 1), y_true)
    plt.plot(model.predict(y_pred.reshape(-1, 1)), y_pred,
             color=color,
             alpha=alpha,
             label=label +
             ': %.2f ' % coeff_corr(y_true, y_pred.reshape(len(y_pred))))
