'''
sbclearn (c) University of Manchester 2017

sbclearn is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=too-many-arguments
from collections import defaultdict

from sklearn import model_selection
from synbiochem.utils import seq_utils

from sbclearn.theanets.utils import Classifier, Regressor


def k_fold_cross_valid((x_data, y_data),
                       regression=True,
                       tests=50,
                       test_size=0.05,
                       hidden_layers=None,
                       hyperparams=None):
    '''k-fold cross validation.'''
    results = defaultdict(list) if regression else []

    for _ in range(tests):
        _k_fold_cross_valid((x_data, y_data),
                            results,
                            regression,
                            test_size,
                            hidden_layers,
                            hyperparams)

    return results


def _k_fold_cross_valid((x_data, y_data),
                        results,
                        regression=True,
                        test_size=0.05,
                        hidden_layers=None,
                        hyperparams=None):
    '''k-fold cross validation.'''
    x_train, x_test, y_train, y_test = \
        model_selection.train_test_split(x_data, y_data,
                                         test_size=test_size)

    model = Regressor(x_train, y_train) \
        if regression \
        else Classifier(x_train, y_train)

    model.train(hidden_layers=hidden_layers, hyperparams=hyperparams)

    for test, pred in zip(y_test, model.predict(x_test)):
        if regression:
            results[test].append(pred[0])
        else:
            results.append((test, pred))
