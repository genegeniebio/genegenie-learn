'''
sbclearn (c) University of Manchester 2017

sbclearn is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=invalid-name
from collections import defaultdict
import sys

from sklearn import model_selection, preprocessing

from sbclearn.theanets.theanets_utils import Regressor
import numpy as np
import pandas as pd
import sbclearn


def get_data(filename):
    '''Gets data.'''
    df = _preprocess(pd.read_table(filename))
    df = set_objective(df)

    # print df
    # df.to_csv('csv.txt')

    x_data = np.array(sbclearn.get_aa_props(df['Sequence'].tolist()))
    y_data = df['obj']
    labels = df['Sequence']

    return x_data, y_data, labels


def learn(x_data, y_data):
    '''Learn.'''
    results = defaultdict(list)

    for _ in range(50):
        x_train, x_test, y_train, y_test = \
            model_selection.train_test_split(x_data, y_data, test_size=0.05)

        regressor = Regressor(x_train, y_train)
        regressor.train(hidden_layers=[100])
        y_preds = regressor.predict(x_test)

        for test, pred in zip(y_test, y_preds):
            results[test].append(pred[0])

    for key, value in results.iteritems():
        print str(key) + '\t' + str(np.mean(value))

    _plot(results)


def _preprocess(df):
    '''Format data.'''

    # Liquid required at pH 4:
    df.loc[df['Physical state pH4'] == 'L', 'Physical state pH4'] = 0.0
    df.loc[df['Physical state pH4'] == 'V', 'Physical state pH4'] = 0.5
    df.loc[df['Physical state pH4'] == 'G', 'Physical state pH4'] = 1.0

    # Gel required at pH 7:
    df.loc[df['Physical state pH7'] == 'G', 'Physical state pH7'] = 0.0
    df.loc[df['Physical state pH7'] == 'V', 'Physical state pH7'] = 0.5
    df.loc[df['Physical state pH7'] == 'L', 'Physical state pH7'] = 0.75
    df.loc[df['Physical state pH7'] == 'P', 'Physical state pH7'] = 1.0

    # Normalise:
    num = preprocessing.MinMaxScaler(
        feature_range=(0.1, 0.9)).fit_transform(df.ix[:, 2:])

    # Maximise:
    df_num = pd.DataFrame(num)
    df_num = df_num.applymap(lambda x: 1 - x)

    # Reform DataFrame:
    df_num.columns = df.ix[:, 2:].columns
    df = pd.concat([df.ix[:, :2], df_num], axis=1)

    return df


def _plot(results):
    '''Plot results.'''
    import matplotlib.pyplot as plt

    plt.title('Prediction of peptide fitness')
    plt.xlabel('Measured')
    plt.ylabel('Predicted')

    plt.errorbar(results.keys(),
                 [np.mean(pred) for pred in results.values()],
                 yerr=[np.std(pred) for pred in results.values()],
                 fmt='o',
                 color='red')

    fit = np.poly1d(np.polyfit(results.keys(),
                               [np.mean(pred)
                                for pred in results.values()], 1))

    plt.plot(results.keys(),
             fit(results.keys()), 'r')

    plt.show()


def set_objective(df):
    '''Set composite objective.'''
    df['obj'] = df.ix[:, 2:].prod(axis=1)
    return df


def main(args):
    '''main method.'''
    x_data, y_data, _ = get_data(args[0])
    learn(x_data, y_data)


if __name__ == '__main__':
    main(sys.argv[1:])
