'''
sbclearn (c) University of Manchester 2017

sbclearn is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=invalid-name
import sys

from sklearn import model_selection, preprocessing
from synbiochem.utils import seq_utils

from sbclearn.theanets.theanets_utils import Regressor
import numpy as np
import pandas as pd


def get_data(filename):
    '''Gets data.'''
    df = _preprocess(pd.read_table(filename))
    df = set_objective(df)

    # print df
    # df.to_csv('csv.txt')

    x_data = np.array(seq_utils.get_aa_props(df['Sequence'].tolist()))
    y_data = df['obj']
    labels = df['Sequence']

    return x_data, y_data, labels


def learn(x_data, y_data):
    '''Learn.'''
    for _ in range(50):
        x_train, x_test, y_train, y_test = \
            model_selection.train_test_split(x_data, y_data, test_size=0.05)

        regressor = Regressor(x_train, y_train)
        regressor.train(hidden_layers=[100, 50, 25])
        y_preds = regressor.predict(x_test)

        for tup in zip(y_test, y_preds):
            print tup


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
    num = preprocessing.MinMaxScaler().fit_transform(df.ix[:, 2:])

    # Maximise:
    df_num = pd.DataFrame(num)
    df_num = df_num.applymap(lambda x: 1 - x)

    # Reform DataFrame:
    df_num.columns = df.ix[:, 2:].columns
    df = pd.concat([df.ix[:, :2], df_num], axis=1)

    return df


def set_objective(df):
    '''Set composite objective.'''
    df['obj'] = df.ix[:, 2:].sum(axis=1)
    return df


def main(args):
    '''main method.'''
    x_data, y_data, _ = get_data(args[0])
    learn(x_data, y_data)


if __name__ == '__main__':
    main(sys.argv[1:])
