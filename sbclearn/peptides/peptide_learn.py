'''
sbclearn (c) University of Manchester 2017

sbclearn is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=invalid-name
from collections import defaultdict
import sys

from sklearn import preprocessing
from synbiochem.utils import seq_utils
import scipy.stats

from sbclearn.theanets.theanets_utils import Regressor
import numpy as np
import pandas as pd
import sbclearn


def preprocess(df):
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
    x_scaled = preprocessing.MinMaxScaler().fit_transform(df.ix[:, 2:])
    df_scaled = pd.DataFrame(x_scaled)
    df_scaled.columns = df.ix[:, 2:].columns
    df = pd.concat([df.ix[:, :2], df_scaled], axis=1)

    return df


def set_objective(df):
    '''Set composite objective.'''
    df['obj'] = df.ix[:, 2:].sum(axis=1)
    return df


def learn(df):
    '''Learn.'''
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

    # Validate:
    for _ in range(50):
        msk = np.random.rand(len(df)) < 0.95
        train = df[msk]
        test = df[~msk]

        regressor = Regressor(train['encoded_peptides'].tolist(),
                              [[y] for y in train['obj'].tolist()])
        regressor.train(hidden_layers=[20, 20, 20], hyperparams=hyperparams)

        y_preds, _ = regressor.predict(test['encoded_peptides'].tolist(),
                                       test['obj'].tolist())

        print test['obj']
        print y_preds


def main(args):
    '''main method.'''
    df = preprocess(pd.read_table(args[0]))
    df = set_objective(df)
    df['encoded_peptides'] = seq_utils.get_aa_props(df['Sequence'].tolist())
    learn(df)


if __name__ == '__main__':
    main(sys.argv[1:])
