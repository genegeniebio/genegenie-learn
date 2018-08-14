'''
(c) GeneGenie Bioinformatics Ltd. 2018

Licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=invalid-name
from gg_learn.keras import classify
from gg_learn.utils.biochem_utils import fasta_to_df, get_ordinal_seq
import pandas as pd


def get_data():
    '''Get data.'''
    pos_df = fasta_to_df('data/thermostability/h.txt')
    neg_df = fasta_to_df('data/thermostability/l.txt')

    pos_df['thermostability'] = 1.0
    neg_df['thermostability'] = 0.0

    df = pd.concat([pos_df, neg_df])
    df = df.sample(frac=1)

    return df


def main():
    '''main method.'''
    df = get_data()

    X = get_ordinal_seq(df['seq'])
    y = df['thermostability']

    scores = classify(X, y)

    print('Accuracy: %.2f%%' % (scores[1] * 100))


if __name__ == '__main__':
    main()
