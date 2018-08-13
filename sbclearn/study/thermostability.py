'''
(c) GeneGenie Bioinformatics Ltd. 2018

Licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=invalid-name
from sklearn.model_selection import train_test_split

import pandas as pd
from sbclearn.keras import get_classify_model
from sbclearn.utils.biochem_utils import fasta_to_df, get_ordinal_seq
from sbclearn.utils.plot_utils import plot_stats


def get_data():
    '''Get data.'''
    pos_df = fasta_to_df('data/thermostability/h.txt')
    neg_df = fasta_to_df('data/thermostability/l.txt')

    pos_df['thermostability'] = 1.0
    neg_df['thermostability'] = 0.0

    df = pd.concat([pos_df, neg_df])
    df = df.sample(frac=1)

    return df


def classify(X_train, X_test, y_train, y_test, batch_size=200, epochs=25):
    '''Classify.'''
    model = get_classify_model(X_train.shape[1])
    # print(model.summary())

    stats = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                      validation_split=0.1)

    plot_stats(stats.history, 'stats.svg', 'acc')

    return model.evaluate(X_test, y_test, verbose=0)


def main():
    '''main method.'''
    df = get_data()

    X = get_ordinal_seq(df['seq'])
    y = df['thermostability']

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.1)

    scores = classify(X_train, X_test, y_train, y_test)

    print('Accuracy: %.2f%%' % (scores[1] * 100))


if __name__ == '__main__':
    main()
