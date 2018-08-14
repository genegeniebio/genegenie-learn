'''
sbclearn (c) University of Manchester 2017

sbclearn is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
import os

from synbiochem.utils import xl_converter

from gg_learn.utils import aligner
import numpy as np
import pandas as pd


def get_aligned_data(df):
    '''Get data.'''
    df = aligner.align(df)
    df.to_csv('aligned.csv')

    learn_df = df.loc[:, ['dif_align_seq', 'geraniol']]
    learn_df.columns = ['seq', 'activity']
    learn_df.to_csv('learn.csv')
    learn_df.dropna(inplace=True)

    return learn_df.values


def get_data(xl_filename, sources):
    '''Get raw data.'''
    dir_name = xl_converter.convert(xl_filename)
    dfs = []

    for dirpath, _, filenames in os.walk(dir_name):
        for filename in filenames:
            df = pd.read_csv(os.path.join(dirpath, filename))
            df['source'] = filename[:-4]
            dfs.append(df)

    df = pd.concat(dfs, sort=True)
    df.set_index('id', inplace=True)
    df = df[df['seq'].notnull()]
    df['seq'] = df['seq'].apply(lambda x: x.replace('*', ''))
    df['mutations'] = df['mutations'].apply(lambda x: '' if x == '[]' else x)

    # Filter rows:
    if sources:
        df = df.loc[df['source'].isin(sources)]

    # Return relative chemical production:
    text_df = df.select_dtypes(exclude=[np.number])
    num_df = df.select_dtypes(include=[np.number]).apply(
        lambda x: x / df.sum(axis=1))

    df = pd.concat([text_df, num_df], axis=1, sort=False)
    df = df.sample(frac=1)
    return df.drop_duplicates()
