'''
sbclearn (c) University of Manchester 2017

sbclearn is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=invalid-name
import os
import sys

from synbiochem.utils import seq_utils, xl_converter

import pandas as pd


def get_data(xl_filename):
    '''Get data.'''
    dir_name = xl_converter.convert(xl_filename)
    dfs = []

    for dirpath, _, filenames in os.walk(dir_name):
        for filename in filenames:
            dfs.append(pd.read_csv(os.path.join(dirpath, filename)))

    df = pd.concat(dfs)
    df.set_index('id', inplace=True)
    df['seq'] = df['seq'].apply(lambda x: x.replace('*', ''))
    df['mutations'] = df['mutations'].apply(lambda x: '' if x == '[]' else x)

    return df.drop_duplicates()


def analyse(df):
    '''Analyse.'''

    # Perform Clustal Omega alignment:
    df['align_seq'] = pd.Series(seq_utils.do_clustal(df.to_dict()['seq']))

    # Strip out positions with identical residues:
    aas = [list(seq) for seq in df['align_seq']]
    char_df = pd.DataFrame(aas,
                           columns=['pos_' + str(val)
                                    for val in range(0, len(aas[0]))],
                           index=df.index)
    char_df = char_df.loc[:, (char_df != char_df.iloc[0]).any()]

    df['dif_align_seq'] = char_df.apply(''.join, axis=1)


def main(args):
    '''main method.'''
    df = get_data(args[0])
    analyse(df)
    df.to_csv('out.csv')


if __name__ == '__main__':
    main(sys.argv[1:])
