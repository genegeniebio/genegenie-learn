'''
sbclearn (c) University of Manchester 2018

sbclearn is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=invalid-name
from synbiochem.utils import seq_utils

import pandas as pd


def align(df, sources=None):
    '''Align.'''
    # Filter rows:
    if sources:
        df = df.loc[df['source'].isin(sources)]

    # Perform Clustal Omega alignment:
    df['align_seq'] = \
        pd.Series(seq_utils.do_clustal(df.to_dict()['seq']))

    # Strip out positions with identical residues:
    aas = [list(seq) for seq in df['align_seq']]

    char_df = pd.DataFrame(aas,
                           columns=['pos_' + str(val)
                                    for val in range(0, len(aas[0]))],
                           index=df.index)
    char_df = char_df.loc[:, (char_df != char_df.iloc[0]).any()]

    df['dif_align_seq'] = char_df.apply(''.join, axis=1)

    return df
