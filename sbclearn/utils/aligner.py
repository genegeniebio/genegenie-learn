'''
sbclearn (c) University of Manchester 2018

sbclearn is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=invalid-name
import pandas as pd
from synbiochem.utils import seq_utils


def align(df):
    '''Align.'''

    # Perform Clustal Omega alignment:
    df['align_seq'] = \
        pd.Series(seq_utils.do_clustal(df.to_dict()['seq']))

    # Strip out positions with identical residues:
    df = df[df['align_seq'].notnull()]
    aas = [list(seq) for seq in df['align_seq']]

    char_df = pd.DataFrame(aas,
                           columns=['pos_' + str(val)
                                    for val in range(0, len(aas[0]))],
                           index=df.index)
    char_df = char_df.loc[:, (char_df != char_df.iloc[0]).any()]

    df['dif_align_seq'] = char_df.apply(''.join, axis=1)

    return df
