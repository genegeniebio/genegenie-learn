'''
sbclearn (c) University of Manchester 2017

sbclearn is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=no-member
# pylint: disable=ungrouped-imports
import sys

from sbclearn.ccs import chem
from sbclearn.theanets import theanets_utils
import numpy as np
import pandas as pd
import sbclearn


def get_data(filename):
    '''Gets data.'''
    df = pd.read_csv(filename)

    x_data = np.array([chem.get_fingerprint(smiles, radius=8)
                       for smiles in df.SMILES])

    return x_data, df.CCS


def main(args):
    '''main method.'''
    x_data, y_data = get_data(args[0])
    results = theanets_utils.k_fold_cross_valid((x_data, y_data))
    sbclearn.plot(results, 'Prediction of ccs')


if __name__ == '__main__':
    main(sys.argv[1:])
