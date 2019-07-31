'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=wrong-import-order
from rdkit.Chem import MolFromSmiles

from liv_learn.keras import k_folds_nn
from liv_learn.utils.biochem_utils import get_tensor_chem
import pandas as pd


def get_data(path):
    '''Get data.'''
    df = pd.read_csv(path)
    df['mol'] = df['SMILES'].apply(MolFromSmiles)
    return df


def main():
    '''main method.'''
    df = get_data('data/solubility/delaney.csv')

    train_x = get_tensor_chem(df['mol'], 1024, 4)
    train_y = df['measured log(solubility:mol/L)'].values

    # Plot the histogram:
    # _plot_histogram(train_y, 'solubility', 'count', 'Solubility histogram')

    # Learn:
    k_folds_nn(train_x, train_y.T, 'coeff_determ', n_splits=10)


if __name__ == '__main__':
    main()
